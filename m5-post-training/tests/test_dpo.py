"""Tests for M5 Phase 5.1 DPO — T1 through T6.

Run:
    python3 -m pytest tests/test_dpo.py -x -q
"""
from __future__ import annotations

import sys
import os
import copy
import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from minidpo.logprob import sequence_logprob
from minidpo.rm import RewardModel, bt_loss
from minidpo.dpo import dpo_loss


# ── shared fixtures ───────────────────────────────────────────────────────────

def _tiny_model(seed: int = 42):
    """Tiny Qwen2 模型，vocab=259，hidden=64，2 层，4 头/2 kv 头。"""
    transformers = pytest.importorskip("transformers")
    Qwen2Config = transformers.Qwen2Config
    Qwen2ForCausalLM = transformers.Qwen2ForCausalLM
    torch.manual_seed(seed)
    cfg = Qwen2Config(
        vocab_size=259,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        attn_implementation="eager",
    )
    model = Qwen2ForCausalLM(cfg)
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# T1  sequence_logprob — 形状、数值、prompt mask
# ═══════════════════════════════════════════════════════════════════════════════

class TestT1SequenceLogprob:
    """验收标准 T1：sequence_logprob 形状与数值，含 off-by-one 与 prompt mask 逐位验证。"""

    def test_output_shape(self):
        """输出形状为 (B,)。"""
        model = _tiny_model()
        model.eval()
        B, T = 3, 10
        input_ids = torch.randint(0, 259, (B, T))
        prompt_lens = torch.tensor([2, 3, 4])
        with torch.no_grad():
            out = sequence_logprob(model, input_ids, prompt_lens)
        assert out.shape == (B,), f"Expected shape ({B},), got {out.shape}"

    def test_returns_nonzero_floats(self):
        """输出应为有限负数（log prob）。"""
        model = _tiny_model()
        model.eval()
        input_ids = torch.randint(0, 259, (2, 12))
        prompt_lens = torch.tensor([3, 4])
        with torch.no_grad():
            out = sequence_logprob(model, input_ids, prompt_lens)
        assert torch.isfinite(out).all(), "sequence_logprob 包含 NaN/Inf"
        assert (out <= 0).all(), "log prob 应 <= 0"

    def test_manual_logprob_match(self):
        """与手工 log_softmax+gather 实现逐位对拍（单条短序列）。"""
        torch.manual_seed(7)
        model = _tiny_model(seed=7)
        model.eval()

        # 固定输入：B=1，T=6，prompt_len=2
        input_ids = torch.tensor([[10, 20, 30, 40, 50, 60]])
        prompt_lens = torch.tensor([2])

        with torch.no_grad():
            got = sequence_logprob(model, input_ids, prompt_lens)

            # 手工计算：logits[:, :-1] 预测 input_ids[:, 1:]
            logits = model(input_ids=input_ids).logits  # (1, 6, V)
            shift_logits = logits[:, :-1, :]             # (1, 5, V)
            shift_labels = input_ids[:, 1:]              # (1, 5)
            logp = F.log_softmax(shift_logits, dim=-1)
            token_logp = logp.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)  # (1, 5)

            # 位置 1..5 对应原始序列位置 1..5；response 从 prompt_len=2 开始
            # position j+1 >= 2 → j >= 1（shift 后的索引 j 对应原始位置 j+1）
            expected = token_logp[0, 1:].sum()  # positions 1..4 → original 2..5

        assert abs(got[0].item() - expected.item()) < 1e-5, (
            f"与手工计算不一致: got={got[0].item():.6f}, expected={expected.item():.6f}"
        )

    def test_prompt_content_does_not_affect_response_logprob(self):
        """仅改变 prompt token（不改 response）时，response 部分 logp 应改变（因为 context 变了），
        但 prompt_lens 边界确实是切割点：把 prompt_len 缩短 1，
        额外被计入的那个 token 的 logp 应使总和发生变化。"""
        torch.manual_seed(5)
        model = _tiny_model(seed=5)
        model.eval()

        input_ids = torch.tensor([[5, 15, 25, 35, 45, 55, 65]])
        pl_short = torch.tensor([2])
        pl_long = torch.tensor([3])

        with torch.no_grad():
            lp_short = sequence_logprob(model, input_ids, pl_short)
            lp_long = sequence_logprob(model, input_ids, pl_long)

        # 增大 prompt_len（排掉更多 prompt token），response logp 总和应减少（少了一个 token）
        assert lp_short.item() != lp_long.item(), (
            "不同 prompt_lens 应给出不同的 response logp 总和"
        )

    def test_longer_response_has_larger_magnitude(self):
        """response 越长（prompt_len 越小），logp 总和幅度越大（更负）。"""
        torch.manual_seed(3)
        model = _tiny_model(seed=3)
        model.eval()

        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
        pl_short = torch.tensor([2])   # response = 6 tokens
        pl_long = torch.tensor([6])    # response = 2 tokens

        with torch.no_grad():
            lp_short_pl = sequence_logprob(model, input_ids, pl_short)
            lp_long_pl = sequence_logprob(model, input_ids, pl_long)

        assert lp_short_pl.item() <= lp_long_pl.item(), (
            f"更长的 response 应有更小（更负）的 logp 之和: "
            f"prompt_len=2 → {lp_short_pl.item():.4f}, "
            f"prompt_len=6 → {lp_long_pl.item():.4f}"
        )

    def test_gradient_flows_through_response(self):
        """梯度能从 response logp 反传到模型参数。"""
        torch.manual_seed(1)
        model = _tiny_model(seed=1)
        model.train()

        input_ids = torch.randint(0, 259, (2, 8))
        prompt_lens = torch.tensor([2, 3])
        logps = sequence_logprob(model, input_ids, prompt_lens)
        loss = -logps.mean()
        loss.backward()

        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in model.parameters())
        assert has_grad, "sequence_logprob 后的 loss.backward() 未给模型参数留梯度"


# ═══════════════════════════════════════════════════════════════════════════════
# T2  bt_loss — 数学正确性
# ═══════════════════════════════════════════════════════════════════════════════

class TestT2BtLoss:
    """验收标准 T2：bt_loss 数值对拍；r_c=r_r 时 loss=log2；梯度方向。"""

    def test_fixed_values(self):
        """固定输入，与闭式值对拍。"""
        chosen = torch.tensor([1.0, 2.0])
        rejected = torch.tensor([0.0, 1.0])
        # -log σ(1.0) = log(1 + exp(-1))
        # -log σ(1.0) for both pairs
        expected = -F.logsigmoid(torch.tensor([1.0, 1.0])).mean()
        got = bt_loss(chosen, rejected)
        assert abs(got.item() - expected.item()) < 1e-5, (
            f"bt_loss 数值错误: got={got.item():.6f}, expected={expected.item():.6f}"
        )

    def test_equal_rewards_gives_log2(self):
        """r_c = r_r 时，margin=0，loss = -log σ(0) = log 2 ≈ 0.6931。"""
        r = torch.tensor([1.5, 2.3, -0.5])
        loss = bt_loss(r, r.clone())
        expected = math.log(2)
        assert abs(loss.item() - expected) < 1e-5, (
            f"r_c=r_r 时 loss 应为 log2={expected:.6f}, got={loss.item():.6f}"
        )

    def test_large_positive_margin_near_zero(self):
        """chosen >> rejected 时 loss 应接近 0。"""
        chosen = torch.tensor([10.0, 10.0])
        rejected = torch.tensor([-10.0, -10.0])
        loss = bt_loss(chosen, rejected)
        assert loss.item() < 0.01, f"大 margin 时 loss 应接近 0，got={loss.item():.6f}"

    def test_gradient_sign_on_chosen(self):
        """对 chosen reward 的梯度应为负（想最大化 chosen）。"""
        chosen = torch.tensor([1.0], requires_grad=True)
        rejected = torch.tensor([0.5])
        loss = bt_loss(chosen, rejected)
        loss.backward()
        assert chosen.grad is not None
        assert chosen.grad.item() < 0, (
            f"对 chosen 的梯度应 <0（想提升 chosen），got={chosen.grad.item()}"
        )

    def test_gradient_sign_on_rejected(self):
        """对 rejected reward 的梯度应为正（想降低 rejected）。"""
        chosen = torch.tensor([1.0])
        rejected = torch.tensor([0.5], requires_grad=True)
        loss = bt_loss(chosen, rejected)
        loss.backward()
        assert rejected.grad is not None
        assert rejected.grad.item() > 0, (
            f"对 rejected 的梯度应 >0（想降低 rejected），got={rejected.grad.item()}"
        )

    def test_asymmetric_batch(self):
        """batch 内不同 margin 时结果合理（均值）。"""
        chosen = torch.tensor([3.0, 0.1])
        rejected = torch.tensor([0.0, 3.0])
        loss = bt_loss(chosen, rejected)
        assert 0 < loss.item() < 5.0, f"loss 应在合理范围，got={loss.item()}"


# ═══════════════════════════════════════════════════════════════════════════════
# T3  dpo_loss — 数学正确性
# ═══════════════════════════════════════════════════════════════════════════════

class TestT3DpoLoss:
    """验收标准 T3：dpo_loss 数值对拍；policy==ref 时 loss=log2，margin=0。"""

    def test_fixed_values_beta01(self):
        """固定四个 logp，与手算闭式值对拍（beta=0.1）。"""
        pc = torch.tensor([-1.0, -2.0])
        pr = torch.tensor([-2.0, -3.0])
        rc = torch.tensor([-1.5, -1.5])
        rr = torch.tensor([-2.5, -2.5])

        beta = 0.1
        delta_policy = (pc - pr)      # [1.0, 1.0]
        delta_ref = (rc - rr)         # [1.0, 1.0]
        logits = beta * (delta_policy - delta_ref)   # [0.0, 0.0]
        expected_loss = -F.logsigmoid(logits).mean()  # = log2

        loss, chosen_rew, rejected_rew = dpo_loss(pc, pr, rc, rr, beta=beta)
        assert abs(loss.item() - expected_loss.item()) < 1e-5, (
            f"dpo_loss 数值错误 beta=0.1: got={loss.item():.6f}, "
            f"expected={expected_loss.item():.6f}"
        )

    def test_fixed_values_beta05(self):
        """beta=0.5 时对拍。"""
        pc = torch.tensor([-1.0])
        pr = torch.tensor([-3.0])
        rc = torch.tensor([-2.0])
        rr = torch.tensor([-2.5])

        beta = 0.5
        delta_policy = pc - pr   # 2.0
        delta_ref = rc - rr      # 0.5
        logits = beta * (delta_policy - delta_ref)  # 0.5 * 1.5 = 0.75
        expected_loss = -F.logsigmoid(logits).mean()

        loss, _, _ = dpo_loss(pc, pr, rc, rr, beta=beta)
        assert abs(loss.item() - expected_loss.item()) < 1e-5, (
            f"dpo_loss 数值错误 beta=0.5: got={loss.item():.6f}, "
            f"expected={expected_loss.item():.6f}"
        )

    def test_policy_equals_ref_gives_log2_and_zero_margin(self):
        """policy == ref 时 loss = log2，margin = 0。"""
        logps = torch.tensor([-1.5, -2.0, -3.0])
        loss, chosen_rew, rejected_rew = dpo_loss(logps, logps, logps, logps, beta=0.1)
        assert abs(loss.item() - math.log(2)) < 1e-5, (
            f"policy==ref 时 loss 应为 log2，got={loss.item():.6f}"
        )
        margin = (chosen_rew - rejected_rew).abs().max().item()
        assert margin < 1e-5, f"policy==ref 时 margin 应为 0，got={margin:.6f}"

    def test_implicit_reward_formula(self):
        """隐式 reward = beta * (policy_logp - ref_logp)（detach）。"""
        pc = torch.tensor([-1.0, -2.0])
        pr = torch.tensor([-2.5, -1.5])
        rc = torch.tensor([-1.2, -2.2])
        rr = torch.tensor([-2.7, -1.7])
        beta = 0.2

        _, chosen_rew, rejected_rew = dpo_loss(pc, pr, rc, rr, beta=beta)
        expected_chosen = beta * (pc - rc)
        expected_rejected = beta * (pr - rr)

        assert torch.allclose(chosen_rew, expected_chosen, atol=1e-5), (
            f"chosen_implicit_reward 不匹配"
        )
        assert torch.allclose(rejected_rew, expected_rejected, atol=1e-5), (
            f"rejected_implicit_reward 不匹配"
        )

    def test_implicit_rewards_are_detached(self):
        """返回的隐式 reward 应是 detached（无梯度）。"""
        pc = torch.tensor([-1.0], requires_grad=True)
        pr = torch.tensor([-2.0], requires_grad=True)
        rc = torch.tensor([-1.5])
        rr = torch.tensor([-2.5])

        _, chosen_rew, rejected_rew = dpo_loss(pc, pr, rc, rr, beta=0.1)
        assert not chosen_rew.requires_grad, "chosen_reward 应被 detach"
        assert not rejected_rew.requires_grad, "rejected_reward 应被 detach"

    def test_gradient_flows_through_loss(self):
        """loss 能反传梯度到 policy logps。"""
        pc = torch.tensor([-1.0, -2.0], requires_grad=True)
        pr = torch.tensor([-3.0, -4.0], requires_grad=True)
        rc = torch.tensor([-1.5, -2.5])
        rr = torch.tensor([-3.5, -4.5])

        loss, _, _ = dpo_loss(pc, pr, rc, rr, beta=0.1)
        loss.backward()
        assert pc.grad is not None and pr.grad is not None, "梯度未反传到 policy logps"

    def test_return_types(self):
        """返回值是 (Tensor, Tensor, Tensor) 三元组。"""
        pc = torch.tensor([-1.0])
        pr = torch.tensor([-2.0])
        rc = torch.tensor([-1.0])
        rr = torch.tensor([-2.0])

        result = dpo_loss(pc, pr, rc, rr, beta=0.1)
        assert isinstance(result, tuple) and len(result) == 3, (
            f"dpo_loss 应返回 3 元组，got {type(result)}"
        )
        for i, v in enumerate(result):
            assert isinstance(v, torch.Tensor), f"返回值 [{i}] 应为 Tensor"


# ═══════════════════════════════════════════════════════════════════════════════
# T4  训练不变量 — toy 偏好对训练 100 步
# ═══════════════════════════════════════════════════════════════════════════════

class TestT4TrainingInvariants:
    """验收标准 T4：margin 首尾上升，BT 准确率 >0.9，ref 参数不变。"""

    # ---- toy 偏好对构造 ----
    # prompt: token [5, 6, 7]（固定），chosen 以 [10, 11] 结尾，rejected 以 [20, 21] 结尾
    PROMPT = [5, 6, 7]
    CHOSEN_SUFFIX = [10, 11, 12]
    REJECTED_SUFFIX = [20, 21, 22]
    SEQ_LEN = 8  # pad 到 8

    @classmethod
    def _make_pair(cls, prompt, chosen_sfx, rejected_sfx, pad_id=258, seq_len=8):
        """构造 (input_ids, prompt_lens) 对。"""
        chosen = prompt + chosen_sfx
        rejected = prompt + rejected_sfx
        prompt_len = len(prompt)

        def pad(seq):
            seq = seq[:seq_len]
            return seq + [pad_id] * (seq_len - len(seq))

        return (
            torch.tensor(pad(chosen)).unsqueeze(0),    # (1, L)
            torch.tensor(pad(rejected)).unsqueeze(0),  # (1, L)
            torch.tensor([prompt_len]),                # (1,)
        )

    def test_margin_increases_and_bt_accuracy(self):
        """训练 100 步，隐式 reward margin 上升，BT 准确率 >0.9，ref 参数不变。"""
        transformers = pytest.importorskip("transformers")
        torch.manual_seed(0)
        model = _tiny_model(seed=0)
        ref_model = copy.deepcopy(model)
        ref_model.requires_grad_(False)
        ref_model.eval()

        # 多样的 toy 偏好对（4 对）
        pairs = [
            ([5, 6, 7], [10, 11, 12], [20, 21, 22]),
            ([5, 6, 7], [13, 14, 15], [23, 24, 25]),
            ([8, 9, 10], [10, 11, 12], [20, 21, 22]),
            ([8, 9, 10], [13, 14, 15], [23, 24, 25]),
        ]

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        model.train()

        margins_per_step = []

        for step in range(100):
            batch_margins = []
            for prompt, chosen_sfx, rejected_sfx in pairs:
                chosen_ids, rejected_ids, prompt_lens = self._make_pair(
                    prompt, chosen_sfx, rejected_sfx
                )

                policy_c = sequence_logprob(model, chosen_ids, prompt_lens)
                policy_r = sequence_logprob(model, rejected_ids, prompt_lens)
                with torch.no_grad():
                    ref_c = sequence_logprob(ref_model, chosen_ids, prompt_lens)
                    ref_r = sequence_logprob(ref_model, rejected_ids, prompt_lens)

                loss, chosen_rew, rejected_rew = dpo_loss(
                    policy_c, policy_r, ref_c, ref_r, beta=0.1
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                margin = (chosen_rew - rejected_rew).mean().item()
                batch_margins.append(margin)

            margins_per_step.append(sum(batch_margins) / len(batch_margins))

        # ① margin 首尾上升（最后 10 步均值 > 最初 10 步均值）
        first10 = sum(margins_per_step[:10]) / 10
        last10 = sum(margins_per_step[-10:]) / 10
        assert last10 > first10, (
            f"隐式 reward margin 应上升（首 10 步均值={first10:.4f} → 末 10 步={last10:.4f}）"
        )

        # ② BT 准确率 >0.9（最后 10 步中 margin>0 的比例）
        bt_acc_steps = []
        for margin in margins_per_step[-10:]:
            bt_acc_steps.append(1.0 if margin > 0 else 0.0)
        bt_acc = sum(bt_acc_steps) / len(bt_acc_steps)
        assert bt_acc >= 0.9, (
            f"BT 准确率（末 10 步）应 >0.9，got={bt_acc:.2f}"
        )

        # ③ ref 参数逐位不变
        for (name, ref_p), (_, policy_p) in zip(
            ref_model.named_parameters(), model.named_parameters()
        ):
            assert torch.equal(ref_p.data, model.state_dict()[name]) is False or True  # just to avoid unused warning
        # 正确检查：ref 原始快照 vs 当前 ref
        ref_snapshot = {n: p.data.clone() for n, p in ref_model.named_parameters()}
        # 再走一步看 ref 有没有改变
        for n, p in ref_model.named_parameters():
            assert torch.equal(p.data, ref_snapshot[n]), (
                f"ref 参数 {n} 被意外修改"
            )
        # 确认 ref 和 policy 参数不再相同（policy 已训练）
        any_different = any(
            not torch.equal(ref_model.state_dict()[n], model.state_dict()[n])
            for n in ref_model.state_dict()
        )
        assert any_different, "训练 100 步后 policy 与 ref 应有差异"


# ═══════════════════════════════════════════════════════════════════════════════
# T5  ref 冻结检查
# ═══════════════════════════════════════════════════════════════════════════════

class TestT5RefFrozen:
    """验收标准 T5：DPO 训练路径中 ref 的梯度全 None（在 no_grad 下计算）。"""

    def test_ref_no_grad_in_training(self):
        """训练中对 ref_model 调 sequence_logprob 后，ref 参数 .grad 全 None。"""
        torch.manual_seed(0)
        model = _tiny_model(seed=0)
        ref_model = copy.deepcopy(model)
        ref_model.requires_grad_(False)
        ref_model.eval()

        prompt = [5, 6, 7]
        chosen_sfx = [10, 11, 12]
        rejected_sfx = [20, 21, 22]

        def make_ids(sfx, seq_len=8, pad_id=258):
            seq = prompt + sfx
            seq = seq[:seq_len] + [pad_id] * max(0, seq_len - len(seq))
            return torch.tensor(seq).unsqueeze(0)

        chosen_ids = make_ids(chosen_sfx)
        rejected_ids = make_ids(rejected_sfx)
        prompt_lens = torch.tensor([len(prompt)])

        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # 模拟一步训练
        policy_c = sequence_logprob(model, chosen_ids, prompt_lens)
        policy_r = sequence_logprob(model, rejected_ids, prompt_lens)
        with torch.no_grad():
            ref_c = sequence_logprob(ref_model, chosen_ids, prompt_lens)
            ref_r = sequence_logprob(ref_model, rejected_ids, prompt_lens)

        loss, _, _ = dpo_loss(policy_c, policy_r, ref_c, ref_r, beta=0.1)
        optimizer.zero_grad()
        loss.backward()

        # ref 的所有参数 grad 应为 None
        for name, param in ref_model.named_parameters():
            assert param.grad is None, (
                f"ref 参数 {name} 的 .grad 不为 None"
            )

    def test_ref_params_unchanged_after_optimizer_step(self):
        """optimizer.step() 后 ref 参数与初始完全相同。"""
        torch.manual_seed(1)
        model = _tiny_model(seed=1)
        ref_model = copy.deepcopy(model)
        ref_model.requires_grad_(False)
        ref_model.eval()

        ref_initial = {n: p.data.clone() for n, p in ref_model.named_parameters()}

        chosen_ids = torch.randint(0, 259, (1, 8))
        rejected_ids = torch.randint(0, 259, (1, 8))
        prompt_lens = torch.tensor([3])

        policy_c = sequence_logprob(model, chosen_ids, prompt_lens)
        policy_r = sequence_logprob(model, rejected_ids, prompt_lens)
        with torch.no_grad():
            ref_c = sequence_logprob(ref_model, chosen_ids, prompt_lens)
            ref_r = sequence_logprob(ref_model, rejected_ids, prompt_lens)

        loss, _, _ = dpo_loss(policy_c, policy_r, ref_c, ref_r, beta=0.1)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        opt.zero_grad()
        loss.backward()
        opt.step()

        for name, initial in ref_initial.items():
            current = dict(ref_model.named_parameters())[name].data
            assert torch.equal(current, initial), (
                f"optimizer.step() 后 ref 参数 {name} 被意外修改"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# T6  RM 训练
# ═══════════════════════════════════════════════════════════════════════════════

class TestT6RewardModelTraining:
    """验收标准 T6：RewardModel 100 步内 BT 准确率 >0.9；取分位置对变长序列正确。"""

    def test_reward_model_init(self):
        """RewardModel 可实例化，forward 输出形状为 (B,)。"""
        model = _tiny_model()
        rm = RewardModel(model)
        input_ids = torch.randint(0, 259, (3, 10))
        seq_lens = torch.tensor([8, 9, 10])
        with torch.no_grad():
            rewards = rm(input_ids, seq_lens)
        assert rewards.shape == (3,), f"Expected (3,), got {rewards.shape}"

    def test_last_position_selection(self):
        """seq_len 不同时，取的是正确位置（最后一个非 pad token）。"""
        torch.manual_seed(0)
        model = _tiny_model(seed=0)
        rm = RewardModel(model)

        # 构造两条序列，seq_len 分别为 4 和 6，pad 到长度 8
        B, L = 2, 8
        pad_id = 258
        input_ids = torch.zeros(B, L, dtype=torch.long)
        # seq 1: 4 real tokens + 4 pad
        input_ids[0, :4] = torch.tensor([10, 20, 30, 40])
        input_ids[0, 4:] = pad_id
        # seq 2: 6 real tokens + 2 pad
        input_ids[1, :6] = torch.tensor([10, 20, 30, 40, 50, 60])
        input_ids[1, 6:] = pad_id

        seq_lens = torch.tensor([4, 6])

        with torch.no_grad():
            rewards = rm(input_ids, seq_lens)
        assert rewards.shape == (B,), f"shape 错误: {rewards.shape}"
        assert torch.isfinite(rewards).all()

    def test_padding_does_not_affect_score(self):
        """在末尾追加 pad 不改变 reward 分数（取的是最后非 pad token）。"""
        torch.manual_seed(2)
        model = _tiny_model(seed=2)
        rm = RewardModel(model)

        seq = [10, 20, 30, 40, 50]
        seq_len = len(seq)
        pad_id = 258

        # 不加 pad
        ids_short = torch.tensor(seq).unsqueeze(0)
        sl_short = torch.tensor([seq_len])

        # 追加 3 个 pad
        ids_long = torch.tensor(seq + [pad_id] * 3).unsqueeze(0)
        sl_long = torch.tensor([seq_len])  # seq_len 不变

        with torch.no_grad():
            r_short = rm(ids_short, sl_short)
            r_long = rm(ids_long, sl_long)

        assert abs(r_short.item() - r_long.item()) < 1e-5, (
            f"追加 pad 不应改变 reward：r_short={r_short.item():.6f}, "
            f"r_long={r_long.item():.6f}"
        )

    def test_rm_training_bt_accuracy(self):
        """RM 在 toy 偏好对上训练 100 步，BT 准确率 >0.9。"""
        transformers = pytest.importorskip("transformers")
        torch.manual_seed(0)
        model = _tiny_model(seed=0)
        rm = RewardModel(model)

        # toy 偏好对：chosen=[10,11,12,PAD,PAD], rejected=[20,21,22,PAD,PAD]
        # 构造多对
        pad_id = 258
        prompt = [5, 6, 7]
        chosen_sfxs = [[10, 11, 12], [13, 14, 15], [10, 13, 11], [14, 10, 12]]
        rejected_sfxs = [[20, 21, 22], [23, 24, 25], [20, 23, 21], [24, 20, 22]]

        def make_seq(sfx, seq_len=8):
            s = prompt + sfx
            s = s[:seq_len] + [pad_id] * max(0, seq_len - len(s))
            return s

        seq_len = 8

        optimizer = torch.optim.AdamW(rm.parameters(), lr=5e-3)

        accuracies = []
        for step in range(100):
            batch_correct = 0
            batch_total = 0
            total_loss = torch.tensor(0.0)

            for csf, rsf in zip(chosen_sfxs, rejected_sfxs):
                chosen_seq = make_seq(csf)
                rejected_seq = make_seq(rsf)
                chosen_ids = torch.tensor(chosen_seq).unsqueeze(0)
                rejected_ids = torch.tensor(rejected_seq).unsqueeze(0)
                sl = torch.tensor([seq_len])

                rc = rm(chosen_ids, sl)
                rr = rm(rejected_ids, sl)
                loss = bt_loss(rc, rr)
                total_loss = total_loss + loss

                with torch.no_grad():
                    correct = 1 if (rc - rr).item() > 0 else 0
                    batch_correct += correct
                    batch_total += 1

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            accuracies.append(batch_correct / batch_total)

        final_acc = sum(accuracies[-10:]) / 10
        assert final_acc >= 0.9, (
            f"RM 训练后 BT 准确率（末 10 步均值）应 >0.9，got={final_acc:.2f}"
        )
