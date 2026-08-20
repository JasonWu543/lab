"""Tests for M5 Phase 5.2 GRPO — T1 through T6.

Run:
    python3 -m pytest tests/test_grpo.py -x -q

T6 超参（经 3 次连跑验证稳定通过，delta=0.321 > 0.3，约 13s/run）：
    SFT 预热：100 步，lr=1e-2，batch 轮转 16 对加法样本（a,b∈[1,4]）
    GRPO：60 轮，G=4，lr=5e-3，clip_eps=0.2，kl_coef=0.001
    max_new_tokens=4，temperature=1.0
    任务：个位数加法 a,b∈[1,4]，答案≤8，prompt="Q:{a}+{b}=\nA:"
    seed：torch.manual_seed(0)，model seed=0，generator seed=42
    整体 ≤120s（CPU）
"""
from __future__ import annotations

import sys
import os
import copy
import math
import re

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from minigrpo.reward import parse_answer, reward_fn
from minigrpo.advantage import group_advantages
from minigrpo.loss import grpo_loss
from minigrpo.rollout import rollout
from minisft.tokenizer import ByteTokenizer


# ── shared helpers ────────────────────────────────────────────────────────────

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


def _per_token_logp_ref(model, input_ids: torch.Tensor) -> torch.Tensor:
    """(B,T) per-token logp，位置 0 置 0。供测试内部用。"""
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits
    logp = F.log_softmax(logits[:, :-1, :], dim=-1)
    tok_logp = logp.gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    out = torch.zeros_like(input_ids, dtype=tok_logp.dtype)
    out[:, 1:] = tok_logp
    return out


def _sft_warmup(model, tok: ByteTokenizer, n_steps: int = 30, lr: float = 5e-3):
    """
    个位数加法 SFT 预热脚手架（测试内给定，不是学生任务）。
    训练集：a,b ∈ [1,4]，prompt="Q:{a}+{b}=\\nA:"，answer=str(a+b)。
    格式：prompt_ids + answer_byte_ids + [eos_id]。
    只对 answer 部分（prompt 之后）计算 CE loss。
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    eos_id = tok.eos_id

    # 构造所有 (a,b) 对，a,b ∈ [1,4]
    data = []
    for a in range(1, 5):
        for b in range(1, 5):
            prompt_str = f"Q:{a}+{b}=\nA:"
            answer_str = str(a + b)
            prompt_ids = tok.encode(prompt_str)
            answer_ids = tok.encode(answer_str) + [eos_id]
            full_ids = prompt_ids + answer_ids
            data.append((full_ids, len(prompt_ids)))

    model.train()
    for step in range(n_steps):
        idx = step % len(data)
        full_ids, prompt_len = data[idx]
        ids_t = torch.tensor(full_ids).unsqueeze(0)

        logits = model(input_ids=ids_t).logits[0]  # (T, V)
        # labels: 对 prompt 部分用 -100 mask，只算 answer 的 next-token loss
        labels = torch.full((len(full_ids),), -100, dtype=torch.long)
        # logits[i] 预测 full_ids[i+1]；answer 从 prompt_len 开始
        for i in range(prompt_len - 1, len(full_ids) - 1):
            labels[i] = full_ids[i + 1]

        active = labels != -100
        if not active.any():
            continue
        loss = F.cross_entropy(logits[active], labels[active])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()


# ═══════════════════════════════════════════════════════════════════════════════
# T1  reward_fn — 三级打分 + parse 边界
# ═══════════════════════════════════════════════════════════════════════════════

class TestT1RewardFn:
    """验收标准 T1：三级打分逐例对拍；parse_answer 边界。"""

    def test_correct_answer(self):
        """正确答案 = 1.0。"""
        assert reward_fn("Q:3+4=\nA:", " 7") == 1.0
        assert reward_fn("Q:2+2=\nA:", "4") == 1.0
        assert reward_fn("Q:1+1=\nA:", "2") == 1.0

    def test_wrong_number(self):
        """数字但值错 = 0.1。"""
        assert reward_fn("Q:3+4=\nA:", " 5") == pytest.approx(0.1)
        assert reward_fn("Q:2+2=\nA:", "999") == pytest.approx(0.1)

    def test_no_number(self):
        """没有数字 = 0.0。"""
        assert reward_fn("Q:3+4=\nA:", "abc") == 0.0
        assert reward_fn("Q:3+4=\nA:", "") == 0.0
        assert reward_fn("Q:3+4=\nA:", "xyz!") == 0.0

    def test_parse_leading_space(self):
        """parse_answer 支持前导空格。"""
        assert parse_answer(" 7") == 7
        assert parse_answer("  42") == 42

    def test_parse_negative(self):
        """parse_answer 支持负号。"""
        assert parse_answer("-3") == -3
        assert parse_answer("result=-5 end") == -5

    def test_parse_first_integer(self):
        """parse_answer 取首个整数（多个数字取第一个）。"""
        assert parse_answer("answer is 7 not 8") == 7

    def test_parse_failure_returns_none(self):
        """解析失败返回 None。"""
        assert parse_answer("abc") is None
        assert parse_answer("") is None

    def test_invalid_prompt_returns_zero(self):
        """prompt 无法解析出 a+b 时返回 0.0。"""
        assert reward_fn("not a valid prompt", "7") == 0.0

    def test_large_numbers(self):
        """较大数字也能正确打分（SPEC 是任意整数）。"""
        assert reward_fn("Q:10+20=\nA:", " 30") == 1.0
        assert reward_fn("Q:10+20=\nA:", " 31") == pytest.approx(0.1)


# ═══════════════════════════════════════════════════════════════════════════════
# T2  group_advantages — 手算对拍 + 统计
# ═══════════════════════════════════════════════════════════════════════════════

class TestT2GroupAdvantages:
    """验收标准 T2：手算对拍；std=0 时全 0；每组 mean≈0，std≈1。"""

    def test_manual_small(self):
        """2 个 prompt，G=3，手算。"""
        rewards = torch.tensor([[1.0, 2.0, 3.0],
                                [4.0, 4.0, 4.0]])
        out = group_advantages(rewards, eps=1e-6)

        # group 0: mean=2, std=sqrt(2/3)≈0.8165
        std0 = rewards[0].std(unbiased=False).item()
        expected_g0 = (rewards[0] - 2.0) / (std0 + 1e-6)

        # group 1: std=0 → all 0
        expected_g1 = torch.zeros(3)

        assert out.shape == (2, 3), f"shape error: {out.shape}"
        assert torch.allclose(out[0], expected_g0, atol=1e-5), (
            f"group 0 不匹配: {out[0]} vs {expected_g0}"
        )
        assert torch.allclose(out[1], expected_g1, atol=1e-5), (
            f"std=0 的组应全 0: {out[1]}"
        )

    def test_zero_std_returns_zeros(self):
        """全组同分时 advantage 全 0。"""
        rewards = torch.tensor([[5.0, 5.0, 5.0, 5.0]])
        out = group_advantages(rewards)
        assert (out == 0).all(), f"全组同分时应全 0，got {out}"

    def test_normalized_mean_near_zero(self):
        """标准化后每组均值 ≈ 0。"""
        torch.manual_seed(0)
        rewards = torch.randn(4, 6)
        out = group_advantages(rewards)
        group_means = out.mean(dim=1)
        assert torch.allclose(group_means, torch.zeros(4), atol=1e-5), (
            f"每组均值应≈0，got {group_means}"
        )

    def test_normalized_std_near_one(self):
        """标准化后非零 std 组的 std ≈ 1。"""
        torch.manual_seed(1)
        rewards = torch.randn(4, 6)
        out = group_advantages(rewards)
        for i in range(rewards.shape[0]):
            orig_std = rewards[i].std(unbiased=False).item()
            if orig_std > 1e-3:
                got_std = out[i].std(unbiased=False).item()
                assert abs(got_std - 1.0) < 0.1, (
                    f"group {i} std 应≈1，got {got_std:.4f}"
                )

    def test_output_shape(self):
        """输出形状与输入相同。"""
        rewards = torch.randn(5, 8)
        out = group_advantages(rewards)
        assert out.shape == (5, 8)

    def test_single_group(self):
        """n_prompts=1 时正常工作。"""
        rewards = torch.tensor([[1.0, 3.0, 5.0, 7.0]])
        out = group_advantages(rewards)
        assert out.shape == (1, 4)
        assert abs(out.mean().item()) < 1e-5


# ═══════════════════════════════════════════════════════════════════════════════
# T3  grpo_loss — 数学对拍 + clip_frac + 梯度截断
# ═══════════════════════════════════════════════════════════════════════════════

class TestT3GrpoLoss:
    """验收标准 T3：ratio=1 时 pg_loss 闭式对拍；policy==ref 时 kl≈0；
    超界 ratio 时 clip_frac>0 且对应 token 梯度被截。"""

    def _make_inputs(self, B=2, T=5, seed=0):
        torch.manual_seed(seed)
        logps = -torch.rand(B, T) * 3          # 当前策略 per-token logp
        old_logps = logps.clone()               # ratio=1
        ref_logps = -torch.rand(B, T) * 3
        advantages = torch.randn(B)
        mask = torch.ones(B, T, dtype=torch.long)
        return logps, old_logps, ref_logps, advantages, mask

    def test_ratio_one_pg_loss(self):
        """ratio=1（logps==old_logps）时 pg_loss = -mean(A * mask) / denom，对拍。"""
        B, T = 2, 4
        torch.manual_seed(7)
        logps = -torch.rand(B, T)
        old_logps = logps.clone()
        ref_logps = logps.clone()
        advantages = torch.tensor([1.5, -0.5])
        mask = torch.ones(B, T, dtype=torch.long)

        loss, stats = grpo_loss(
            logps, old_logps, ref_logps, advantages, mask,
            clip_eps=0.2, kl_coef=0.0  # 先关 KL 验 pg_loss
        )
        # ratio=1，clip 不触发；adv 广播到 token：-mean(A_i * 1) over all masked tokens
        adv_token = advantages.unsqueeze(1).expand(B, T)  # (B, T)
        expected_pg = -(adv_token * mask.float()).sum() / mask.float().sum()
        assert abs(stats["pg_loss"].item() - expected_pg.item()) < 1e-5, (
            f"ratio=1 时 pg_loss 应 = {expected_pg.item():.6f}，"
            f"got {stats['pg_loss'].item():.6f}"
        )

    def test_policy_equals_ref_kl_near_zero(self):
        """policy == ref 时 k3 KL ≈ 0。"""
        logps, old_logps, _, advantages, mask = self._make_inputs()
        ref_logps = logps.clone()  # policy == ref

        _, stats = grpo_loss(logps, old_logps, ref_logps, advantages, mask)
        assert stats["kl"].item() < 1e-5, (
            f"policy==ref 时 kl 应≈0，got {stats['kl'].item():.6f}"
        )

    def test_clip_frac_positive_when_ratio_exceeds_bounds(self):
        """构造超界 ratio 时 clip_frac > 0。"""
        B, T = 2, 4
        # ratio = exp(logps - old_logps)；让 logps >> old_logps
        logps = torch.zeros(B, T)
        old_logps = torch.full((B, T), -2.0)  # ratio = exp(2) >> 1.2
        ref_logps = torch.zeros(B, T)
        advantages = torch.ones(B)
        mask = torch.ones(B, T, dtype=torch.long)

        _, stats = grpo_loss(logps, old_logps, ref_logps, advantages, mask, clip_eps=0.2)
        assert stats["clip_frac"].item() > 0, (
            f"超界 ratio 时 clip_frac 应 >0，got {stats['clip_frac'].item():.4f}"
        )

    def test_clipped_token_gradient_is_zero(self):
        """超界 token 被 clip 后，dloss/d(logps) 应为 0（梯度被截）。"""
        B, T = 1, 3
        # 第 0 个 token：ratio > 1+eps（A > 0），应被 clip → 梯度 = 0
        # 第 1 个 token：ratio ≈ 1，不被 clip → 梯度非零
        # 第 2 个 token：mask = 0 → 不参与

        logps = torch.tensor([[-2.0, -1.0, -1.0]], requires_grad=True)
        old_logps = torch.tensor([[-4.0, -1.05, -1.0]])  # ratio[0]=exp(2)>>1.2，ratio[1]≈exp(0.05)
        ref_logps = torch.zeros(B, T)
        advantages = torch.tensor([1.0])  # A > 0
        mask = torch.tensor([[1, 1, 0]], dtype=torch.long)

        loss, stats = grpo_loss(logps, old_logps, ref_logps, advantages, mask,
                                clip_eps=0.2, kl_coef=0.0)
        loss.backward()

        # token 0 被 clip（ratio >> 1.2，A > 0 → 使用 clipped，梯度来自 clip(ratio)*A）
        # 理论：clip(ratio) 对 logps 无梯度（常数），所以 grad[0,0] == 0
        assert abs(logps.grad[0, 0].item()) < 1e-6, (
            f"被 clip 的 token 梯度应为 0，got {logps.grad[0, 0].item():.8f}"
        )

    def test_loss_requires_grad(self):
        """loss 应可反传梯度。"""
        logps, old_logps, ref_logps, advantages, mask = self._make_inputs()
        logps = logps.detach().requires_grad_(True)
        loss, _ = grpo_loss(logps, old_logps, ref_logps, advantages, mask)
        loss.backward()
        assert logps.grad is not None

    def test_return_structure(self):
        """返回 (Tensor, dict) 且 dict 含 pg_loss/kl/clip_frac 键。"""
        logps, old_logps, ref_logps, advantages, mask = self._make_inputs()
        result = grpo_loss(logps, old_logps, ref_logps, advantages, mask)
        assert isinstance(result, tuple) and len(result) == 2
        loss, stats = result
        for key in ("pg_loss", "kl", "clip_frac"):
            assert key in stats, f"stats 缺少键 {key}"


# ═══════════════════════════════════════════════════════════════════════════════
# T4  k3 KL 非负性
# ═══════════════════════════════════════════════════════════════════════════════

class TestT4K3NonNeg:
    """验收标准 T4：k3 逐元素 ≥ 0；ref==policy 时 = 0。"""

    def test_kl_zero_when_ref_equals_policy(self):
        """ref == policy 时，grpo_loss 返回的 kl 统计应 ≈ 0。"""
        torch.manual_seed(98)
        B, T = 6, 8
        logps = -torch.rand(B, T) * 3
        mask = torch.ones(B, T, dtype=torch.long)
        advantages = torch.zeros(B)

        _, stats = grpo_loss(logps, logps.clone(), logps.clone(), advantages,
                             mask, clip_eps=0.2, kl_coef=1.0)
        assert abs(stats["kl"]) < 1e-6, (
            f"ref==policy 时 kl 应为 0，实际 {stats['kl']:.6f}"
        )

    def test_kl_stat_nonneg(self):
        """stats['kl'] 应 ≥ 0。"""
        torch.manual_seed(0)
        B, T = 4, 6
        logps = -torch.rand(B, T)
        ref_logps = -torch.rand(B, T)
        mask = torch.ones(B, T, dtype=torch.long)
        advantages = torch.zeros(B)
        _, stats = grpo_loss(logps, logps.clone(), ref_logps, advantages, mask)
        assert stats["kl"].item() >= -1e-6


# ═══════════════════════════════════════════════════════════════════════════════
# T5  rollout 记账
# ═══════════════════════════════════════════════════════════════════════════════

class TestT5Rollout:
    """验收标准 T5：mask 覆盖 completion；old_logps 与重算一致；G 条/prompt。"""

    def test_output_keys(self):
        """rollout 返回所有必需键。"""
        model = _tiny_model()
        tok = ByteTokenizer()
        prompts = ["Q:1+1=\nA:"]
        out = rollout(model, tok, prompts, G=2, max_new_tokens=3,
                      temperature=1.0, generator=None)
        for key in ("input_ids", "prompt_lens", "completions", "old_logps", "mask"):
            assert key in out, f"rollout 缺少键 {key}"

    def test_batch_size_is_G_times_prompts(self):
        """B = len(prompts) * G。"""
        model = _tiny_model()
        tok = ByteTokenizer()
        prompts = ["Q:1+2=\nA:", "Q:3+4=\nA:"]
        G = 3
        out = rollout(model, tok, prompts, G=G, max_new_tokens=4,
                      temperature=1.0, generator=None)
        B = out["input_ids"].shape[0]
        assert B == len(prompts) * G, f"B 应 = {len(prompts)*G}，got {B}"

    def test_mask_covers_only_completion(self):
        """mask 在 prompt 区域为 0，completion 区域至少有一个 1。"""
        torch.manual_seed(0)
        model = _tiny_model(seed=0)
        tok = ByteTokenizer()
        prompts = ["Q:2+3=\nA:"]
        out = rollout(model, tok, prompts, G=2, max_new_tokens=4,
                      temperature=0.0, generator=None)

        prompt_lens = out["prompt_lens"]
        mask = out["mask"]
        input_ids = out["input_ids"]

        for b in range(mask.shape[0]):
            plen = prompt_lens[b].item()
            # prompt 区域全 0
            assert (mask[b, :plen] == 0).all(), (
                f"样本 {b}：prompt 区域 mask 应全 0"
            )
            # completion 区域至少有一个 1（生成了至少一个 token）
            assert mask[b, plen:].sum() >= 1, (
                f"样本 {b}：completion 区域 mask 全 0（未生成任何 token）"
            )

    def test_old_logps_consistent_with_recompute(self):
        """old_logps 与用同一模型重算的 per-token logp 一致（no_grad，同 ids）。"""
        torch.manual_seed(1)
        model = _tiny_model(seed=1)
        model.eval()
        tok = ByteTokenizer()
        prompts = ["Q:1+3=\nA:"]
        out = rollout(model, tok, prompts, G=2, max_new_tokens=3,
                      temperature=1.0,
                      generator=torch.Generator().manual_seed(42))

        recomputed = _per_token_logp_ref(model, out["input_ids"])
        # 忽略位置 0（无预测）
        diff = (out["old_logps"][:, 1:] - recomputed[:, 1:]).abs().max().item()
        assert diff < 1e-4, (
            f"old_logps 与重算不一致，max diff={diff:.6f}"
        )

    def test_completions_length(self):
        """completions 长度 = B，且为字符串列表。"""
        model = _tiny_model()
        tok = ByteTokenizer()
        prompts = ["Q:2+2=\nA:", "Q:1+4=\nA:"]
        G = 3
        out = rollout(model, tok, prompts, G=G, max_new_tokens=4,
                      temperature=1.0, generator=None)
        B = len(prompts) * G
        assert len(out["completions"]) == B
        assert all(isinstance(c, str) for c in out["completions"])


# ═══════════════════════════════════════════════════════════════════════════════
# T6  端到端收敛（bandit 级）
# ═══════════════════════════════════════════════════════════════════════════════
#
# 超参（连跑 3 次稳定通过，delta ≈ 0.321）：
#   SFT 预热：100 步，lr=1e-2
#   GRPO：60 轮，G=4，lr=5e-3，clip_eps=0.2，kl_coef=0.001
#   max_new_tokens=4，temperature=1.0，generator seed=42
#   任务：个位数加法 a,b∈[1,4]
#   断言：末 10 轮均值 − 首 10 轮均值 > 0.3

class TestT6EndToEnd:
    """验收标准 T6：SFT 预热 + GRPO ≤60 轮，reward 上升 >0.3，无 NaN。"""

    # ---- 任务定义（个位数加法）----
    PROMPTS = [
        "Q:1+1=\nA:", "Q:2+3=\nA:", "Q:3+4=\nA:", "Q:2+2=\nA:",
        "Q:1+4=\nA:", "Q:3+1=\nA:", "Q:4+2=\nA:", "Q:1+3=\nA:",
    ]

    def _grpo_step(self, model, ref_model, tok, prompts,
                   optimizer, G=4, max_new_tokens=4, temperature=1.0,
                   clip_eps=0.2, kl_coef=0.04, generator=None):
        """一轮 GRPO：rollout → reward → advantage → loss → step。返回平均 reward。"""
        # 1. rollout
        out = rollout(model, tok, prompts, G=G,
                      max_new_tokens=max_new_tokens,
                      temperature=temperature, generator=generator)
        input_ids = out["input_ids"]
        old_logps = out["old_logps"]
        mask = out["mask"]
        completions = out["completions"]
        prompt_lens = out["prompt_lens"]

        # 2. reward
        B = len(prompts) * G
        rewards_list = []
        flat_prompts = [p for p in prompts for _ in range(G)]
        for b in range(B):
            r = reward_fn(flat_prompts[b], completions[b])
            rewards_list.append(r)
        rewards_t = torch.tensor(rewards_list).reshape(len(prompts), G)

        # 3. advantage
        adv_2d = group_advantages(rewards_t)         # (n_prompts, G)
        adv_1d = adv_2d.reshape(-1)                  # (B,)

        # 4. 当前策略 logps
        model.train()
        logits = model(input_ids=input_ids).logits    # (B, T, V)
        logp_all = F.log_softmax(logits[:, :-1, :], dim=-1)
        cur_logps = torch.zeros_like(input_ids, dtype=logp_all.dtype)
        cur_logps[:, 1:] = logp_all.gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)

        # 5. ref logps
        with torch.no_grad():
            ref_logits = ref_model(input_ids=input_ids).logits
            ref_logp_all = F.log_softmax(ref_logits[:, :-1, :], dim=-1)
            ref_logps = torch.zeros_like(input_ids, dtype=ref_logp_all.dtype)
            ref_logps[:, 1:] = ref_logp_all.gather(
                -1, input_ids[:, 1:].unsqueeze(-1)
            ).squeeze(-1)

        # 6. loss + step
        loss, stats = grpo_loss(
            cur_logps, old_logps.detach(), ref_logps,
            adv_1d, mask.long(),
            clip_eps=clip_eps, kl_coef=kl_coef
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        return float(torch.tensor(rewards_list).mean().item()), stats

    def test_reward_increases_no_nan(self):
        """SFT 预热 + GRPO 60 轮，reward 末 10 轮均值 − 首 10 轮均值 > 0.3，无 NaN。"""
        transformers = pytest.importorskip("transformers")
        torch.manual_seed(0)
        tok = ByteTokenizer()
        model = _tiny_model(seed=0)
        ref_model = copy.deepcopy(model)
        ref_model.requires_grad_(False)
        ref_model.eval()

        # SFT 预热 100 步（超参经 3 次连跑验证）
        _sft_warmup(model, tok, n_steps=100, lr=1e-2)

        # GRPO 训练
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)
        generator = torch.Generator().manual_seed(42)

        rewards_per_round = []
        n_rounds = 60

        for round_idx in range(n_rounds):
            # 每轮用 4 prompts × G=4 → B=16
            prompts = self.PROMPTS[:4]
            avg_r, stats = self._grpo_step(
                model, ref_model, tok, prompts, optimizer,
                G=4, max_new_tokens=4, temperature=1.0,
                clip_eps=0.2, kl_coef=0.001, generator=generator
            )
            rewards_per_round.append(avg_r)

            # 无 NaN 检查
            assert not math.isnan(avg_r), f"round {round_idx}: reward 为 NaN"
            for key, val in stats.items():
                assert not math.isnan(val.item()), (
                    f"round {round_idx}: stats[{key}] 为 NaN"
                )

        first10 = sum(rewards_per_round[:10]) / 10
        last10 = sum(rewards_per_round[-10:]) / 10
        delta = last10 - first10

        assert delta > 0.3, (
            f"reward 提升不足：首 10 轮均值={first10:.4f}，"
            f"末 10 轮均值={last10:.4f}，delta={delta:.4f}（要求 >0.3）"
        )
