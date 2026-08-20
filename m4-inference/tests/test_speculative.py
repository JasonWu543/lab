"""
Phase 4.1: 投机解码测试套件（CPU，tiny 随机模型 + mock 常数分布模型）

运行方式:
    python3 -m pytest tests/test_speculative.py -x -q

统计功效说明（T2/T3 为什么这样设计）：
    经验分布 vs 真分布的 TV 距离噪声 ~ sqrt(V/N)。vocab=64、N=5000 时噪声
    ≈0.11，测不出 0.05 的阈值；所以分布一致性测试用 vocab=16 的模型
    （T2）和 vocab=8 的 mock 常数分布模型（T3），并全部固定 seed。
"""
import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from minivllm.speculative import speculative_generate, SpecStats
from transformers import Qwen2Config, AutoModelForCausalLM


# ──────────────────────────── fixtures / helpers ────────────────────────────

def _tiny_model(vocab, hidden, layers, heads, kv_heads, seed):
    torch.manual_seed(seed)
    cfg = Qwen2Config(
        vocab_size=vocab, hidden_size=hidden, num_hidden_layers=layers,
        num_attention_heads=heads, num_key_value_heads=kv_heads,
        intermediate_size=hidden * 2, max_position_embeddings=256,
        attn_implementation="eager",
    )
    m = AutoModelForCausalLM.from_config(cfg)
    m.eval()
    return m


@pytest.fixture(scope="module")
def pair64():
    """vocab 64：draft 更小更差，用于 greedy/记账/边界测试。"""
    target = _tiny_model(64, 64, 2, 4, 2, seed=0)
    draft = _tiny_model(64, 32, 1, 2, 1, seed=1)
    return target, draft


@pytest.fixture(scope="module")
def pair16():
    """vocab 16：用于分布一致性统计测试。"""
    target = _tiny_model(16, 64, 2, 4, 2, seed=2)
    draft = _tiny_model(16, 32, 1, 2, 1, seed=3)
    return target, draft


class ConstLogitsLM(nn.Module):
    """mock：任何位置都输出同一组 logits。用于精确控制 p/q。"""

    def __init__(self, logits_row: torch.Tensor):
        super().__init__()
        self.row = logits_row  # (V,)

    def forward(self, input_ids):
        B, T = input_ids.shape
        logits = self.row.view(1, 1, -1).expand(B, T, -1)
        return type("Out", (), {"logits": logits})()

    __call__ = forward


def _target_greedy(target, input_ids, n):
    """oracle：target 单独 greedy。"""
    seq = input_ids.clone()
    with torch.no_grad():
        for _ in range(n):
            logits = target(seq).logits[:, -1, :]
            seq = torch.cat([seq, logits.argmax(-1, keepdim=True)], dim=1)
    return seq


def _processed_probs(logits, temperature, top_p):
    """oracle：SPEC 分布约定（temperature → softmax → top-p 截断归一化）。"""
    probs = torch.softmax(logits / temperature, dim=-1)
    if top_p < 1.0:
        sp, si = torch.sort(probs, descending=True, dim=-1)
        cum = sp.cumsum(-1)
        sp = sp.masked_fill((cum - sp) >= top_p, 0.0)
        probs = torch.zeros_like(probs).scatter_(-1, si, sp)
        probs = probs / probs.sum(-1, keepdim=True)
    return probs


def _tv(a: torch.Tensor, b: torch.Tensor) -> float:
    return 0.5 * (a - b).abs().sum().item()


# ──────────────────────────────────── T1 ───────────────────────────────────

class TestT1GreedyLossless:
    """T1: temperature=0 时输出与 target 单独 greedy 完全一致"""

    @pytest.mark.parametrize("k", [1, 4, 8])
    @pytest.mark.parametrize("prompt_seed", [10, 11])
    def test_greedy_matches_target(self, pair64, k, prompt_seed):
        target, draft = pair64
        g = torch.Generator().manual_seed(0)
        torch.manual_seed(prompt_seed)
        prompt = torch.randint(0, 64, (1, 5))
        n_new = 12

        expected = _target_greedy(target, prompt, n_new)
        out, stats = speculative_generate(
            target, draft, prompt, max_new_tokens=n_new, k=k,
            temperature=0.0, generator=g)

        assert torch.equal(out, expected), (
            f"greedy 无损性破坏（k={k}）：\nspec  ={out.tolist()}\n"
            f"target={expected.tolist()}\n"
            "提示：检查 target 验证 logits 的位置切片（off-by-one）和拒绝后的重采样"
        )


# ──────────────────────────────────── T2 ───────────────────────────────────

class TestT2SamplingLossless:
    """T2: 采样模式下第一个新 token 的分布与 target 直接采样一致"""

    N = 8000

    @pytest.mark.parametrize("temperature,top_p", [(1.0, 1.0), (0.8, 0.9)])
    def test_first_token_distribution(self, pair16, temperature, top_p):
        target, draft = pair16
        torch.manual_seed(20)
        prompt = torch.randint(0, 16, (1, 4))

        # 精确的 target 分布（oracle）
        with torch.no_grad():
            t_logits = target(prompt).logits[:, -1, :]
        p_exact = _processed_probs(t_logits, temperature, top_p)[0]

        g = torch.Generator().manual_seed(42)
        counts = torch.zeros(16)
        for _ in range(self.N):
            out, _ = speculative_generate(
                target, draft, prompt, max_new_tokens=1, k=1,
                temperature=temperature, top_p=top_p, generator=g)
            counts[out[0, -1].item()] += 1
        empirical = counts / counts.sum()

        tv = _tv(empirical, p_exact)
        assert tv < 0.05, (
            f"spec 采样分布偏离 target 分布：TV={tv:.4f} >= 0.05 "
            f"(temperature={temperature}, top_p={top_p})。"
            "提示：接受判据或拒绝后的重采样分布推错了都会走到这里"
        )


# ──────────────────────────────────── T3 ───────────────────────────────────

class TestT3CorrectionMath:
    """T3: 用 mock 常数分布模型精确验证 accept/reject 的数学"""

    N = 50000

    @pytest.fixture(scope="class")
    def pq_setup(self):
        # vocab 8 的固定 p / q（手工造出明显差异）
        p_logits = torch.tensor([2.0, 1.0, 0.5, 0.0, -0.5, -1.0, -1.5, -2.0])
        q_logits = torch.tensor([0.0, 0.5, 2.0, 1.0, -1.0, -0.5, -2.0, -1.5])
        p = torch.softmax(p_logits, -1)
        q = torch.softmax(q_logits, -1)
        target = ConstLogitsLM(p_logits)
        draft = ConstLogitsLM(q_logits)
        return target, draft, p, q

    def test_acceptance_rate_and_residual(self, pq_setup):
        target, draft, p, q = pq_setup
        prompt = torch.zeros(1, 3, dtype=torch.long)
        g = torch.Generator().manual_seed(7)

        first_counts = torch.zeros(8)
        rejected_counts = torch.zeros(8)
        n_accept = 0
        n_reject = 0
        for _ in range(self.N):
            out, stats = speculative_generate(
                target, draft, prompt, max_new_tokens=1, k=1,
                temperature=1.0, generator=g)
            tok = out[0, -1].item()
            first_counts[tok] += 1
            if stats.accepted == 1:
                n_accept += 1
            else:
                n_reject += 1
                rejected_counts[tok] += 1

        # ① 接受率 ≈ Σ min(p, q)
        alpha_expected = torch.minimum(p, q).sum().item()
        alpha_actual = n_accept / self.N
        assert abs(alpha_actual - alpha_expected) < 0.02, (
            f"接受率 {alpha_actual:.4f} ≠ 理论值 Σmin(p,q)={alpha_expected:.4f}"
        )

        # ② 拒绝后重采样的经验分布 ≈ norm(max(0, p−q))
        residual = torch.clamp(p - q, min=0.0)
        residual = residual / residual.sum()
        rej_empirical = rejected_counts / max(n_reject, 1)
        assert _tv(rej_empirical, residual) < 0.02, (
            f"拒绝后重采样分布错误：TV={_tv(rej_empirical, residual):.4f}"
        )

        # ③ 合成分布 ≈ p（无损性的直接体现）
        first_empirical = first_counts / first_counts.sum()
        assert _tv(first_empirical, p) < 0.02, (
            f"合成输出分布 ≠ target 分布 p：TV={_tv(first_empirical, p):.4f}"
        )

    def test_zero_rng_draw_does_not_accept_zero_probability(self, monkeypatch):
        """Finite-precision RNG includes 0, which must not pass alpha == 0."""
        target = ConstLogitsLM(torch.tensor([10.0, 0.0, -10.0]))
        draft = ConstLogitsLM(torch.tensor([-10.0, 0.0, 10.0]))
        prompt = torch.zeros(1, 1, dtype=torch.long)
        monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: torch.zeros(1))
        out, stats = speculative_generate(
            target, draft, prompt, max_new_tokens=1, k=1, temperature=0.0,
            generator=torch.Generator().manual_seed(0),
        )
        assert out[0, -1].item() == 0
        assert stats.accepted == 0


# ──────────────────────────────────── T4 ───────────────────────────────────

class TestT4Stats:
    """T4: 统计记账"""

    def test_accepted_le_proposed(self, pair64):
        target, draft = pair64
        g = torch.Generator().manual_seed(3)
        torch.manual_seed(30)
        prompt = torch.randint(0, 64, (1, 5))
        out, stats = speculative_generate(
            target, draft, prompt, max_new_tokens=20, k=4,
            temperature=1.0, generator=g)
        assert 0 < stats.proposed
        assert stats.accepted <= stats.proposed
        assert stats.draft_forwards >= stats.proposed
        assert 0.0 <= stats.acceptance_rate <= 1.0

    def test_identical_models_accept_everything(self, pair64):
        """draft == target 时 p == q，接受率应为 1（数值容差内）"""
        target, _ = pair64
        g = torch.Generator().manual_seed(4)
        torch.manual_seed(31)
        prompt = torch.randint(0, 64, (1, 5))
        out, stats = speculative_generate(
            target, target, prompt, max_new_tokens=24, k=4,
            temperature=1.0, generator=g)
        assert stats.acceptance_rate > 0.99, (
            f"draft==target 时应全接受，实际 {stats.acceptance_rate:.3f}"
        )
        # 全接受时：生成数 = accepted + bonus（每轮 1 个）
        n_generated = out.shape[1] - prompt.shape[1]
        assert n_generated == 24
        assert stats.accepted <= n_generated  # 差值即 bonus token 数


# ──────────────────────────────────── T5 ───────────────────────────────────

class TestT5Boundaries:
    """T5: 生成长度精确截断 + k 边界 + 可复现性"""

    @pytest.mark.parametrize("max_new,k", [(1, 4), (5, 8), (7, 1), (8, 4)])
    def test_exact_length(self, pair64, max_new, k):
        target, draft = pair64
        g = torch.Generator().manual_seed(5)
        torch.manual_seed(32)
        prompt = torch.randint(0, 64, (1, 6))
        out, _ = speculative_generate(
            target, draft, prompt, max_new_tokens=max_new, k=k,
            temperature=1.0, generator=g)
        assert out.shape == (1, 6 + max_new), (
            f"要求恰好 {max_new} 个新 token，得到 {out.shape[1] - 6} 个"
            "（bonus token 越界？最后一轮的 k 没截断？）"
        )
        assert torch.equal(out[:, :6], prompt), "prompt 部分被改动"

    def test_reproducible(self, pair64):
        target, draft = pair64
        torch.manual_seed(33)
        prompt = torch.randint(0, 64, (1, 5))
        outs = []
        for _ in range(2):
            g = torch.Generator().manual_seed(99)
            out, _ = speculative_generate(
                target, draft, prompt, max_new_tokens=10, k=4,
                temperature=0.9, top_p=0.9, generator=g)
            outs.append(out)
        assert torch.equal(outs[0], outs[1]), "固定 generator 不可复现"
