"""Phase 3.2 frozen acceptance tests (CPU, deterministic independent oracles)."""

from __future__ import annotations

import time

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from minikda.config import KDAConfig
from minikda.delta import delta_rule_chunked, delta_rule_recurrent
from minikda.layer import DeltaAttention, state_bytes
from minikda.tasks import generate_associative_recall_batch


def _seed(value=0):
    torch.manual_seed(value)


# ---------------- T1: chunk equivalence ----------------
@pytest.mark.parametrize("with_alpha", [False, True])
@pytest.mark.parametrize("chunk_size", [1, 4, 16, 19])
def test_t1_chunked_matches_recurrent_non_divisible(with_alpha, chunk_size):
    """T1: all required chunk sizes, T=19 tail, with/without alpha."""
    _seed(101 + chunk_size + with_alpha)
    q, k, v = (torch.randn(2, 3, 19, 5) for _ in range(3))
    beta = torch.sigmoid(torch.randn(2, 3, 19))
    alpha = torch.sigmoid(torch.randn(2, 3, 19)) if with_alpha else None
    expected = delta_rule_recurrent(q, k, v, beta, alpha)
    actual = delta_rule_chunked(q, k, v, beta, alpha, chunk_size)
    # fp32 triangular solve and sequential rank-1 path differ only by rounding.
    assert torch.allclose(actual, expected, atol=1e-4, rtol=1e-5)


# ---------------- T2: exact overwrite ----------------
def test_t2_second_value_exactly_overwrites_same_key():
    """T2: beta=alpha=1 must replace, rather than add to, one key slot."""
    key = F.normalize(torch.tensor([1.0, 2.0, -1.0]), dim=0)
    q = key.view(1, 1, 1, 3).expand(1, 1, 2, 3).clone()
    k = q.clone()
    first = torch.tensor([2.0, -3.0, 4.0])
    second = torch.tensor([-1.0, 5.0, 0.5])
    v = torch.stack([first, second]).view(1, 1, 2, 3)
    one = torch.ones(1, 1, 2)
    out = delta_rule_recurrent(q, k, v, one, one)
    # Exact unit-key algebra has only ordinary fp32 dot-product rounding.
    assert torch.allclose(out[0, 0, 1], second, atol=1e-5, rtol=0)


# ---------------- T3: orthogonal table and alpha=1 degeneration ----------------
def test_t3_orthogonal_slots_match_hand_computation():
    """T3: independent basis keys behave like a hand-written slot table."""
    k = torch.eye(3).view(1, 1, 3, 3)
    q = k.clone()
    v = torch.tensor([[[[2.0, 1.0, -1.0], [3.0, -2.0, 4.0], [0.5, 6.0, 1.0]]]])
    beta = torch.ones(1, 1, 3)
    out = delta_rule_recurrent(q, k, v, beta)
    # At t each basis query reads the value just written into its untouched slot.
    assert torch.allclose(out, v, atol=1e-6, rtol=0)


def test_t3_alpha_one_is_deltanet():
    """T3: explicitly gated alpha=1 degenerates to no-alpha DeltaNet."""
    _seed(303)
    q, k, v = (torch.randn(1, 2, 9, 4) for _ in range(3))
    beta = torch.rand(1, 2, 9)
    plain = delta_rule_recurrent(q, k, v, beta)
    gated = delta_rule_recurrent(q, k, v, beta, torch.ones_like(beta))
    assert torch.allclose(plain, gated, atol=1e-6, rtol=0)


# ---------------- T4: forgetting direction ----------------
def test_t4_early_association_decays_more_than_recent_one():
    """T4: a key not rewritten later still loses magnitude under alpha=0.5."""
    # e0 is written at t0, e1 at t1; two batch rows query them at the same t2.
    e0, e1 = torch.eye(2)
    q = torch.zeros(2, 1, 3, 2)
    q[0, 0, -1], q[1, 0, -1] = e0, e1
    k = torch.stack([e0, e1, e0]).view(1, 1, 3, 2).expand(2, -1, -1, -1).clone()
    values = torch.tensor([[2.0, 0.0], [0.0, 2.0], [0.0, 0.0]])
    v = values.view(1, 1, 3, 2).expand(2, -1, -1, -1).clone()
    beta = torch.tensor([1.0, 1.0, 0.0]).view(1, 1, 3).expand(2, -1, -1)
    alpha = torch.tensor([0.5, 0.5, 1.0]).view(1, 1, 3).expand(2, -1, -1)
    out = delta_rule_recurrent(q, k, v, beta, alpha)
    early_at_end = out[0, 0, -1].norm()
    recent_at_end = out[1, 0, -1].norm()
    assert early_at_end < recent_at_end


# ---------------- T5: streaming equivalence ----------------
def test_t5_forward_matches_step_and_state_is_fixed_shape():
    """T5: full forward and token streaming agree at every position."""
    _seed(505)
    cfg = KDAConfig(hidden_size=24, num_heads=3, head_dim=8, chunk_size=4)
    layer = DeltaAttention(cfg).eval()
    x = torch.randn(2, 11, cfg.hidden_size)
    full = layer(x)
    state = None
    pieces = []
    for t in range(x.shape[1]):
        y, state = layer.step(x[:, t : t + 1], state)
        pieces.append(y)
        assert state.shape == (2, cfg.num_heads, cfg.head_dim, cfg.head_dim)
    streamed = torch.cat(pieces, dim=1)
    # Both paths accumulate fp32; only block solve ordering differs.
    assert torch.allclose(full, streamed, atol=1e-4, rtol=1e-5)


# ---------------- T6: fixed-state bytes ----------------
def test_t6_state_bytes_closed_form_and_actual_tensor():
    """T6: closed form equals a real one-request fp32 decode state."""
    cfg = KDAConfig(hidden_size=32, num_heads=4, head_dim=8, dtype=torch.float16)
    expected = cfg.num_heads * cfg.head_dim * cfg.head_dim * 4  # fp32 accumulation state
    actual_state = torch.empty(1, cfg.num_heads, cfg.head_dim, cfg.head_dim, dtype=torch.float32)
    assert state_bytes(cfg) == expected
    assert state_bytes(cfg) == actual_state.element_size() * actual_state.nelement()


# ---------------- T7: causality ----------------
def test_t7_future_perturbation_cannot_change_prefix():
    """T7: independently perturbing future x leaves the computed prefix bitwise fixed."""
    _seed(707)
    cfg = KDAConfig(hidden_size=16, num_heads=2, head_dim=8, chunk_size=5)
    layer = DeltaAttention(cfg).eval()
    x = torch.randn(2, 13, 16)
    altered = x.clone()
    altered[:, 7:] = torch.randn_like(altered[:, 7:]) * 20
    prefix_a, prefix_b = layer(x)[:, :7], layer(altered)[:, :7]
    assert torch.equal(prefix_a, prefix_b)


class _RecallNet(nn.Module):
    """Small two-layer model used only by the T8 smoke-training acceptance."""

    def __init__(self, vocab_size=8, hidden=24):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(2 * vocab_size, hidden)
        cfg = KDAConfig(hidden_size=hidden, num_heads=3, head_dim=8, chunk_size=8)
        self.layers = nn.ModuleList([DeltaAttention(cfg), DeltaAttention(cfg)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden), nn.LayerNorm(hidden)])
        self.head = nn.Linear(hidden, vocab_size)

    def forward(self, tokens):
        x = self.embed(tokens)
        for layer, norm in zip(self.layers, self.norms):
            x = norm(x + layer(x))
        return self.head(x[:, -1])


@pytest.mark.slow
def test_t8_toy_associative_recall_trains_above_90_percent():
    """T8: fixed-seed, two-layer, exactly 200-step CPU recall smoke test."""
    _seed(808)
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(809)
    model = _RecallNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-3)
    started = time.monotonic()
    model.train()
    for _ in range(200):
        tokens, target = generate_associative_recall_batch(
            32, 1, 8, generator=generator
        )
        loss = F.cross_entropy(model(tokens), target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for _ in range(10):
            tokens, target = generate_associative_recall_batch(
                64, 1, 8, generator=generator
            )
            correct += (model(tokens).argmax(-1) == target).sum().item()
            total += target.numel()
    assert correct / total > 0.90
    assert time.monotonic() - started < 60.0
