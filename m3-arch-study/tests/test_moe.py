"""Phase 3.0 correctness 测试：全部 CPU / fp32 / 固定 seed。

运行：python3 -m pytest tests/test_moe.py -x -q
"""
import math

import pytest
import torch

from minimoe.config import MoEConfig
from minimoe.moe import Router, MoELayer, SwiGLU
from minimoe.mla import MLA, MLACache, RMSNorm, precompute_rope, mha_cache_bytes
from minimoe.mtp import MTPHead
from minimoe.model import MiniMoELM
from minimoe.parity import activated_ffn_params, dense_config_matching_flops


def _seed(s=0):
    torch.manual_seed(s)


def small_cfg(**kw):
    base = dict(
        hidden_size=32,
        intermediate_size=64,
        n_routed_experts=8,
        n_shared_experts=1,
        top_k=2,
        moe_intermediate_size=16,
        q_lora_rank=24,
        kv_lora_rank=16,
        qk_nope_head_dim=8,
        qk_rope_head_dim=4,
        v_head_dim=8,
        num_heads=4,
        vocab_size=64,
        num_layers=2,
        max_seq_len=32,
    )
    base.update(kw)
    return MoEConfig(**base)


# ---------------- T1 Router ----------------
def test_t1_router_topk_matches_naive():
    _seed(0)
    cfg = small_cfg()
    router = Router(cfg)
    x = torch.randn(20, cfg.hidden_size)
    topk_idx, topk_weight = router(x)

    logits = x @ router.weight.T
    score = torch.sigmoid(logits)
    biased = score + router.bias
    # naive for-loop top-k
    for n in range(x.shape[0]):
        row = biased[n]
        expected = torch.topk(row, cfg.top_k).indices
        assert set(topk_idx[n].tolist()) == set(expected.tolist())


def test_t1_gating_sums_to_one():
    _seed(1)
    cfg = small_cfg()
    router = Router(cfg)
    x = torch.randn(20, cfg.hidden_size)
    _, topk_weight = router(x)
    assert torch.allclose(topk_weight.sum(-1), torch.ones(20), atol=1e-5)


def test_t1_bias_does_not_affect_gating():
    _seed(2)
    cfg = small_cfg()
    router = Router(cfg)
    x = torch.randn(20, cfg.hidden_size)
    idx0, w0 = router(x)
    # 改 bias
    router.bias.add_(torch.randn(cfg.n_routed_experts) * 0.01)
    idx1, w1 = router(x)
    # 只要选择不变的 token，其 gating 权重必须逐位数值相同
    same = (idx0 == idx1).all(dim=-1)
    assert same.any(), "需要至少一些 token 选择不变以验证"
    assert torch.allclose(w0[same], w1[same], atol=1e-7)


# ---------------- T2 MoELayer ----------------
def test_t2_dropless_equivalence():
    _seed(3)
    cfg = small_cfg()
    layer = MoELayer(cfg).eval()  # eval 避免 update_bias 改状态
    x = torch.randn(2, 5, cfg.hidden_size)
    out = layer(x)

    x_flat = x.reshape(-1, cfg.hidden_size)
    idx, w = layer.router(x_flat)
    naive = torch.zeros_like(x_flat)
    for n in range(x_flat.shape[0]):
        for k in range(cfg.top_k):
            e = idx[n, k].item()
            naive[n] += w[n, k] * layer.experts[e](x_flat[n : n + 1]).squeeze(0)
        naive[n] += layer.shared(x_flat[n : n + 1]).squeeze(0)
    assert torch.allclose(out.reshape(-1, cfg.hidden_size), naive, atol=1e-5)


def test_t2_empty_expert_no_nan():
    _seed(4)
    cfg = small_cfg(n_routed_experts=8, top_k=1)
    layer = MoELayer(cfg).eval()
    x = torch.randn(1, 2, cfg.hidden_size)
    out = layer(x)
    assert not torch.isnan(out).any()


# ---------------- T3 load balancing ----------------
def _skewed_batch(cfg, n=256):
    # 构造偏斜输入：少数 cluster 占绝大多数 token，导致 expert 负载严重不均
    torch.manual_seed(123)
    n_clusters = 3
    centers = torch.randn(n_clusters, cfg.hidden_size) * 3.0
    # 高度不均的簇大小：一个簇几乎吃掉全部 token
    sizes = [int(n * 0.8), int(n * 0.15)]
    sizes.append(n - sum(sizes))
    parts = []
    for c, s in zip(centers, sizes):
        parts.append(c.unsqueeze(0).repeat(s, 1) + 0.1 * torch.randn(s, cfg.hidden_size))
    return torch.cat(parts, dim=0)


def test_t3_no_bias_update_is_skewed():
    _seed(5)
    cfg = small_cfg()
    router = Router(cfg)
    x = _skewed_batch(cfg)
    idx, _ = router(x)
    load = torch.bincount(idx.flatten(), minlength=cfg.n_routed_experts).float()
    load = load[load > 0]
    assert (load.max() / load.min()) > 4


def test_t3_bias_update_converges():
    _seed(5)
    cfg = small_cfg()
    router = Router(cfg)
    x = _skewed_batch(cfg)
    for _ in range(500):
        idx, _ = router(x)
        router.update_bias(idx)
    idx, _ = router(x)
    load = torch.bincount(idx.flatten(), minlength=cfg.n_routed_experts).float()
    load = load[load > 0]
    assert (load.max() / load.min()) < 2


def test_t3_bias_has_no_grad():
    cfg = small_cfg()
    router = Router(cfg)
    assert router.bias.grad is None
    assert not router.bias.requires_grad


# ---------------- T4 MLA cached vs non-cached ----------------
def test_t4_cached_matches_full():
    _seed(6)
    cfg = small_cfg()
    mla = MLA(cfg).eval()
    cos, sin = precompute_rope(cfg.qk_rope_head_dim, cfg.max_seq_len, cfg.rope_theta)
    B, T = 2, 8
    x = torch.randn(B, T, cfg.hidden_size)
    positions = torch.arange(T)
    full = mla(x, cos, sin, positions)

    # prefill 前 4，逐 token decode 后 4
    cache = MLACache(cfg, B, cfg.max_seq_len, x.device, torch.float32)
    out_chunks = []
    pref = 4
    out_chunks.append(mla(x[:, :pref], cos, sin, torch.arange(pref), cache))
    for t in range(pref, T):
        pos = torch.arange(t, t + 1)
        out_chunks.append(mla(x[:, t : t + 1], cos, sin, pos, cache))
    cached = torch.cat(out_chunks, dim=1)
    assert torch.allclose(full, cached, atol=1e-5)


def test_t4_greedy_generation_identical():
    _seed(7)
    cfg = small_cfg()
    model = MiniMoELM(cfg, use_moe=True).eval()
    B, T = 2, 4  # batch>1：cache 的 batch 维 bug 在 B=1 下不暴露
    prompt = torch.randint(0, cfg.vocab_size, (B, T))

    # 无 cache：每步全序列 forward
    seq_a = prompt.clone()
    for _ in range(5):
        logits = model(seq_a)
        nxt = logits[:, -1].argmax(-1, keepdim=True)
        seq_a = torch.cat([seq_a, nxt], dim=1)

    # 有 cache：prefill + 逐 token
    caches = model.make_caches(B, cfg.max_seq_len)
    logits = model(prompt, caches)
    nxt = logits[:, -1].argmax(-1, keepdim=True)
    seq_b = torch.cat([prompt, nxt], dim=1)
    for _ in range(4):
        logits = model(nxt, caches)
        nxt = logits[:, -1].argmax(-1, keepdim=True)
        seq_b = torch.cat([seq_b, nxt], dim=1)
    assert torch.equal(seq_a, seq_b)


# ---------------- T5 MLA memory ----------------
def test_t5_memory_closed_form():
    cfg = small_cfg()
    B, S = 2, 16
    cache = MLACache(cfg, B, cfg.max_seq_len, torch.device("cpu"), torch.float32)
    latent = torch.randn(B, S, cfg.kv_lora_rank)
    k_rope = torch.randn(B, S, cfg.qk_rope_head_dim)
    cache.update(latent, k_rope)
    expected = B * S * (cfg.kv_lora_rank + cfg.qk_rope_head_dim) * 4
    assert cache.memory_bytes() == expected


def test_t5_memory_below_40pct_mha():
    cfg = small_cfg()
    B, S = 2, 16
    cache = MLACache(cfg, B, cfg.max_seq_len, torch.device("cpu"), torch.float32)
    cache.update(torch.randn(B, S, cfg.kv_lora_rank), torch.randn(B, S, cfg.qk_rope_head_dim))
    mha = mha_cache_bytes(cfg, B, S, torch.float32)
    assert cache.memory_bytes() < 0.4 * mha


# ---------------- T6 MTP ----------------
def test_t6_mtp_output_shape():
    _seed(8)
    cfg = small_cfg()
    model = MiniMoELM(cfg, use_moe=True, use_mtp=True)
    B, T = 2, 6
    ids = torch.randint(0, cfg.vocab_size, (B, T))
    logits, mtp_logits = model(ids)
    assert mtp_logits.shape == (B, T - 1, cfg.vocab_size)


def test_t6_joint_loss_decreases():
    _seed(9)
    cfg = small_cfg(num_layers=2)
    model = MiniMoELM(cfg, use_moe=True, use_mtp=True)
    B, T = 4, 8
    ids = torch.randint(0, cfg.vocab_size, (B, T))
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)

    def loss_fn():
        logits, mtp_logits = model(ids)
        # main: predict t+1
        l_main = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, cfg.vocab_size), ids[:, 1:].reshape(-1)
        )
        # mtp: predict t+2
        l_mtp = torch.nn.functional.cross_entropy(
            mtp_logits[:, :-1].reshape(-1, cfg.vocab_size), ids[:, 2:].reshape(-1)
        )
        return l_main + l_mtp

    init = loss_fn().item()
    for _ in range(300):
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        opt.step()
    final = loss_fn().item()
    assert final < 0.5 * init


def test_t6_mtp_grad_flows_to_trunk():
    _seed(10)
    cfg = small_cfg()
    model = MiniMoELM(cfg, use_moe=True, use_mtp=True)
    B, T = 2, 6
    ids = torch.randint(0, cfg.vocab_size, (B, T))
    _, mtp_logits = model(ids)
    # 只用 MTP loss 反传
    loss = torch.nn.functional.cross_entropy(
        mtp_logits[:, :-1].reshape(-1, cfg.vocab_size), ids[:, 2:].reshape(-1)
    )
    loss.backward()
    # 主干某参数应有梯度（MLA 的 o_proj）
    g = model.blocks[0].attn.o_proj.weight.grad
    assert g is not None and g.abs().sum() > 0


# ---------------- T7 parity ----------------
def test_t7_parity_error_under_2pct():
    cfg = MoEConfig()
    dense_cfg = dense_config_matching_flops(cfg)
    moe_p = activated_ffn_params(cfg, moe=True)
    dense_p = activated_ffn_params(dense_cfg, moe=False)
    err = abs(dense_p - moe_p) / moe_p
    assert err < 0.02
