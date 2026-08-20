"""Phase 3.1 验收：每个测试对应冻结 SPEC 的 T 编号。"""

import math

import pytest
import torch

from minidsv4.hyper_conn import HyperConnection
from minidsv4.muon import Muon, newton_schulz_msign
from minidsv4.sparse_attn import (
    SparseAttnConfig, build_block_mask, sparse_attention, sparse_attn_flops,
)
from minidsv4.toy_model import HCToyModel, ResidualToyModel, copy_block_weights


def _dense_causal(q, k, v):
    """独立 oracle：不复用被测 mask 或 attention。"""
    T, D = q.shape[-2:]
    scores = q @ k.transpose(-1, -2) / math.sqrt(D)
    future = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    return torch.softmax(scores.masked_fill(future, -torch.inf), -1) @ v


def test_t1_sparse_degenerates_to_dense():
    """T1：全部块覆盖时，对拍手写 dense causal attention。"""
    torch.manual_seed(101)
    q, k, v = (torch.randn(2, 3, 11, 8) for _ in range(3))
    cfg = SparseAttnConfig(block_size=4, top_k=3, local_blocks=3, sink_blocks=1)
    actual = sparse_attention(q, k, v, cfg)
    expected = _dense_causal(q, k, v)
    # fp32 两条路径只有 mask 构造不同；softmax/matmul相同，1e-5 足够覆盖舍入。
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_t2_causality_enumeration_and_perturbation():
    """T2：20 组 cfg 枚举块因果性，并验证未来 value 零敏感。"""
    generator = torch.Generator().manual_seed(202)
    for _ in range(20):
        block_size = int(torch.randint(1, 6, (), generator=generator))
        n_blocks = int(torch.randint(2, 9, (), generator=generator))
        T = n_blocks * block_size
        cfg = SparseAttnConfig(
            block_size,
            int(torch.randint(0, n_blocks + 1, (), generator=generator)),
            int(torch.randint(1, n_blocks + 1, (), generator=generator)),
            int(torch.randint(0, 3, (), generator=generator)),
        )
        scores = torch.randn(2, 2, T, n_blocks, generator=generator)
        pos = torch.arange(T)
        mask = build_block_mask(scores, cfg, pos)
        block_ids = torch.arange(n_blocks)
        future = block_ids.view(1, -1) > (pos // block_size).view(-1, 1)
        assert not mask.masked_select(future.view(1, 1, T, n_blocks)).any()

    q, k, v = (torch.randn(1, 2, 13, 6, generator=generator) for _ in range(3))
    altered = v.clone()
    altered[:, :, 7:] += 1000 * torch.randn(altered[:, :, 7:].shape, generator=generator)
    cfg = SparseAttnConfig(3, 2, 2, 1)
    before = sparse_attention(q, k, v, cfg)[:, :, :7]
    after = sparse_attention(q, k, altered, cfg)[:, :, :7]
    torch.testing.assert_close(before, after, atol=0, rtol=0)

    # 未来 K 扰动同样必须零敏感——这里藏着一个隐蔽泄漏路径：当前块的
    # pooled 分数含未来 K，若它参与 top-k 竞争，可把过去块挤出选择集。
    # cfg 取 top_k=1/local=1/sink=0 使竞争最激烈，100 组随机输入全查。
    cfg_tight = SparseAttnConfig(block_size=4, top_k=1, local_blocks=1,
                                 sink_blocks=0)
    for _ in range(100):
        q2, k2, v2 = (torch.randn(1, 1, 20, 6, generator=generator)
                      for _ in range(3))
        k_alt = k2.clone()
        k_alt[:, :, 10:] += 1000 * torch.randn(
            k_alt[:, :, 10:].shape, generator=generator)
        before_k = sparse_attention(q2, k2, v2, cfg_tight)[:, :, :10]
        after_k = sparse_attention(q2, k_alt, v2, cfg_tight)[:, :, :10]
        torch.testing.assert_close(before_k, after_k, atol=0, rtol=0)


def test_t3_exact_block_union_and_flops():
    """T3：手工分数断言并集，并用实际 token 数独立核对 FLOPs。"""
    cfg = SparseAttnConfig(block_size=4, top_k=1, local_blocks=2, sink_blocks=1)
    scores = torch.tensor([[[[-9., -8., -7., 10., -6., -5.]]]])
    mask = build_block_mask(scores, cfg, torch.tensor([23]))
    expected = torch.tensor([True, False, False, True, True, True])
    assert torch.equal(mask[0, 0, 0], expected)
    selected_tokens = int(expected.sum()) * cfg.block_size
    manual = 2 * selected_tokens * 8 * 2  # QK 与 AV，各是乘加 2 FLOPs。
    assert sparse_attn_flops(24, 8, cfg) == manual


@pytest.mark.parametrize("n", [2, 4])
def test_t4_hc_identity_matches_residual_bitwise(n):
    """T4：同权重注入后，n=2/4 的 HC 与 Pre-Norm residual 逐位相等。"""
    torch.manual_seed(404)
    residual = ResidualToyModel(dim=12, depth=5)
    hc = HCToyModel(dim=12, depth=5, expand_n=n)
    copy_block_weights(residual, hc)
    x = torch.randn(3, 7, 12)
    assert torch.equal(hc(x), residual(x))


def test_t5_hc_deep_gradient_flow_not_worse():
    """T5：固定随机初始化，比较 16 层共享 MLP 的逐层梯度范数跨度。"""
    torch.manual_seed(505)
    residual = ResidualToyModel(dim=16, depth=16)
    hc = HCToyModel(dim=16, depth=16, expand_n=4)
    copy_block_weights(residual, hc)
    x = torch.randn(4, 6, 16)
    target = torch.randn_like(x)
    ((residual(x) - target) ** 2).mean().backward()
    ((hc(x) - target) ** 2).mean().backward()
    residual_norms = torch.tensor([b.down.weight.grad.norm() for b in residual.blocks])
    hc_norms = torch.tensor([b.down.weight.grad.norm() for b in hc.blocks])
    residual_ratio = residual_norms.max() / residual_norms.min()
    hc_ratio = hc_norms.max() / hc_norms.min()
    # 恒等起点理论相同；1e-6 仅容纳不同 stream 图的 fp32 reduction 舍入。
    assert hc_ratio <= residual_ratio * (1 + 1e-6)


def test_t6_msign_near_orthogonality_and_svd_oracle():
    """T6（SPEC 2026-08 修订口径）：五步固定系数 NS 不收敛到精确极分解，
    验收改为三条可判别性质。门槛依据：随机 64×32 高斯输入实测
    奇异值 ∈ [0.682, 1.134]、UVᵀ 相对误差 ≤ 0.232、正交输入偏差 0.0085。"""
    for seed in (606, 607, 608):
        torch.manual_seed(seed)
        G = torch.randn(64, 32)
        O = newton_schulz_msign(G)
        # (a) 奇异值全部落在 [0.6, 1.2]：排除清零与单纯 G/||G|| 归一化
        sv = torch.linalg.svdvals(O.float())
        assert sv.min() > 0.6 and sv.max() < 1.2
        # (b) 与精确极分解 UVᵀ 的相对 Frobenius 误差 < 0.3（独立 SVD oracle）
        U, _, Vh = torch.linalg.svd(G, full_matrices=False)
        exact = U @ Vh
        assert (O - exact).norm() / exact.norm() < 0.3
    # (c) 正交输入是近不动点：f(1)=a+b+c≈1.001，五步后奇异值 ≈1.019
    torch.manual_seed(609)
    Q, _ = torch.linalg.qr(torch.randn(64, 32))
    assert (newton_schulz_msign(Q) - Q).abs().max() < 0.05


def test_t7_muon_momentum_shape_validation_and_scaling():
    """T7：两步 momentum 手算、非 2D 拒绝、rows/cols 缩放对拍。"""
    with pytest.raises(ValueError, match="2D"):
        Muon([torch.nn.Parameter(torch.zeros(3))])

    p = torch.nn.Parameter(torch.zeros(4, 2))
    opt = Muon([p], lr=0.1, momentum=0.5, nesterov=True)
    g1 = torch.tensor([[1., 2.], [3., 4.], [5., 6.], [7., 8.]])
    p.grad = g1.clone()
    expected1 = -0.1 * math.sqrt(2) * newton_schulz_msign(g1 + 0.5 * g1)
    opt.step()
    torch.testing.assert_close(p, expected1, atol=2e-7, rtol=2e-6)
    assert torch.equal(opt.state[p]["momentum_buffer"], g1)

    g2 = torch.flip(g1, dims=(0,))
    p.grad = g2.clone()
    buffer2 = 0.5 * g1 + g2
    update2 = g2 + 0.5 * buffer2
    expected2 = expected1 - 0.1 * math.sqrt(2) * newton_schulz_msign(update2)
    opt.step()
    torch.testing.assert_close(opt.state[p]["momentum_buffer"], buffer2)
    torch.testing.assert_close(p, expected2, atol=3e-7, rtol=2e-6)


def test_t8_muon_beats_best_same_grid_sgd_momentum():
    """T8：固定 seed 小回归，Muon 优于同 lr 网格的最佳 SGD-momentum。"""
    torch.manual_seed(808)
    x = torch.randn(96, 12)
    truth = torch.randn(12, 7)
    y = x @ truth
    initial = torch.zeros(12, 7)

    def train(kind, lr):
        w = torch.nn.Parameter(initial.clone())
        optimizer = (Muon([w], lr=lr, momentum=0.9)
                     if kind == "muon" else torch.optim.SGD([w], lr=lr, momentum=0.9))
        for _ in range(35):
            optimizer.zero_grad()
            loss = ((x @ w - y) ** 2).mean()
            loss.backward()
            optimizer.step()
        return float(((x @ w - y) ** 2).mean())

    # 对两种优化器使用完全相同、覆盖 Muon 默认量级附近的学习率网格。
    grid = (0.1, 0.2, 0.4)
    muon_loss = min(train("muon", lr) for lr in grid)
    sgd_loss = min(train("sgd", lr) for lr in grid)
    assert muon_loss < sgd_loss
