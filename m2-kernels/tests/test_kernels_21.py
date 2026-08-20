"""Phase 2.1 验收。无 CUDA/Triton 时模块级 skip。"""
import gc
import math

import pytest
import torch
import torch.nn.functional as F

if not torch.cuda.is_available():
    pytest.skip("CUDA GPU 不可用，跳过 Phase 2.1", allow_module_level=True)
pytest.importorskip("triton")

from kernels.rope import apply_rope, rope_fwd
from kernels.block_attn import block_attention


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(210)
    torch.cuda.manual_seed_all(210)


def _eager_rope(x, cos, sin):
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half].float(), x[..., half:].float()
    c = cos.float()[None, None]
    s = sin.float()[None, None]
    return torch.cat((x1 * c - x2 * s, x2 * c + x1 * s), -1).to(x.dtype)


@pytest.mark.parametrize("shape", [(2, 3, 17, 32), (1, 4, 65, 64)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_t1_rope_forward(shape, dtype):
    q = torch.randn(shape, device="cuda", dtype=dtype)
    k = torch.randn_like(q)
    cos = torch.randn(shape[-2], shape[-1] // 2, device="cuda")
    sin = torch.randn_like(cos)
    oq, ok = rope_fwd(q, k, cos, sin)
    # bf16 单次量化的单位舍入约 2^-7；旋转含两个乘积和，2e-2 覆盖最坏累积。
    atol = 1e-5 if dtype == torch.float32 else 2e-2
    torch.testing.assert_close(oq.float(), _eager_rope(q, cos, sin).float(), rtol=0, atol=atol)
    torch.testing.assert_close(ok.float(), _eager_rope(k, cos, sin).float(), rtol=0, atol=atol)


def test_t2_rope_backward_and_norm():
    shape = (2, 2, 19, 32)
    q = torch.randn(shape, device="cuda", requires_grad=True)
    k = torch.randn(shape, device="cuda", requires_grad=True)
    theta = torch.randn(shape[-2], shape[-1] // 2, device="cuda")
    cos, sin = theta.cos(), theta.sin()
    qr, kr = q.detach().clone().requires_grad_(True), k.detach().clone().requires_grad_(True)
    goq, gok = torch.randn_like(q), torch.randn_like(k)
    oq, ok = apply_rope(q, k, cos, sin)
    (oq * goq).sum().add((ok * gok).sum()).backward()
    ((_eager_rope(qr, cos, sin) * goq).sum() +
     (_eager_rope(kr, cos, sin) * gok).sum()).backward()
    torch.testing.assert_close(q.grad, qr.grad, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(k.grad, kr.grad, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(q.grad.norm(dim=-1), goq.norm(dim=-1), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("T,D,blocks", [(73, 32, (32, 32)), (129, 64, (64, 32)), (67, 128, (32, 64)), (33, 64, (64, 64))])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_t3_block_attention_matches_sdpa(causal, T, D, blocks, dtype):
    q, k, v = [torch.randn(1, 2, T, D, device="cuda", dtype=dtype) for _ in range(3)]
    got = block_attention(q, k, v, causal=causal, BLOCK_M=blocks[0], BLOCK_N=blocks[1])
    ref = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), is_causal=causal).to(dtype)
    atol = 2e-5 if dtype == torch.float32 else 3e-2
    torch.testing.assert_close(got.float(), ref.float(), rtol=0, atol=atol)


def test_t4_online_softmax_large_common_offset():
    q, k, v = [torch.randn(1, 1, 96, 32, device="cuda") * 0.1 for _ in range(3)]
    q[..., 0] = math.sqrt(80.0)
    k[..., 0] = math.sqrt(80.0) * math.sqrt(32.0)  # scale 后给所有 score 加约 80
    got = block_attention(q, k, v)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert torch.isfinite(got).all()
    torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-5)


def test_t5_no_score_materialization():
    B, H, T, D = 1, 1, 4096, 64
    q, k, v = [torch.randn(B, H, T, D, device="cuda") for _ in range(3)]
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    out = block_attention(q, k, v)
    torch.cuda.synchronize()
    delta = torch.cuda.max_memory_allocated() - baseline
    score_bytes = B * H * T * T * 4
    assert delta < score_bytes / 2, (delta, score_bytes)
    del out, q, k, v
    gc.collect()
    torch.cuda.empty_cache()


def test_t6_benchmark_scaffold(capsys):
    from benchmarks.bench_advanced import run_smoke
    run_smoke()
    output = capsys.readouterr().out
    assert "RoPE triton" in output and "eager RoPE" in output
    assert "BlockAttention" in output and "SDPA" in output
