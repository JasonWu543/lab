"""Phase 2.2 Grouped GEMM 验收；无 CUDA/Triton 时模块级 skip。"""
import importlib
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

if not torch.cuda.is_available():
    pytest.skip("CUDA GPU 不可用，跳过 Phase 2.2", allow_module_level=True)
pytest.importorskip("triton")

from kernels.grouped_gemm import grouped_gemm, moe_ffn_grouped


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(220)
    torch.cuda.manual_seed_all(220)


def _oracle(x, weights, offsets):
    out = torch.empty(x.shape[0], weights.shape[-1], device=x.device, dtype=torch.float32)
    for e in range(weights.shape[0]):
        s, t = int(offsets[e]), int(offsets[e + 1])
        if s < t:
            out[s:t] = x[s:t].float() @ weights[e].float()
    return out.to(x.dtype)


@pytest.mark.parametrize("E", [1, 8, 64])
@pytest.mark.parametrize("profile", ["empty", "one", "tail"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_t1_random_groups_match_loop(E, profile, dtype):
    # 每个 expert 数都显式覆盖全空、单行和非 tile 整除尾块，避免依赖随机 seed
    # 碰巧制造边界；E>1 时其余随机组继续覆盖混合分布。
    sizes = torch.randint(2, 24, (E,), dtype=torch.int64)
    sizes[0] = {"empty": 0, "one": 1, "tail": 19}[profile]
    if profile == "empty":
        sizes.zero_()
    offsets = torch.cat((torch.zeros(1, dtype=torch.int64), sizes.cumsum(0))).cuda()
    N, K, M = int(sizes.sum()), 33, 47
    x = torch.randn(N, K, device="cuda", dtype=dtype)
    w = torch.randn(E, K, M, device="cuda", dtype=dtype)
    got, ref = grouped_gemm(x, w, offsets), _oracle(x, w, offsets)
    # K=33 的 fp32 累加只在输出处量化；bf16 ULP 约 2^-7，给 2% 相对+0.08 绝对余量。
    torch.testing.assert_close(got.float(), ref.float(), rtol=0 if dtype == torch.float32 else 2e-2,
                               atol=2e-5 if dtype == torch.float32 else 8e-2)


@pytest.mark.parametrize("sizes", [[97, 0, 0, 0], [16, 16, 16, 16]])
def test_t2_extreme_distributions(sizes):
    offsets = torch.tensor([0, *torch.tensor(sizes).cumsum(0).tolist()], device="cuda")
    x = torch.randn(sum(sizes), 32, device="cuda")
    w = torch.randn(4, 32, 24, device="cuda")
    torch.testing.assert_close(grouped_gemm(x, w, offsets), _oracle(x, w, offsets),
                               rtol=1e-5, atol=1e-5)


def test_t3_offsets_contract():
    x = torch.randn(5, 8, device="cuda")
    w = torch.randn(2, 8, 4, device="cuda")
    with pytest.raises(ValueError, match="非递减"):
        grouped_gemm(x, w, torch.tensor([0, 4, 3], device="cuda"))
    with pytest.raises(ValueError, match="N_total"):
        grouped_gemm(x, w, torch.tensor([0, 2, 4], device="cuda"))


def test_t4_moe_ffn_matches_m3_reference():
    ref_dir = Path(__file__).resolve().parents[2] / "m3-arch-study" / "reference" / "3.0-moe"
    sys.path.insert(0, str(ref_dir))
    try:
        SwiGLU = importlib.import_module("moe_solution").SwiGLU
    finally:
        sys.path.pop(0)
    sizes, K, M = [3, 0, 5, 2], 16, 12
    offsets = torch.tensor([0, 3, 3, 8, 10], device="cuda")
    x = torch.randn(sum(sizes), K, device="cuda")
    wg = torch.randn(4, K, M, device="cuda")
    wu = torch.randn_like(wg)
    wd = torch.randn(4, M, K, device="cuda")
    got = moe_ffn_grouped(x, wg, wu, wd, offsets)
    ref = torch.empty_like(got)
    for e, (s, t) in enumerate(zip(offsets[:-1].tolist(), offsets[1:].tolist())):
        if s == t:
            continue
        expert = SwiGLU(K, M).cuda()
        with torch.no_grad():
            expert.gate.weight.copy_(wg[e].T)
            expert.up.weight.copy_(wu[e].T)
            expert.down.weight.copy_(wd[e].T)
        ref[s:t] = expert(x[s:t])
    torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-4)


def test_t5_benchmark_scaffold(capsys):
    from benchmarks.bench_grouped_gemm import run_smoke
    run_smoke()
    output = capsys.readouterr().out
    assert "grouped" in output and "loop" in output and "padding-bmm" in output
