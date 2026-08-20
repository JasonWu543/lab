"""Phase 2.1 真卡 benchmark：warmup 后输出 eager/SDPA 与 Triton 中位耗时。"""
from __future__ import annotations

import statistics
import time

import torch
import torch.nn.functional as F


def _median_ms(fn, warmup=5, repeat=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    values = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        values.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(values)


def _eager_rope(q, k, cos, sin):
    """独立 eager rotate-half 基线，避免 benchmark 调回被测实现。"""
    half = q.shape[-1] // 2
    cos = cos[None, None]
    sin = sin[None, None]

    def rotate(x):
        lo, hi = x[..., :half], x[..., half:]
        return torch.cat((lo * cos - hi * sin, hi * cos + lo * sin), dim=-1)

    return rotate(q), rotate(k)


def run_smoke():
    from kernels.rope import apply_rope
    from kernels.block_attn import block_attention
    B, H, T, D = 1, 2, 128, 64
    q, k, v = [torch.randn(B, H, T, D, device="cuda") for _ in range(3)]
    theta = torch.randn(T, D // 2, device="cuda")
    cos, sin = theta.cos(), theta.sin()
    eager_rope_ms = _median_ms(lambda: _eager_rope(q, k, cos, sin), repeat=5)
    rope_ms = _median_ms(lambda: apply_rope(q, k, cos, sin), repeat=5)
    tri_ms = _median_ms(lambda: block_attention(q, k, v), repeat=5)
    sdpa_ms = _median_ms(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True), repeat=5)
    print(f"RoPE triton: {rope_ms:.3f} ms | eager RoPE: {eager_rope_ms:.3f} ms")
    print(f"BlockAttention triton: {tri_ms:.3f} ms | SDPA: {sdpa_ms:.3f} ms")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA GPU 不可用，跳过 Phase 2.1 benchmark。")
    else:
        run_smoke()
