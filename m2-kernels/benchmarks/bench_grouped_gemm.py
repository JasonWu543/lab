"""Phase 2.2 benchmark：grouped vs 逐组 mm vs padding-bmm。"""
from __future__ import annotations

import statistics
import time

import torch


def _median(fn, warmup=5, repeat=15):
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


def run_smoke():
    from kernels.grouped_gemm import grouped_gemm
    sizes, K, M = [1, 7, 0, 19, 3, 12, 2, 8], 64, 96
    offsets = torch.tensor([0, *torch.tensor(sizes).cumsum(0).tolist()], device="cuda")
    x = torch.randn(sum(sizes), K, device="cuda")
    w = torch.randn(len(sizes), K, M, device="cuda")

    def loop():
        return [x[int(offsets[e]):int(offsets[e + 1])] @ w[e] for e in range(len(sizes))]

    max_n = max(sizes)
    padded = torch.zeros(len(sizes), max_n, K, device="cuda")
    for e, (s, t) in enumerate(zip(offsets[:-1].tolist(), offsets[1:].tolist())):
        padded[e, :t - s] = x[s:t]

    grouped_ms = _median(lambda: grouped_gemm(x, w, offsets), repeat=5)
    loop_ms = _median(loop, repeat=5)
    bmm_ms = _median(lambda: torch.bmm(padded, w), repeat=5)
    print(f"grouped: {grouped_ms:.3f} ms | loop: {loop_ms:.3f} ms | padding-bmm: {bmm_ms:.3f} ms")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA GPU 不可用，跳过 Phase 2.2 benchmark。")
    else:
        run_smoke()
