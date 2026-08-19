"""
bench_kernels.py — Phase 2.0 Triton kernel 性能 benchmark

测量内容：
  - kernel vs torch eager 的 fwd+bwd 耗时（ms）
  - 实测带宽（GB/s）
  - 与理论带宽的利用率（%）

运行方式（需要 CUDA GPU）：
    cd m2-kernels && python3 benchmarks/bench_kernels.py

输出示例（T4 A10G，理论带宽 600 GB/s）：
    === RMSNorm H=4096 B=8 T=512 ===
    torch   : 1.23 ms  |  bandwidth  84.3 GB/s  |  utilization 14.1%
    triton  : 0.87 ms  |  bandwidth 119.2 GB/s  |  utilization 19.9%
    ...

注意：本文件不进 pytest，单独运行。
"""

import sys
import os
import torch

# ─── 无 CUDA / 无 triton 时友好退出 ──────────────────────────────────────────
if not torch.cuda.is_available():
    print("CUDA GPU 不可用，跳过 benchmark。")
    sys.exit(0)

try:
    import triton  # noqa: F401
except ImportError:
    print("triton 未安装，跳过 benchmark。")
    sys.exit(0)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from kernels.rmsnorm import rmsnorm
from kernels.swiglu import swiglu_mul
from kernels.cross_entropy import fused_cross_entropy

import torch.nn.functional as F

# ─── 工具 ─────────────────────────────────────────────────────────────────────

def _ref_rmsnorm(x, weight, eps=1e-6):
    x_f32 = x.float()
    rms = torch.sqrt(x_f32.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x_f32 / rms * weight.float()).to(x.dtype)


def benchmark(fn, warmup=20, rep=100):
    """测量 fn() 的中位耗时（ms）。"""
    # warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    import time
    times = []
    for _ in range(rep):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)  # ms

    times.sort()
    return times[len(times) // 2]   # 中位数


def get_device_bandwidth_gbps():
    """尝试从 torch 获取理论带宽（GB/s）。不同卡结果不同，此处给常见值供参考。"""
    props = torch.cuda.get_device_properties(0)
    # 近似：mem_clock(Hz) * bus_width(bits) / 8 * 2（DDR）/ 1e9
    # torch 没有直接接口，给个占位值；实际跑时替换为你的卡的规格
    # 常见卡理论带宽：A100-80G 2TB/s, A10G 600GB/s, T4 300GB/s, RTX3090 936GB/s
    return None   # 用 None 表示未知，输出时不算利用率


def fmt(label, ms, bw_gbps, theory_bw):
    if theory_bw and bw_gbps:
        util = bw_gbps / theory_bw * 100
        return f"  {label:10s}: {ms:7.3f} ms  |  bandwidth {bw_gbps:6.1f} GB/s  |  utilization {util:.1f}%"
    elif bw_gbps:
        return f"  {label:10s}: {ms:7.3f} ms  |  bandwidth {bw_gbps:6.1f} GB/s"
    else:
        return f"  {label:10s}: {ms:7.3f} ms"


theory_bw = get_device_bandwidth_gbps()
device_name = torch.cuda.get_device_name(0)
print(f"\n设备：{device_name}")
if theory_bw:
    print(f"理论带宽：{theory_bw:.0f} GB/s")
else:
    print("理论带宽：未知（请手动填入你的卡的规格以计算利用率）")
print()

# ─── RMSNorm ──────────────────────────────────────────────────────────────────

for B, T, H in [(8, 512, 4096), (8, 512, 768)]:
    print(f"=== RMSNorm  B={B} T={T} H={H} ===")
    x = torch.randn(B, T, H, device="cuda", dtype=torch.float32)
    w = torch.randn(H,      device="cuda", dtype=torch.float32)

    # torch eager（全程 f32）
    def torch_fn():
        xr = x.requires_grad_(True)
        y = _ref_rmsnorm(xr, w)
        y.sum().backward()
    ms_torch = benchmark(torch_fn)

    # triton fused
    def triton_fn():
        xr = x.requires_grad_(True)
        wr = w.requires_grad_(True)
        y = rmsnorm(xr, wr)
        y.sum().backward()
    ms_tri = benchmark(triton_fn)

    # 带宽估算：fwd 读 x+w 写 y；bwd 读 dy x w 写 dx；rstd 很小忽略
    elem_bytes = x.element_size()
    nbytes = (3 * B * T * H + 2 * H) * elem_bytes * 2   # fwd+bwd 粗估
    bw_torch = nbytes / (ms_torch * 1e-3) / 1e9
    bw_tri   = nbytes / (ms_tri   * 1e-3) / 1e9

    print(fmt("torch",  ms_torch, bw_torch, theory_bw))
    print(fmt("triton", ms_tri,   bw_tri,   theory_bw))
    print(f"  加速比：{ms_torch/ms_tri:.2f}x")
    print()

# ─── SwiGLU ───────────────────────────────────────────────────────────────────

for B, T, D in [(8, 512, 14336), (8, 512, 2048)]:
    print(f"=== SwiGLU   B={B} T={T} D={D} ===")
    gate = torch.randn(B, T, D, device="cuda", dtype=torch.float32)
    up   = torch.randn(B, T, D, device="cuda", dtype=torch.float32)

    def torch_fn():
        gr = gate.requires_grad_(True)
        ur = up.requires_grad_(True)
        y = F.silu(gr) * ur
        y.sum().backward()
    ms_torch = benchmark(torch_fn)

    def triton_fn():
        gr = gate.requires_grad_(True)
        ur = up.requires_grad_(True)
        y = swiglu_mul(gr, ur)
        y.sum().backward()
    ms_tri = benchmark(triton_fn)

    # 带宽估算：fwd 读 gate+up 写 out；bwd 读 dy gate up 写 dgate dup
    elem_bytes = gate.element_size()
    N_total = B * T * D
    nbytes_torch = 5 * N_total * elem_bytes   # naive: silu_gate 中间有额外读写
    nbytes_tri   = 5 * N_total * elem_bytes   # fused: 无中间 buffer
    bw_torch = nbytes_torch / (ms_torch * 1e-3) / 1e9
    bw_tri   = nbytes_tri   / (ms_tri   * 1e-3) / 1e9

    print(fmt("torch",  ms_torch, bw_torch, theory_bw))
    print(fmt("triton", ms_tri,   bw_tri,   theory_bw))
    print(f"  加速比：{ms_torch/ms_tri:.2f}x")
    print()

# ─── Cross-Entropy ────────────────────────────────────────────────────────────

for N, V in [(4096, 32768), (512, 128256)]:
    print(f"=== CrossEntropy  N={N} V={V} ===")
    logits  = torch.randn(N, V, device="cuda", dtype=torch.float32)
    targets = torch.randint(0, V, (N,), device="cuda")

    def torch_fn():
        lr = logits.requires_grad_(True)
        loss = F.cross_entropy(lr, targets)
        loss.backward()
    ms_torch = benchmark(torch_fn)

    def triton_fn():
        lr = logits.requires_grad_(True)
        loss = fused_cross_entropy(lr, targets)
        loss.backward()
    ms_tri = benchmark(triton_fn)

    # 带宽估算：
    #   torch 额外物化 (N,V) probs → 多读写 N*V*4 bytes
    #   fused 只读写 logits + logsumexp(N) + targets(N)
    elem_bytes = logits.element_size()
    nbytes_torch = (2 * N * V + N) * elem_bytes   # 读 logits，写 probs，读 probs+targets
    nbytes_tri   = (    N * V + N) * elem_bytes   # 读 logits（单 pass），写 logsumexp
    bw_torch = nbytes_torch / (ms_torch * 1e-3) / 1e9
    bw_tri   = nbytes_tri   / (ms_tri   * 1e-3) / 1e9

    print(fmt("torch",  ms_torch, bw_torch, theory_bw))
    print(fmt("triton", ms_tri,   bw_tri,   theory_bw))
    print(f"  加速比：{ms_torch/ms_tri:.2f}x")
    print()

print("benchmark 完成。将以上数字填入 docs/2.0-kernels/POSTMORTEM.md 第一节。")
