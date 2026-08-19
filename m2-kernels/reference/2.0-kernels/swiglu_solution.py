"""参考答案 — SwiGLU fused Triton kernel

⚠️  先自己实现 kernels/swiglu.py，卡 30 分钟以上再看。
看完必须能回答：
  1. d_gate 的完整表达式（乘积法则展开后整理）？
  2. bwd 里为什么不需要 save silu_gate？重算代价是多少？
  3. BLOCK_SIZE 选 1024 vs 4096 的取舍？

梯度推导（完整版）：
  silu(x) = x * σ(x)，σ = sigmoid
  d/dx silu(x) = σ(x) + x * σ(x)*(1-σ(x))
               = σ(x) * (1 + x*(1-σ(x)))
               = σ(x) * (1 + x - x*σ(x))

  out = silu(gate) * up
  d_up   = dy * silu(gate)
  d_gate = dy * up * d/dx silu(x)|_{x=gate}
         = dy * up * σ(gate) * (1 + gate - gate*σ(gate))
"""
from __future__ import annotations

import math
import torch
from torch import Tensor

import triton
import triton.language as tl


BLOCK_SIZE = 1024


@triton.jit
def _swiglu_fwd_kernel(
    GATE_ptr, UP_ptr, OUT_ptr,
    N_total,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_total

    gate_raw = tl.load(GATE_ptr + offsets, mask=mask, other=0.0)
    up_raw   = tl.load(UP_ptr   + offsets, mask=mask, other=0.0)

    gate = gate_raw.to(tl.float32)
    up   = up_raw.to(tl.float32)

    sigma = 1.0 / (1.0 + tl.exp(-gate))
    silu_gate = gate * sigma

    out = silu_gate * up
    tl.store(OUT_ptr + offsets, out.to(gate_raw.dtype), mask=mask)


@triton.jit
def _swiglu_bwd_kernel(
    DY_ptr, GATE_ptr, UP_ptr,
    DGATE_ptr, DUP_ptr,
    N_total,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_total

    dy_raw   = tl.load(DY_ptr   + offsets, mask=mask, other=0.0)
    gate_raw = tl.load(GATE_ptr + offsets, mask=mask, other=0.0)
    up_raw   = tl.load(UP_ptr   + offsets, mask=mask, other=0.0)

    dy   = dy_raw.to(tl.float32)
    gate = gate_raw.to(tl.float32)
    up   = up_raw.to(tl.float32)

    sigma = 1.0 / (1.0 + tl.exp(-gate))
    silu_gate = gate * sigma

    d_up   = dy * silu_gate
    # d/dx silu = σ(x)*(1 + x - x*σ(x))
    d_gate = dy * up * sigma * (1.0 + gate - gate * sigma)

    tl.store(DGATE_ptr + offsets, d_gate.to(gate_raw.dtype), mask=mask)
    tl.store(DUP_ptr   + offsets, d_up.to(up_raw.dtype),     mask=mask)


class SwiGLUFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, gate: Tensor, up: Tensor) -> Tensor:
        gate_c = gate.contiguous()
        up_c   = up.contiguous()
        out = torch.empty_like(gate_c)
        N_total = gate_c.numel()

        grid = (math.ceil(N_total / BLOCK_SIZE),)
        _swiglu_fwd_kernel[grid](gate_c, up_c, out, N_total, BLOCK_SIZE=BLOCK_SIZE)

        ctx.save_for_backward(gate_c, up_c)
        return out

    @staticmethod
    def backward(ctx, dy: Tensor):
        gate, up = ctx.saved_tensors
        dy_c = dy.contiguous()
        d_gate = torch.empty_like(gate)
        d_up   = torch.empty_like(up)
        N_total = gate.numel()

        grid = (math.ceil(N_total / BLOCK_SIZE),)
        _swiglu_bwd_kernel[grid](
            dy_c, gate, up, d_gate, d_up, N_total, BLOCK_SIZE=BLOCK_SIZE,
        )
        return d_gate, d_up


def swiglu_mul(gate: Tensor, up: Tensor) -> Tensor:
    """参考答案版 swiglu_mul（与骨架接口相同）。"""
    return SwiGLUFunction.apply(gate, up)
