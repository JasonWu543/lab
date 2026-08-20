"""参考答案 — RMSNorm Triton kernel

⚠️  先自己实现 kernels/rmsnorm.py，卡 30 分钟以上再看。
看完必须能回答：
  1. bwd 里 dx 的「修正项」是什么？怎么推出来的？
  2. 为什么 bwd 需要两次分块循环（或者为什么可以合成一次）？
  3. dw 为什么要先存每行局部贡献、再在 Python 侧 sum？

参考 Triton 官方 tutorials/06-layer-norm.py（LayerNorm bwd 结构完全相同）。
"""
from __future__ import annotations

import math
import torch
from torch import Tensor

import triton
import triton.language as tl


BLOCK_SIZE = 1024


@triton.jit
def _rmsnorm_fwd_kernel(
    X_ptr, W_ptr, Y_ptr, RSTD_ptr,
    stride_row,
    H,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    row_start = row * stride_row

    # --- Step 1: 计算 mean(x^2) ---
    mean_sq = tl.zeros([1], dtype=tl.float32)
    for block_start in range(0, H, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < H
        x = tl.load(X_ptr + row_start + offsets, mask=mask, other=0.0)
        x_f32 = x.to(tl.float32)
        mean_sq += tl.sum(x_f32 * x_f32, axis=0)
    mean_sq = mean_sq / H

    # --- Step 2: rstd ---
    rstd = 1.0 / tl.sqrt(mean_sq + eps)

    # --- Step 3: 保存 rstd ---
    tl.store(RSTD_ptr + row, rstd)

    # --- Step 4: normalize + scale ---
    for block_start in range(0, H, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < H
        x = tl.load(X_ptr + row_start + offsets, mask=mask, other=0.0)
        w = tl.load(W_ptr + offsets, mask=mask, other=1.0)
        x_f32 = x.to(tl.float32)
        y = x_f32 * rstd * w.to(tl.float32)
        tl.store(Y_ptr + row_start + offsets, y.to(x.dtype), mask=mask)


@triton.jit
def _rmsnorm_bwd_kernel(
    DY_ptr, X_ptr, W_ptr, RSTD_ptr,
    DX_ptr, DW_ptr,
    stride_row,
    H,
    BLOCK_SIZE: tl.constexpr,
):
    """bwd 梯度推导：
    y_i = x_i * rstd * w_i，rstd = (mean(x^2)+eps)^{-0.5}
    ∂L/∂x_i = Σ_j (∂L/∂y_j)(∂y_j/∂x_i)
             = dy_i * w_i * rstd   +   rstd^3 * x_i * (-1/H) * Σ_j(dy_j * w_j * x_j)
    整理：dx_i = rstd * (dy_i*w_i  -  (rstd^2/H) * dot(dy*w, x) * x_i)

    所以修正项 = (rstd^2 / H) * sum_j(dy_j * w_j * x_j)，是一个标量。
    需要第一次循环算出这个标量，第二次循环写出 dx。
    （可以合并：第一循环收集 dot，暂存 dx_partial；第二循环加修正——
     但寄存器压力大，教程风格是两次循环）
    """
    row = tl.program_id(0)
    row_start = row * stride_row

    rstd = tl.load(RSTD_ptr + row).to(tl.float32)

    # --- 第一次循环：计算 dot(dy*w, x)，同时算 dw 局部贡献 ---
    dot = tl.zeros([1], dtype=tl.float32)
    for block_start in range(0, H, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < H
        dy = tl.load(DY_ptr + row_start + offsets, mask=mask, other=0.0).to(tl.float32)
        x  = tl.load(X_ptr  + row_start + offsets, mask=mask, other=0.0).to(tl.float32)
        w  = tl.load(W_ptr  + offsets,             mask=mask, other=1.0).to(tl.float32)

        dot += tl.sum(dy * w * x, axis=0)

        # dw 局部贡献（fp32；Python 侧 sum over rows）
        dw_local = dy * x * rstd
        tl.store(DW_ptr + row_start + offsets, dw_local, mask=mask)

    # 修正项（标量）
    correction = (rstd * rstd / H) * dot

    # --- 第二次循环：写出 dx ---
    for block_start in range(0, H, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < H
        dy = tl.load(DY_ptr + row_start + offsets, mask=mask, other=0.0).to(tl.float32)
        x  = tl.load(X_ptr  + row_start + offsets, mask=mask, other=0.0).to(tl.float32)
        w  = tl.load(W_ptr  + offsets,             mask=mask, other=1.0).to(tl.float32)

        # 原始 dtype（为了 store）
        x_raw = tl.load(X_ptr + row_start + offsets, mask=mask, other=0.0)
        dx = rstd * (dy * w - correction * x)
        tl.store(DX_ptr + row_start + offsets, dx.to(x_raw.dtype), mask=mask)


class RMSNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: Tensor, weight: Tensor, eps: float) -> Tensor:
        shape_orig = x.shape
        H = shape_orig[-1]
        x_2d = x.contiguous().view(-1, H)
        weight_c = weight.contiguous()
        N = x_2d.shape[0]

        y = torch.empty_like(x_2d)
        rstd = torch.empty(N, dtype=torch.float32, device=x.device)

        grid = (N,)
        _rmsnorm_fwd_kernel[grid](
            x_2d, weight_c, y, rstd,
            x_2d.stride(0),
            H, eps,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        ctx.save_for_backward(x_2d, weight_c, rstd)
        ctx.H = H
        ctx.shape_orig = shape_orig
        return y.view(shape_orig)

    @staticmethod
    def backward(ctx, dy: Tensor):
        x_2d, weight, rstd = ctx.saved_tensors
        H = ctx.H
        N = x_2d.shape[0]

        dy_2d = dy.contiguous().view(N, H)
        dx = torch.empty_like(x_2d)
        # DW_ptr 需要 N×H 的 buffer（每行存 dw 局部贡献）
        dw_rows = torch.empty((N, H), dtype=torch.float32, device=x_2d.device)

        grid = (N,)
        _rmsnorm_bwd_kernel[grid](
            dy_2d, x_2d, weight, rstd,
            dx, dw_rows,
            x_2d.stride(0),
            H,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        dw = dw_rows.sum(dim=0).to(weight.dtype)
        return dx.view(ctx.shape_orig), dw, None


def rmsnorm(x: Tensor, weight: Tensor, eps: float = 1e-6) -> Tensor:
    """参考答案版 rmsnorm（与骨架接口相同）。"""
    return RMSNormFunction.apply(x, weight, eps)
