"""Phase 2.2 参考答案：单 launch Grouped GEMM + MoE FFN wrapper。

host 仅用向量操作生成每组 row-tile 数的前缀和。每个 Triton program 扫描这份
短前缀和，在 device 端确定所属 expert，再对该 expert 的 K 维做 fp32 累加。
"""
from __future__ import annotations

import torch
from torch import Tensor

import triton
import triton.language as tl

from kernels.swiglu import swiglu_mul


BLOCK_ROWS = 16
BLOCK_COLS = 32
BLOCK_K = 32


@triton.jit
def _grouped_gemm_kernel(
    X, W, OFFSETS, TILE_OFFSETS, OUT,
    N, E, K, M,
    BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr, BLOCK_K: tl.constexpr,
):
    tile_pid = tl.program_id(0)
    col_pid = tl.program_id(1)
    total_tiles = tl.load(TILE_OFFSETS + E)
    active = tile_pid < total_tiles

    # program_id 是标量 tensor；乘零得到同样的标量形态，便于后续动态指针寻址。
    group = tile_pid * 0
    for e in range(0, E):
        lo = tl.load(TILE_OFFSETS + e)
        hi = tl.load(TILE_OFFSETS + e + 1)
        group = tl.where(active & (tile_pid >= lo) & (tile_pid < hi), e, group)

    group_tile0 = tl.load(TILE_OFFSETS + group)
    row0 = tl.load(OFFSETS + group) + (tile_pid - group_tile0) * BLOCK_R
    row_end = tl.load(OFFSETS + group + 1)
    rows = row0 + tl.arange(0, BLOCK_R)
    cols = col_pid * BLOCK_C + tl.arange(0, BLOCK_C)
    acc = tl.zeros((BLOCK_R, BLOCK_C), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        kk = k0 + tl.arange(0, BLOCK_K)
        x_mask = active & (rows[:, None] < row_end) & (rows[:, None] < N) & (kk[None, :] < K)
        w_mask = active & (kk[:, None] < K) & (cols[None, :] < M)
        xv = tl.load(X + rows[:, None] * K + kk[None, :], mask=x_mask, other=0.0).to(tl.float32)
        wv = tl.load(W + group * K * M + kk[:, None] * M + cols[None, :],
                     mask=w_mask, other=0.0).to(tl.float32)
        acc += tl.dot(xv, wv, input_precision="ieee")

    store_mask = active & (rows[:, None] < row_end) & (rows[:, None] < N) & (cols[None, :] < M)
    tl.store(OUT + rows[:, None] * M + cols[None, :], acc, mask=store_mask)


def _validate(x: Tensor, weights: Tensor, group_offsets: Tensor) -> None:
    if x.ndim != 2 or weights.ndim != 3:
        raise ValueError("x 必须 2D，weights 必须 3D")
    if x.shape[1] != weights.shape[1]:
        raise ValueError("x.K 与 weights.K 不一致")
    if x.device != weights.device or x.dtype != weights.dtype:
        raise ValueError("x/weights 必须同 device、dtype")
    E = weights.shape[0]
    if group_offsets.ndim != 1 or group_offsets.numel() != E + 1:
        raise ValueError("group_offsets 长度必须是 E+1")
    if group_offsets.dtype not in (torch.int32, torch.int64):
        raise ValueError("group_offsets 必须是整数")
    if group_offsets.device != x.device:
        raise ValueError("group_offsets 必须与 x 同 device")
    if int(group_offsets[0].item()) != 0:
        raise ValueError("group_offsets[0] 必须为 0")
    if bool((group_offsets[1:] < group_offsets[:-1]).any().item()):
        raise ValueError("group_offsets 必须非递减")
    if int(group_offsets[-1].item()) != x.shape[0]:
        raise ValueError("group_offsets[-1] 必须等于 N_total")


def grouped_gemm(x: Tensor, weights: Tensor, group_offsets: Tensor) -> Tensor:
    _validate(x, weights, group_offsets)
    N, K = x.shape
    E, _, M = weights.shape
    sizes = group_offsets[1:] - group_offsets[:-1]
    tiles = torch.div(sizes + BLOCK_ROWS - 1, BLOCK_ROWS, rounding_mode="floor")
    tile_offsets = torch.cat((torch.zeros(1, device=x.device, dtype=tiles.dtype),
                              tiles.cumsum(0))).contiguous()
    out = torch.empty((N, M), device=x.device, dtype=x.dtype)
    # 紧上界 Σceil(n_e/BR) ≤ ceil(N/BR) + E；N+E 会超发 ~BR 倍无效 program。
    _grouped_gemm_kernel[(triton.cdiv(N, BLOCK_ROWS) + E, triton.cdiv(M, BLOCK_COLS))](
        x.contiguous(), weights.contiguous(), group_offsets.contiguous(),
        tile_offsets, out, N, E, K, M,
        BLOCK_R=BLOCK_ROWS, BLOCK_C=BLOCK_COLS, BLOCK_K=BLOCK_K,
    )
    return out


def moe_ffn_grouped(x, w_gate, w_up, w_down, group_offsets) -> Tensor:
    gate = grouped_gemm(x, w_gate, group_offsets)
    up = grouped_gemm(x, w_up, group_offsets)
    return grouped_gemm(swiglu_mul(gate, up), w_down, group_offsets)
