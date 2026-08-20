"""Phase 2.2 学生任务：MoE Grouped GEMM Triton kernel。

offset 校验、tile 前缀和准备和完整 FFN wrapper 已给；学生只实现 device 端
program→group 查找与分块矩阵乘。
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
    """一个 program 计算某 expert 的一个 row×column tile。

    思考：不同 expert 的 row tile 数不同，如何用 tile 前缀和在 device 端找到
    program 所属的 group？空组为何自然占用零个 tile？
    """
    raise NotImplementedError("实现 grouped GEMM kernel")


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
    # tile 总数 Σceil(n_e/BR) ≤ ceil(N/BR) + E（每组最多多出一个不满 tile），
    # 用这个紧上界做 grid；多余 program 由 TILE_OFFSETS[E] mask。
    # 不用 N+E：那会超发 ~BLOCK_ROWS 倍 program，且 inactive program 仍跑
    # 完整 K 循环，benchmark 会被这个 artifact 打爆。
    grid = (triton.cdiv(N, BLOCK_ROWS) + E, triton.cdiv(M, BLOCK_COLS))
    _grouped_gemm_kernel[grid](
        x.contiguous(), weights.contiguous(), group_offsets.contiguous(),
        tile_offsets, out, N, E, K, M,
        BLOCK_R=BLOCK_ROWS, BLOCK_C=BLOCK_COLS, BLOCK_K=BLOCK_K,
    )
    return out


def moe_ffn_grouped(x, w_gate, w_up, w_down, group_offsets) -> Tensor:
    gate = grouped_gemm(x, w_gate, group_offsets)
    up = grouped_gemm(x, w_up, group_offsets)
    hidden = swiglu_mul(gate, up)
    return grouped_gemm(hidden, w_down, group_offsets)
