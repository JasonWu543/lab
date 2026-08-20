"""参考答案 — Fused Cross-Entropy Triton kernel（online softmax）

⚠️  先自己实现 kernels/cross_entropy.py，卡 30 分钟以上再看。
看完必须能回答：
  1. online softmax 的 rescale 公式是什么？为什么这样等价于正确的 logsumexp？
  2. bwd 里为什么可以用保存的 logsumexp 直接重算 softmax，而不需要重跑 online 算法？
  3. 如果不用 online 算法而是两 pass（先求 max 再求 sum），带宽开销差多少？
"""
from __future__ import annotations

import math
import torch
from torch import Tensor

import triton
import triton.language as tl


BLOCK_SIZE = 512


@triton.jit
def _ce_fwd_kernel(
    LOGITS_ptr, TARGETS_ptr,
    LOSS_ptr, LOGSUMEXP_ptr,
    stride_row,
    V,
    ignore_index,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    target = tl.load(TARGETS_ptr + row)
    is_ignored = (target == ignore_index)

    # online softmax：单 pass 维护 (row_max, row_sum)
    row_max = tl.full([1], float("-inf"), dtype=tl.float32)
    row_sum = tl.zeros([1], dtype=tl.float32)

    for block_start in range(0, V, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < V
        logit = tl.load(
            LOGITS_ptr + row * stride_row + offsets,
            mask=mask, other=float("-inf"),
        ).to(tl.float32)

        new_max = tl.maximum(row_max, tl.max(logit, axis=0))
        # rescale 旧 sum 再加新块的贡献
        row_sum = row_sum * tl.exp(row_max - new_max) + tl.sum(tl.exp(logit - new_max), axis=0)
        row_max = new_max

    logsumexp = row_max + tl.log(row_sum)

    # 取 logit[target]
    logit_target = tl.load(
        LOGITS_ptr + row * stride_row + target,
        mask=(target >= 0) & (target < V),
        other=0.0,
    ).to(tl.float32)

    loss = tl.where(is_ignored, 0.0, logsumexp - logit_target)
    tl.store(LOSS_ptr + row, loss)
    tl.store(LOGSUMEXP_ptr + row, logsumexp)


@triton.jit
def _ce_bwd_kernel(
    LOGITS_ptr, TARGETS_ptr, LOGSUMEXP_ptr,
    DLOGITS_ptr,
    grad_output,
    stride_row,
    V,
    ignore_index,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    target = tl.load(TARGETS_ptr + row)
    is_ignored = (target == ignore_index)
    logsumexp = tl.load(LOGSUMEXP_ptr + row).to(tl.float32)

    for block_start in range(0, V, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < V

        logit = tl.load(
            LOGITS_ptr + row * stride_row + offsets,
            mask=mask, other=0.0,
        ).to(tl.float32)

        # 利用保存的 logsumexp 重算 softmax（无需重跑 online 算法）
        softmax_val = tl.exp(logit - logsumexp)

        # 减去 one-hot
        is_target = (offsets == target)
        softmax_val = softmax_val - tl.where(is_target, 1.0, 0.0)

        dlogit = tl.where(is_ignored, 0.0, grad_output * softmax_val)

        # 正确的 tl.store：(ptr, value, mask=...)
        tl.store(DLOGITS_ptr + row * stride_row + offsets,
                 dlogit.to(DLOGITS_ptr.dtype.element_ty), mask=mask)


class FusedCrossEntropyFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, logits: Tensor, targets: Tensor, ignore_index: int) -> Tensor:
        assert logits.ndim == 2
        N, V = logits.shape
        logits_c = logits.contiguous()
        targets_c = targets.contiguous()

        loss_per_row = torch.empty(N, dtype=torch.float32, device=logits.device)
        logsumexp = torch.empty(N, dtype=torch.float32, device=logits.device)

        grid = (N,)
        _ce_fwd_kernel[grid](
            logits_c, targets_c,
            loss_per_row, logsumexp,
            logits_c.stride(0), V, ignore_index,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        valid_mask = targets_c != ignore_index
        N_valid = valid_mask.sum()
        loss = loss_per_row.sum() / N_valid

        ctx.save_for_backward(logits_c, targets_c, logsumexp)
        ctx.ignore_index = ignore_index
        ctx.N_valid = N_valid
        return loss

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        logits, targets, logsumexp = ctx.saved_tensors
        N, V = logits.shape
        dlogits = torch.empty_like(logits)
        n_valid = ctx.N_valid.item()
        # Match torch.nn.functional.cross_entropy for an all-ignored batch:
        # the mean loss is NaN (0 / 0), but every input gradient is zero.
        grad_scale = 0.0 if n_valid == 0 else grad_output.item() / n_valid

        grid = (N,)
        _ce_bwd_kernel[grid](
            logits, targets, logsumexp,
            dlogits, grad_scale,
            logits.stride(0), V, ctx.ignore_index,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return dlogits, None, None


def fused_cross_entropy(
    logits: Tensor,
    targets: Tensor,
    ignore_index: int = -100,
) -> Tensor:
    """参考答案版 fused_cross_entropy（与骨架接口相同）。"""
    return FusedCrossEntropyFunction.apply(logits, targets, ignore_index)
