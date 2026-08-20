"""Phase 2.1 学生任务：Fused RoPE forward/backward Triton kernel。

wrapper 与 autograd.Function 已给。学生只实现两个 kernel；前半/后半维配对，
中间计算使用 fp32，尾部 program 必须 mask。
"""
from __future__ import annotations

import math

import torch
from torch import Tensor

import triton
import triton.language as tl


BLOCK_PAIRS = 256


@triton.jit
def _rope_kernel(
    Q_ptr, K_ptr, COS_ptr, SIN_ptr, OQ_ptr, OK_ptr,
    N_pairs, T, HALF_D, sign,
    BLOCK: tl.constexpr,
):
    """每个 program 处理若干 (向量, 维度对)。

    思考：如何从扁平 pair 下标还原 token 位置与两个半维元素？旋转矩阵的
    转置和把角度取负有什么关系？
    """
    raise NotImplementedError("实现 fused RoPE kernel")


class RoPEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: Tensor, k: Tensor, cos: Tensor, sin: Tensor):
        q_c, k_c = q.contiguous(), k.contiguous()
        cos_c = cos.to(device=q.device, dtype=torch.float32).contiguous()
        sin_c = sin.to(device=q.device, dtype=torch.float32).contiguous()
        oq, ok = torch.empty_like(q_c), torch.empty_like(k_c)
        half = q.shape[-1] // 2
        n_pairs = q.numel() // 2
        _rope_kernel[(triton.cdiv(n_pairs, BLOCK_PAIRS),)](
            q_c, k_c, cos_c, sin_c, oq, ok,
            n_pairs, q.shape[-2], half, 1.0, BLOCK=BLOCK_PAIRS,
        )
        ctx.save_for_backward(cos_c, sin_c)
        ctx.shape = q.shape
        return oq, ok

    @staticmethod
    def backward(ctx, grad_q: Tensor, grad_k: Tensor):
        cos, sin = ctx.saved_tensors
        gq, gk = grad_q.contiguous(), grad_k.contiguous()
        dq, dk = torch.empty_like(gq), torch.empty_like(gk)
        half = ctx.shape[-1] // 2
        n_pairs = gq.numel() // 2
        _rope_kernel[(triton.cdiv(n_pairs, BLOCK_PAIRS),)](
            gq, gk, cos, sin, dq, dk,
            n_pairs, ctx.shape[-2], half, -1.0, BLOCK=BLOCK_PAIRS,
        )
        return dq, dk, None, None


def _validate(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> None:
    if q.shape != k.shape or q.ndim != 4:
        raise ValueError("q/k 必须同为 (B,H,T,D)")
    if q.device != k.device or q.dtype != k.dtype:
        raise ValueError("q/k 必须同 device、dtype")
    if q.shape[-1] % 2:
        raise ValueError("RoPE head dim 必须为偶数")
    expected = (q.shape[-2], q.shape[-1] // 2)
    if cos.shape != expected or sin.shape != expected:
        raise ValueError(f"cos/sin 必须为 {expected}")


def rope_fwd(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    _validate(q, k, cos, sin)
    return RoPEFunction.forward(_ForwardOnlyContext(), q, k, cos, sin)


class _ForwardOnlyContext:
    """仅供无 autograd 的公开 rope_fwd 复用 wrapper。"""
    def save_for_backward(self, *args):
        self.saved_tensors = args


def apply_rope(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor):
    _validate(q, k, cos, sin)
    return RoPEFunction.apply(q, k, cos, sin)
