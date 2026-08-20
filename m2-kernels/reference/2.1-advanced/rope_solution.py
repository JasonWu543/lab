"""Phase 2.1 参考答案：前半/后半配对的 fused RoPE。

每对 (x1,x2) 左乘 R(theta)。反传左乘 R(theta)^T=R(-theta)，因此同一
kernel 只需把 sin 的符号取反即可复用。
"""
from __future__ import annotations

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
    pair = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = pair < N_pairs
    dim = pair % HALF_D
    vector = pair // HALF_D
    token = vector % T
    base = vector * (2 * HALF_D)
    lo = base + dim
    hi = lo + HALF_D

    c = tl.load(COS_ptr + token * HALF_D + dim, mask=mask, other=1.0).to(tl.float32)
    s = tl.load(SIN_ptr + token * HALF_D + dim, mask=mask, other=0.0).to(tl.float32) * sign

    q1 = tl.load(Q_ptr + lo, mask=mask, other=0.0).to(tl.float32)
    q2 = tl.load(Q_ptr + hi, mask=mask, other=0.0).to(tl.float32)
    k1 = tl.load(K_ptr + lo, mask=mask, other=0.0).to(tl.float32)
    k2 = tl.load(K_ptr + hi, mask=mask, other=0.0).to(tl.float32)

    # 显式 cast 回输出指针 dtype（bf16 时不依赖 Triton 隐式转换——2.0 的前车之鉴）
    tl.store(OQ_ptr + lo, (q1 * c - q2 * s).to(OQ_ptr.dtype.element_ty), mask=mask)
    tl.store(OQ_ptr + hi, (q2 * c + q1 * s).to(OQ_ptr.dtype.element_ty), mask=mask)
    tl.store(OK_ptr + lo, (k1 * c - k2 * s).to(OK_ptr.dtype.element_ty), mask=mask)
    tl.store(OK_ptr + hi, (k2 * c + k1 * s).to(OK_ptr.dtype.element_ty), mask=mask)


class RoPEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: Tensor, k: Tensor, cos: Tensor, sin: Tensor):
        q_c, k_c = q.contiguous(), k.contiguous()
        cos_c = cos.to(device=q.device, dtype=torch.float32).contiguous()
        sin_c = sin.to(device=q.device, dtype=torch.float32).contiguous()
        oq, ok = torch.empty_like(q_c), torch.empty_like(k_c)
        half, n_pairs = q.shape[-1] // 2, q.numel() // 2
        _rope_kernel[(triton.cdiv(n_pairs, BLOCK_PAIRS),)](
            q_c, k_c, cos_c, sin_c, oq, ok, n_pairs, q.shape[-2], half, 1.0,
            BLOCK=BLOCK_PAIRS,
        )
        ctx.save_for_backward(cos_c, sin_c)
        ctx.shape = q.shape
        return oq, ok

    @staticmethod
    def backward(ctx, grad_q: Tensor, grad_k: Tensor):
        cos, sin = ctx.saved_tensors
        gq, gk = grad_q.contiguous(), grad_k.contiguous()
        dq, dk = torch.empty_like(gq), torch.empty_like(gk)
        half, n_pairs = ctx.shape[-1] // 2, gq.numel() // 2
        _rope_kernel[(triton.cdiv(n_pairs, BLOCK_PAIRS),)](
            gq, gk, cos, sin, dq, dk, n_pairs, ctx.shape[-2], half, -1.0,
            BLOCK=BLOCK_PAIRS,
        )
        return dq, dk, None, None


def _validate(q, k, cos, sin):
    if q.shape != k.shape or q.ndim != 4:
        raise ValueError("q/k 必须同为 (B,H,T,D)")
    if q.device != k.device or q.dtype != k.dtype:
        raise ValueError("q/k 必须同 device、dtype")
    if q.shape[-1] % 2:
        raise ValueError("RoPE head dim 必须为偶数")
    if cos.shape != (q.shape[-2], q.shape[-1] // 2) or sin.shape != cos.shape:
        raise ValueError("cos/sin shape 不匹配")


def rope_fwd(q, k, cos, sin):
    _validate(q, k, cos, sin)
    with torch.no_grad():
        return RoPEFunction.apply(q, k, cos, sin)


def apply_rope(q, k, cos, sin):
    _validate(q, k, cos, sin)
    return RoPEFunction.apply(q, k, cos, sin)
