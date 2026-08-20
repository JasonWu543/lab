"""Phase 2.1 参考答案：不物化 score 的 blockwise online softmax attention。"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

import triton
import triton.language as tl


@triton.jit
def _block_attn_fwd(
    Q, K, V, O, T, D, scale, causal: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    bh = tl.program_id(0)
    m_block = tl.program_id(1)
    offs_m = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    base = bh * T * D
    q_mask = (offs_m[:, None] < T) & (offs_d[None, :] < D)
    q = tl.load(Q + base + offs_m[:, None] * D + offs_d[None, :],
                mask=q_mask, other=0.0).to(tl.float32)

    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)

    for start_n in range(0, T, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        kv_mask = (offs_n[:, None] < T) & (offs_d[None, :] < D)
        k = tl.load(K + base + offs_n[:, None] * D + offs_d[None, :],
                    mask=kv_mask, other=0.0).to(tl.float32)
        v = tl.load(V + base + offs_n[:, None] * D + offs_d[None, :],
                    mask=kv_mask, other=0.0).to(tl.float32)
        # fp32 验收门槛是 1e-5；禁用 Ampere 默认 TF32 截断，使用 IEEE fp32 点积。
        scores = tl.dot(q, tl.trans(k), input_precision="ieee") * scale
        valid = (offs_m[:, None] < T) & (offs_n[None, :] < T)
        if causal:
            valid = valid & (offs_m[:, None] >= offs_n[None, :])
        scores = tl.where(valid, scores, -float("inf"))

        block_max = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, block_max)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])
        p = tl.where(valid, p, 0.0)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(
            p.to(tl.float32), v, input_precision="ieee"
        )
        m_i = m_new

    out = (acc / l_i[:, None]).to(O.dtype.element_ty)
    tl.store(O + base + offs_m[:, None] * D + offs_d[None, :], out,
             mask=q_mask)


class BlockAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale, BLOCK_M, BLOCK_N):
        q_c, k_c, v_c = q.contiguous(), k.contiguous(), v.contiguous()
        out = torch.empty_like(q_c)
        B, H, T, D = q_c.shape
        _block_attn_fwd[(B * H, triton.cdiv(T, BLOCK_M))](
            q_c, k_c, v_c, out, T, D, sm_scale, causal=causal,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=D,
        )
        ctx.save_for_backward(q_c, k_c, v_c)
        ctx.causal, ctx.sm_scale = causal, sm_scale
        return out

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v = ctx.saved_tensors
        with torch.enable_grad():
            qr, kr, vr = (x.detach().requires_grad_(True) for x in (q, k, v))
            y = F.scaled_dot_product_attention(qr, kr, vr,
                                               is_causal=ctx.causal,
                                               scale=ctx.sm_scale)
            grads = torch.autograd.grad(y, (qr, kr, vr), grad_out)
        return *grads, None, None, None, None


def block_attention(q: Tensor, k: Tensor, v: Tensor, causal: bool = True,
                    sm_scale: float | None = None,
                    BLOCK_M: int = 64, BLOCK_N: int = 64) -> Tensor:
    if q.shape != k.shape or q.shape != v.shape or q.ndim != 4:
        raise ValueError("q/k/v 必须同为 (B,H,T,D)")
    if q.dtype != k.dtype or q.dtype != v.dtype or q.device != k.device or q.device != v.device:
        raise ValueError("q/k/v 必须同 device、dtype")
    if q.shape[-1] not in (32, 64, 128):
        raise ValueError("D 必须是 32/64/128")
    scale = 1.0 / math.sqrt(q.shape[-1]) if sm_scale is None else float(sm_scale)
    return BlockAttentionFunction.apply(q, k, v, causal, scale, BLOCK_M, BLOCK_N)
