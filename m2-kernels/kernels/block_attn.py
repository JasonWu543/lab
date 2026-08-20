"""Phase 2.1 学生任务：online-softmax block attention forward kernel。"""
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
    """一个 program 负责一个 (batch*head, query block)。

    思考：每读入一个 K/V block 后，旧分母和旧 accumulator 为什么必须按新的
    running max 重标度？尾块和 causal 的两层 mask 分别作用在哪里？
    """
    raise NotImplementedError("实现 online-softmax block attention")


class BlockAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale, BLOCK_M, BLOCK_N):
        q_c, k_c, v_c = q.contiguous(), k.contiguous(), v.contiguous()
        out = torch.empty_like(q_c)
        B, H, T, D = q_c.shape
        grid = (B * H, triton.cdiv(T, BLOCK_M))
        _block_attn_fwd[grid](
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
            qr = q.detach().requires_grad_(True)
            kr = k.detach().requires_grad_(True)
            vr = v.detach().requires_grad_(True)
            y = F.scaled_dot_product_attention(
                qr, kr, vr, is_causal=ctx.causal, scale=ctx.sm_scale,
            )
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
