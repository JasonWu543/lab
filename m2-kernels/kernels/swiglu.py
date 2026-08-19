"""Phase 2.0 — 学生实现文件：SwiGLU fused kernel

任务：实现 swiglu_mul(gate, up) 的 Triton fwd + bwd kernel，通过 T2。

语义：
    fwd: out = silu(gate) * up
         silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
    bwd: 同时算 d_gate 和 d_up（单 kernel，两组 tl.store）

为什么 fuse？
    naive 写法需要：读 gate → 写 silu_gate → 读 silu_gate,up → 写 out
    fused 写法：读 gate,up → 写 out（中间结果 silu_gate 不落显存）
    bwd 同理：节省一次 silu_gate 的读写。

闯关顺序建议：
  Step 1  实现 fwd kernel（逐元素，最简单的 Triton pattern）
           grid = (ceil(N_total / BLOCK_SIZE),)，没有行循环，直接 1D 分块
  Step 2  实现 bwd kernel：
           先自己推 d_gate 和 d_up 的表达式（引导见下方），再写 kernel
  Step 3  包装 SwiGLUFunction，暴露 swiglu_mul()，跑 T2

梯度引导（推导练习，不直接给结果）：
    out = silu(gate) * up
    - d_up   = dy * silu(gate)                  ← 容易，直接写出
    - d_gate = dy * up * d/dx[silu(x)]|_{x=gate}
               d/dx silu(x) 自己推：silu = x·σ(x)，用乘积法则
               答案里会有 σ(gate) 和 σ(gate)*(1-σ(gate)) 两项，整理一下

bwd 注意：sigma = sigmoid(gate) 需要在 bwd kernel 里重新计算（fwd 没有 save 它）；
这是逐元素操作，重算一次 sigmoid 代价极小（节省了显存带宽）。

运行测试：
    cd m2-kernels && python3 -m pytest tests/test_kernels.py::test_swiglu -x -q

卡住了再看：reference/2.0-kernels/swiglu_solution.py
"""
from __future__ import annotations

import math
import torch
from torch import Tensor

import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# 超参数
# ---------------------------------------------------------------------------

# SwiGLU 是纯逐元素操作，BLOCK_SIZE 选大一些以提高 memory throughput。
# 1024 在大多数 GPU 上是较优选择；如需 autotune 可尝试 {512, 1024, 2048, 4096}。
BLOCK_SIZE = 1024


# ---------------------------------------------------------------------------
# Forward kernel（学生实现）
# ---------------------------------------------------------------------------

@triton.jit
def _swiglu_fwd_kernel(
    GATE_ptr, UP_ptr, OUT_ptr,
    N_total,            # gate/up 的总元素数（= batch * seq * ffn_dim）
    BLOCK_SIZE: tl.constexpr,
):
    """逐元素 fused silu(gate) * up。

    grid = (ceil(N_total / BLOCK_SIZE),)  — 每个 program 处理 BLOCK_SIZE 个元素

    TODO：实现以下逻辑

        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N_total

        gate = tl.load(GATE_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        up   = tl.load(UP_ptr   + offsets, mask=mask, other=0.0).to(tl.float32)

        # --- silu(gate) ---
        # 提示：sigmoid 在 Triton 里没有内置，用 1/(1+exp(-x)) 自己算
        sigma = ...
        silu_gate = ...

        out = silu_gate * up

        # 注意：输出需要转回原始 dtype；从 gate 推断 dtype
        # tl.load 返回的是 fp32（已转），但 store 时要转回去
        # 骨架里传入了原始指针，dtype 由调用方（Python 侧）决定
        tl.store(OUT_ptr + offsets, out.to(gate_orig_dtype), mask=mask)
        # ↑ 实际实现：需要先 load 原始 dtype 的 gate，记下 dtype，再 store
    """
    # TODO: 实现
    raise NotImplementedError("_swiglu_fwd_kernel — 学生实现")


# ---------------------------------------------------------------------------
# Backward kernel（学生实现）
# ---------------------------------------------------------------------------

@triton.jit
def _swiglu_bwd_kernel(
    DY_ptr, GATE_ptr, UP_ptr,
    DGATE_ptr, DUP_ptr,
    N_total,
    BLOCK_SIZE: tl.constexpr,
):
    """bwd：同时计算 d_gate 和 d_up，单 kernel 单 pass。

    TODO：根据上方引导，推出 d_gate 和 d_up 后在此实现。

    提示模式（框架代码，细节自填）：

        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N_total

        dy   = tl.load(DY_ptr   + offsets, mask=mask, other=0.0).to(tl.float32)
        gate = tl.load(GATE_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        up   = tl.load(UP_ptr   + offsets, mask=mask, other=0.0).to(tl.float32)

        sigma = 1.0 / (1.0 + tl.exp(-gate))
        silu_gate = gate * sigma

        d_up   = ...     # 用 dy 和 silu_gate
        d_gate = ...     # 用 dy, up, sigma, gate — 注意乘积法则

        tl.store(DGATE_ptr + offsets, d_gate.to(...), mask=mask)
        tl.store(DUP_ptr   + offsets, d_up.to(...),   mask=mask)
    """
    # TODO: 实现
    raise NotImplementedError("_swiglu_bwd_kernel — 学生实现")


# ---------------------------------------------------------------------------
# autograd.Function 包装（脚手架，已完整；不需要修改）
# ---------------------------------------------------------------------------

class SwiGLUFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, gate: Tensor, up: Tensor) -> Tensor:
        assert gate.shape == up.shape, "gate 和 up 形状必须相同"
        gate_c = gate.contiguous()
        up_c = up.contiguous()
        out = torch.empty_like(gate_c)
        N_total = gate_c.numel()

        grid = (math.ceil(N_total / BLOCK_SIZE),)
        _swiglu_fwd_kernel[grid](
            gate_c, up_c, out,
            N_total,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        ctx.save_for_backward(gate_c, up_c)
        return out

    @staticmethod
    def backward(ctx, dy: Tensor):
        gate, up = ctx.saved_tensors
        dy_c = dy.contiguous()
        d_gate = torch.empty_like(gate)
        d_up = torch.empty_like(up)
        N_total = gate.numel()

        grid = (math.ceil(N_total / BLOCK_SIZE),)
        _swiglu_bwd_kernel[grid](
            dy_c, gate, up,
            d_gate, d_up,
            N_total,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return d_gate, d_up


# ---------------------------------------------------------------------------
# 对外接口（SPEC 冻结签名）
# ---------------------------------------------------------------------------

def swiglu_mul(gate: Tensor, up: Tensor) -> Tensor:
    """Fused SwiGLU：silu(gate) * up。

    参数：
        gate — (..., D) 门控分支（经过 W_gate 的线性输出）
        up   — (..., D) 直通分支（经过 W_up 的线性输出），与 gate 同 shape

    返回：与 gate 同 shape / dtype 的张量。
    注意：两侧的 matmul 留在 torch；本 kernel 只做逐元素 fused 部分。
    """
    return SwiGLUFunction.apply(gate, up)
