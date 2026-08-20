"""Phase 2.0 — 学生实现文件：RMSNorm fused kernel

任务：实现 RMSNorm 的 Triton fwd + bwd kernel，通过 tests/test_kernels.py T1。

语义参考（来自 m1-foundation MiniLM 1.2）：
    y = x / rms(x) * weight
    rms(x) = sqrt(mean(x^2) + eps)
    输出 dtype = 输入 dtype；内部累加在 fp32。

闯关顺序建议：
  Step 1  先读 Triton layer-norm tutorial（官方仓库 tutorials/06-layer-norm.py）
           理解 fwd：按行划分 program，循环分块 tl.load → fp32 累加 sum(x^2)
  Step 2  实现 _rmsnorm_fwd_kernel，让 fwd 数值对齐
  Step 3  读懂 bwd 需要哪些 saved tensor（提示：rstd = 1/rms(x)，以及 x 本身）
  Step 4  实现 _rmsnorm_bwd_kernel：
           思考：bwd 要对每一行的 x_i 算梯度；你需要先做一次分块求和（点积），
           再做第二次分块写出 dx；两次循环还是合一？参考 tutorial bwd 结构。
  Step 5  包装进 RMSNormFunction，暴露 rmsnorm()，跑 T1

梯度引导（不直接给公式，自己推）：
  - y_i = x_i * rstd * w_i，其中 rstd = (mean(x^2)+eps)^{-0.5}
  - d_weight = sum over rows of (dy * x * rstd)     ← 这一行可以直接给，是 reduction
  - dx 的推导：先写出 dy/dx_i 的链式法则，注意 rstd 依赖整行 x，
    所以 dx_i 里会出现一个 「全局」修正项（思考：它是一个标量乘以 x_i）
    自己推出这个修正项的表达式，再实现

运行测试：
    cd m2-kernels && python3 -m pytest tests/test_kernels.py::test_rmsnorm -x -q

卡住了再看：reference/2.0-kernels/rmsnorm_solution.py
（先自己写；看完要能说出 bwd 修正项是什么、为什么要两次分块归约）
"""
from __future__ import annotations

import torch
from torch import Tensor

# triton 只在 Linux+CUDA 可用；Mac/CPU 下 import 会失败，由调用方（测试）处理
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# 超参数
# ---------------------------------------------------------------------------

# 每个 program 每次处理的列数。
# 选择依据：SRAM（L1 cache / shared mem）足够大、且是 warp 宽度（32）的倍数。
# BLOCK_SIZE 太小 → 每行启动太多 loop iteration，overhead 大；
# BLOCK_SIZE 太大 → 寄存器溢出（register spill）。
# 此处固定 1024，非 2 幂的 H 通过 mask 处理。
BLOCK_SIZE = 1024


# ---------------------------------------------------------------------------
# Forward kernel（学生实现）
# ---------------------------------------------------------------------------

@triton.jit
def _rmsnorm_fwd_kernel(
    X_ptr, W_ptr, Y_ptr, RSTD_ptr,
    stride_row,     # X/Y 相邻两行之间的元素间距（= H）
    H,              # 每行的列数
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """每个 program 处理一行 x[row, :]。

    参数说明：
      X_ptr     — 输入矩阵基址（已展平为 2D：N_rows × H）
      W_ptr     — weight 向量基址（长度 H）
      Y_ptr     — 输出矩阵基址
      RSTD_ptr  — 保存每行的 rstd（= 1/rms）供 bwd 使用
      stride_row — 行步长（单位：元素数，= H）
      H         — 每行列数
      eps       — 数值稳定项
      BLOCK_SIZE — 编译时常量：每次处理的列数

    TODO：实现以下逻辑（伪代码框架，细节自己填）：

        row = tl.program_id(0)
        row_start = row * stride_row

        # --- Step 1: 计算 mean(x^2) ---
        # 提示：用 offsets + mask 做分块循环；fp32 累加
        mean_sq = tl.zeros([1], dtype=tl.float32)
        for block_start in range(0, H, BLOCK_SIZE):
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < H
            x = tl.load(X_ptr + row_start + offsets, mask=mask, other=0.0)
            x_f32 = x.to(tl.float32)
            mean_sq += tl.sum(x_f32 * x_f32, axis=0)
        mean_sq = mean_sq / H

        # --- Step 2: 计算 rstd ---
        # 思考：rstd = ?  (eps 加在哪里？)
        rstd = ...

        # --- Step 3: 保存 rstd 到 RSTD_ptr[row] ---
        ...

        # --- Step 4: 逐块 normalize + scale，写入 Y ---
        for block_start in range(0, H, BLOCK_SIZE):
            offsets = ...
            mask = ...
            x = tl.load(...)
            w = tl.load(W_ptr + offsets, mask=mask, other=1.0)
            x_f32 = x.to(tl.float32)
            y = x_f32 * rstd * w.to(tl.float32)
            tl.store(Y_ptr + row_start + offsets, y.to(x.dtype), mask=mask)
    """
    # TODO: 把上方伪代码翻译成真实 Triton 代码
    raise NotImplementedError("_rmsnorm_fwd_kernel — 学生实现")


# ---------------------------------------------------------------------------
# Backward kernel（学生实现）
# ---------------------------------------------------------------------------

@triton.jit
def _rmsnorm_bwd_kernel(
    DY_ptr, X_ptr, W_ptr, RSTD_ptr,
    DX_ptr, DW_ptr,
    stride_row,
    H,
    BLOCK_SIZE: tl.constexpr,
):
    """每个 program 处理一行的 dx 和对 dw 的局部贡献。

    已知量（来自 fwd saved tensors）：dy, x, w, rstd
    需要算：
      - dx（每行）
      - dw 的局部贡献（每行 dy * x * rstd）→ 之后在 Python 侧 sum over rows

    梯度推导框架（不给公式，引导推导）：
      1) y_i = x_i * rstd * w_i，把 y 看成 x 的函数（rstd 也依赖 x）
      2) 用链式法则：∂L/∂x_i = Σ_j (∂L/∂y_j)(∂y_j/∂x_i)
         注意 y_j 只通过 rstd = (mean(x^2)+eps)^{-0.5} 依赖 x_i（对 j≠i 也有贡献！）
      3) 把 ∂rstd/∂x_i 算出来 → 会出现一个 「全行 x 的点积」 修正项
      4) 整理成：dx_i = rstd * (dy_i * w_i  -  [修正项] * x_i)
         自己推出 [修正项] 是什么

    TODO：
      Step 1 — 第一次分块循环：计算「全行修正项」（一个标量，涉及 sum(dy * w * x)）
               提示：修正项 = 某个系数 * sum_j(dy_j * w_j * x_j * rstd^2) * rstd
               （推出来是什么系数？）
      Step 2 — 计算 dw 局部贡献（可以合在 Step 1 循环里）
               dw_local[col] = dy[col] * x[col] * rstd
               之后在 Python 侧把所有行的 dw_local 求和得到真正的 dw
      Step 3 — 第二次分块循环（也可与 Step1 合并，需要寄存器够）：
               写出 dx[col] = rstd * (dy[col]*w[col] - correction * x[col])
    """
    # TODO: 实现
    raise NotImplementedError("_rmsnorm_bwd_kernel — 学生实现")


# ---------------------------------------------------------------------------
# autograd.Function 包装（脚手架，已完整；不需要修改）
# ---------------------------------------------------------------------------

class RMSNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: Tensor, weight: Tensor, eps: float) -> Tensor:
        # 展平到 2D：(..., H) → (N_rows, H)
        shape_orig = x.shape
        H = shape_orig[-1]
        x_2d = x.contiguous().view(-1, H)
        # kernel 的 W_ptr 按连续元素寻址；保留非连续 weight 的公开接口语义。
        weight_c = weight.contiguous()
        N = x_2d.shape[0]

        y = torch.empty_like(x_2d)
        rstd = torch.empty(N, dtype=torch.float32, device=x.device)

        # grid：每行一个 program
        grid = (N,)
        _rmsnorm_fwd_kernel[grid](
            x_2d, weight_c, y, rstd,
            x_2d.stride(0),   # stride_row = H（连续张量）
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
        # dw 先存每行贡献，之后 sum；这里用 float32 累加避免精度丢失
        dw_rows = torch.empty((N, H), dtype=torch.float32, device=x_2d.device)

        grid = (N,)
        _rmsnorm_bwd_kernel[grid](
            dy_2d, x_2d, weight, rstd,
            dx, dw_rows,
            x_2d.stride(0),
            H,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        # sum over rows → dw，再转回 weight 的 dtype
        dw = dw_rows.sum(dim=0).to(weight.dtype)
        return dx.view(ctx.shape_orig), dw, None   # eps 无梯度


# ---------------------------------------------------------------------------
# 对外接口（SPEC 冻结签名）
# ---------------------------------------------------------------------------

def rmsnorm(x: Tensor, weight: Tensor, eps: float = 1e-6) -> Tensor:
    """RMSNorm fused Triton kernel。

    参数：
        x      — (..., H) 任意前缀维度
        weight — (H,) 可学习缩放参数
        eps    — 数值稳定项（默认 1e-6）

    返回：与 x 同 shape / dtype 的归一化张量。
    约定：内部以 fp32 累加；输出 dtype = 输入 dtype。
    """
    return RMSNormFunction.apply(x, weight, eps)
