"""Phase 2.0 — 学生实现文件：Fused Cross-Entropy kernel

任务：实现 fused_cross_entropy(logits, targets) 的 Triton fwd + bwd kernel，
      通过 tests/test_kernels.py T3/T4/T5。

语义：
    fwd: loss = mean(  -logits[i, t_i] + log(sum_j exp(logits[i,j]))  )
                其中 t_i = targets[i]，ignore_index 的行不计入 mean
    bwd: dlogits[i, j] = softmax(logits[i])[j] - 1{j == t_i}
                         （同样 ignore_index 的行 dlogits = 0）

    关键：fwd 用 online softmax（单 pass 同时维护 row_max 和 logsumexp），
          不物化 (N, V) 的 probs，显存节省 ≈ 4*N*V bytes。

为什么这是显存收益最大的 kernel（T4 考的）：
    torch.nn.CrossEntropyLoss 内部要算 softmax(logits)，即物化 (N, V) float32 矩阵；
    N=4096, V=32768 时 = 4096*32768*4 bytes ≈ 512 MB。
    fused 版本只需保存 logsumexp (N,) 和 logits 本身供 bwd，
    不存 probs，峰值显存大幅降低。

online softmax（U 的核心推导，公式自己推，否则 T5 无法通过）：
    问题：直接算 log(sum exp(logits)) 在 logits 含大值时会 overflow；
         经典解法 max-then-sum 需要读两遍数据（两个 pass）。
    你的任务：设计单 pass 递推——维护一对状态 (running_max, running_sum)，
    每处理一个新分块就更新它们，结束时能恢复出正确的 logsumexp。
    推导入口：已知前 k 块的 (m, s) 满足「s = Σ exp(x_i - m)」，
    新块到来使 max 变大时，旧的 s 怎么无损地换算到新 max 下？
    自查：用两个小数组手算，合并后的 logsumexp 必须等于一次性计算的值。
    （这个技巧是 FlashAttention 的基石之一，推明白它 M2 就值回票价）

    bwd 中重算 softmax：想想 softmax[j] 能否只用 logits[j] 和保存的
    logsumexp[row] 两个数表示？（这就是 fwd 只需保存 (N,) 向量的原因）

闯关顺序建议：
  Step 1  实现 _ce_fwd_kernel：online softmax 单 pass，输出 loss 和 logsumexp
  Step 2  测试 T5（数值稳定性）先通
  Step 3  实现 _ce_bwd_kernel：利用 saved logsumexp 重算 softmax，原地写 dlogits
  Step 4  跑 T3/T4（T4 用 torch.cuda.max_memory_allocated 比较）

运行测试：
    cd m2-kernels && python3 -m pytest tests/test_kernels.py -k "cross_entropy" -x -q

卡住了再看：reference/2.0-kernels/cross_entropy_solution.py
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

# V（vocab size）通常很大（32k~128k），BLOCK_SIZE 决定每次分块处理多少列。
# 建议从 512/1024 开始；V 不是 BLOCK_SIZE 整数倍时靠 mask 补齐。
BLOCK_SIZE = 512


# ---------------------------------------------------------------------------
# Forward kernel（学生实现）
# ---------------------------------------------------------------------------

@triton.jit
def _ce_fwd_kernel(
    LOGITS_ptr, TARGETS_ptr,
    LOSS_ptr, LOGSUMEXP_ptr,
    stride_row,     # logits 相邻两行的元素间距（= V）
    V,              # vocab size
    ignore_index,
    BLOCK_SIZE: tl.constexpr,
):
    """每个 program 处理一行（一个 token）的 fwd loss。

    输出：
        LOSS_ptr[row]      — 该行的 cross-entropy loss（ignore_index 行置 0）
        LOGSUMEXP_ptr[row] — 该行的 logsumexp（供 bwd 使用）

    实现框架（结构给你，核心递推自己推）：

        row = tl.program_id(0)
        target = tl.load(TARGETS_ptr + row)
        is_ignored = (target == ignore_index)

        # --- Step 1: online softmax 单 pass ---
        # 初始化 running 状态（Triton 写法：tl.full([1], float('-inf'), tl.float32)
        # 和 tl.zeros([1], tl.float32)）
        for block_start in range(0, V, BLOCK_SIZE):
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < V
            logit = tl.load(LOGITS_ptr + row * stride_row + offsets,
                            mask=mask, other=-float('inf')).to(tl.float32)
            # TODO: 用文件头推导出的递推式更新 (running_max, running_sum)
            #       块内归约用 tl.max(logit, axis=0) / tl.sum(..., axis=0)
        # TODO: 由最终 (running_max, running_sum) 恢复 logsumexp

        # --- Step 2: 取 logit[target] ---
        # 单独 load 第 target 列；mask 做 boundary check（ignore_index 是 -100，
        # 直接当下标去 load 会越界——想想 mask 该写什么）

        # --- Step 3: 计算 loss 并写出 ---
        # loss 语义见文件头；ignore_index 行 loss = 0，
        # 但 logsumexp 照常存（bwd 里对 ignore 行另行判断）
        # tl.store 签名：tl.store(ptr, value, mask=...)
    """
    # TODO: 按上述框架实现
    raise NotImplementedError("_ce_fwd_kernel — 学生实现")


# ---------------------------------------------------------------------------
# Backward kernel（学生实现）
# ---------------------------------------------------------------------------

@triton.jit
def _ce_bwd_kernel(
    LOGITS_ptr, TARGETS_ptr, LOGSUMEXP_ptr,
    DLOGITS_ptr,
    grad_output,    # 来自上游的标量梯度（通常是 1/N_valid）
    stride_row,
    V,
    ignore_index,
    BLOCK_SIZE: tl.constexpr,
):
    """每个 program 处理一行的 dlogits。

    dlogits[row, j] = grad_output * (softmax(logits[row])[j] - 1{j == target[row]})
    ignore_index 行：dlogits[row, :] = 0

    实现框架：

        row = tl.program_id(0)
        # 载入 target / logsumexp，判断 is_ignored

        for block_start in range(0, V, BLOCK_SIZE):
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < V
            # tl.load 本块 logits（fp32）
            # TODO: 用保存的 logsumexp 重算本块 softmax（见文件头引导）
            # TODO: 按文件头的 bwd 语义写出 dlogit
            #       （one-hot 减法用 offsets == target 构造，配 tl.where）
            #       ignore_index 行整行为 0
            # tl.store 写出本块（签名：tl.store(ptr, value, mask=...)）
    """
    # TODO: 按上述框架实现
    raise NotImplementedError("_ce_bwd_kernel — 学生实现")


# ---------------------------------------------------------------------------
# autograd.Function 包装（脚手架，已完整；不需要修改）
# ---------------------------------------------------------------------------

class FusedCrossEntropyFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, logits: Tensor, targets: Tensor, ignore_index: int) -> Tensor:
        assert logits.ndim == 2, "logits 必须是 (N, V)"
        assert targets.ndim == 1, "targets 必须是 (N,)"
        N, V = logits.shape

        logits_c = logits.contiguous()
        targets_c = targets.contiguous()

        loss_per_row = torch.empty(N, dtype=torch.float32, device=logits.device)
        logsumexp = torch.empty(N, dtype=torch.float32, device=logits.device)

        grid = (N,)
        _ce_fwd_kernel[grid](
            logits_c, targets_c,
            loss_per_row, logsumexp,
            logits_c.stride(0),   # stride_row = V
            V, ignore_index,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        # mean over valid（非 ignore_index）的行
        valid_mask = targets_c != ignore_index
        N_valid = valid_mask.sum().clamp(min=1)
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
        # grad_output 是标量（来自 loss.backward()），需要除以 N_valid 得到每行权重
        grad_scale = grad_output.item() / ctx.N_valid.item()

        grid = (N,)
        _ce_bwd_kernel[grid](
            logits, targets, logsumexp,
            dlogits,
            grad_scale,
            logits.stride(0),
            V,
            ctx.ignore_index,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        return dlogits, None, None   # targets/ignore_index 无梯度


# ---------------------------------------------------------------------------
# 对外接口（SPEC 冻结签名）
# ---------------------------------------------------------------------------

def fused_cross_entropy(
    logits: Tensor,
    targets: Tensor,
    ignore_index: int = -100,
) -> Tensor:
    """Fused cross-entropy loss（online softmax，不物化 probs）。

    参数：
        logits       — (N, V) 未归一化 log-probabilities
        targets      — (N,) int64 目标 token id
        ignore_index — 该值对应的行不参与 loss 计算（默认 -100）

    返回：标量 mean loss（float32）。
    显存优势：不存 (N, V) softmax probs，只存 logsumexp (N,) 向量。
    """
    return FusedCrossEntropyFunction.apply(logits, targets, ignore_index)
