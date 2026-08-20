"""scheduler.py — FCFS 调度器 + continuous batching 准入（Phase 4.0，学生实现文件）

U2 任务（U1 完成后来这里）：
  实现 Scheduler.schedule()，让 T2/T3 全绿。

────────────────────────────────────────────────────────────────────────────────
接口约定：

  add(seq)    : 把 seq 放入 waiting 队列（FCFS，保持到达顺序）
  schedule()  : 返回 (prefill_seqs, decode_seqs)
                规则：
                  1. 先把所有 RUNNING 的都塞进 decode（can_append 失败时 raise，不做抢占）
                  2. 剩余 token 预算内，按 FCFS 顺序逐一尝试准入 WAITING 的 seq（prefill）
                     准入条件：can_allocate(seq) AND effective_prompt_len ≤ remaining_budget
                     effective_prompt_len = prompt_len - prefix_cache_命中数（至少为 1）
                     遇到第一个不满足的立即停止（不跳过——FCFS）
  finish(seq) : 释放 block，seq 移出 running

────────────────────────────────────────────────────────────────────────────────
引导问题：

  Q1. decode 阶段每条 seq 消耗几个 token 的 KV 计算预算？

  Q2. "不跳过"的 FCFS 语义：若队首 seq 块不足，后面的 seq 能准入吗？
      为什么真 vLLM 需要更复杂的调度策略？

  Q3. 准入时 effective_prompt_len 为什么要 max(..., 1)？
      （提示：prefix 命中整个 prompt 时，模型前向仍需要至少 1 个 token 的输入）

────────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

from collections import deque

from .request import Sequence, SeqStatus
from .block_manager import BlockManager


class Scheduler:
    def __init__(self, block_manager: BlockManager, max_batch_tokens: int) -> None:
        self.block_manager = block_manager
        self.max_batch_tokens = max_batch_tokens

        self._waiting: deque[Sequence] = deque()
        self._running: list[Sequence] = []

    # ── 公开接口 ──────────────────────────────────────────────────────────

    def add(self, seq: Sequence) -> None:
        """把 seq 放入 waiting 队列（FCFS）。已给出，无需修改。"""
        self._waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], list[Sequence]]:
        """返回 (prefill 序列列表, decode 序列列表)。"""
        # TODO（U2）：
        #   步骤 1：把所有 _running 的 seq 放进 decode_seqs；
        #           若 can_append 失败，raise RuntimeError（不做抢占）
        #   步骤 2：在剩余预算内按 FCFS 准入 _waiting 的 seq；
        #           准入后调用 block_manager.allocate，修改 seq.status = RUNNING
        #   步骤 3：更新 _waiting / _running 列表
        #   步骤 4：返回 (prefill_seqs, decode_seqs)
        raise NotImplementedError

    def finish(self, seq: Sequence) -> None:
        """把完成的 seq 从 running 移出，并释放 block。"""
        # TODO（U2）
        raise NotImplementedError

    # ── 调试用属性（测试会用到）──────────────────────────────────────────

    @property
    def num_waiting(self) -> int:
        return len(self._waiting)

    @property
    def num_running(self) -> int:
        return len(self._running)
