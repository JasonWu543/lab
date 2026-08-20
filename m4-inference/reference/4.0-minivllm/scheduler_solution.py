"""scheduler_solution.py — FCFS 调度器 + continuous batching 准入（参考实现）"""
from __future__ import annotations

from collections import deque

from .request_solution import Sequence, SeqStatus
from .block_manager_solution import BlockManager


class Scheduler:
    """
    FCFS 连续批调度器。

    schedule() 返回 (prefill_seqs, decode_seqs)：
    1. 先把所有 RUNNING 的 seq 都塞进 decode（若 can_append 失败直接报错，不做抢占）。
    2. 在剩余 token 预算内，按 FCFS 顺序依次尝试把 WAITING 的 seq prefill 准入。
       准入条件：can_allocate + 该 seq 的 prompt 不超过剩余预算。
    """

    def __init__(self, block_manager: BlockManager, max_batch_tokens: int) -> None:
        self.block_manager = block_manager
        self.max_batch_tokens = max_batch_tokens

        self._waiting: deque[Sequence] = deque()   # FCFS 队列
        self._running: list[Sequence] = []         # 当前正在 decode 的序列

    # ── 公开接口 ──────────────────────────────────────────────────────────

    def add(self, seq: Sequence) -> None:
        """把新 seq 放入 waiting 队列（FCFS）。"""
        self._waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], list[Sequence]]:
        """返回 (prefill 序列列表, decode 序列列表)。"""
        decode_seqs: list[Sequence] = []
        prefill_seqs: list[Sequence] = []

        # ── 步骤 1：保证所有 RUNNING 的序列都能 decode ──────────────────
        for seq in self._running:
            if not self.block_manager.can_append(seq):
                raise RuntimeError(
                    f"seq {seq.request.req_id} cannot append but no preemption implemented"
                )
            decode_seqs.append(seq)

        # 已消耗的 token 预算（decode 每条 seq 消耗 1 token 输入）
        used_tokens = len(decode_seqs)

        # ── 步骤 2：在剩余预算内，按 FCFS 准入 WAITING 的 prefill ───────
        admitted: list[Sequence] = []
        for seq in list(self._waiting):
            prompt_len = len(seq.request.prompt_ids)
            remaining = self.max_batch_tokens - used_tokens

            # 考虑 prefix cache：实际需要处理的 token 数
            _, cached_tokens = self.block_manager.match_prefix(seq.request.prompt_ids)
            effective_len = prompt_len - cached_tokens  # 需要真正计算的 token 数
            effective_len = max(effective_len, 1)  # 至少输入 1 个 token

            if effective_len > remaining:
                break  # FCFS：不跳过，直接停止
            if not self.block_manager.can_allocate(seq):
                break  # 块不够，停止（不做抢占）

            self.block_manager.allocate(seq)
            seq.status = SeqStatus.RUNNING
            admitted.append(seq)
            prefill_seqs.append(seq)
            used_tokens += effective_len

        # 把准入的 seq 从 waiting 移出
        for seq in admitted:
            self._waiting.remove(seq)
        self._running.extend(admitted)

        return prefill_seqs, decode_seqs

    def finish(self, seq: Sequence) -> None:
        """把完成的 seq 从 running 移出，并释放 block。"""
        self.block_manager.free(seq)
        seq.status = SeqStatus.FINISHED
        if seq in self._running:
            self._running.remove(seq)

    # ── 调试用 ────────────────────────────────────────────────────────────

    @property
    def num_waiting(self) -> int:
        return len(self._waiting)

    @property
    def num_running(self) -> int:
        return len(self._running)
