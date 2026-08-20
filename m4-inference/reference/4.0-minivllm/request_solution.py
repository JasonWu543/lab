"""request_solution.py — Request / Sequence 数据结构（参考实现）"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class SeqStatus(Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


@dataclass
class Request:
    req_id: int
    prompt_ids: list[int]
    max_new_tokens: int
    temperature: float = 0.0   # 0 = greedy
    top_p: float = 1.0
    arrival_step: int = 0      # 到达时刻（用引擎 step 数模拟时钟）


class Sequence:
    """一个请求的运行态。"""

    def __init__(self, request: Request) -> None:
        self.request = request
        self.status = SeqStatus.WAITING
        self.output_ids: list[int] = []
        self.block_table: list[int] = []          # 已分配的物理 block id 列表
        self.num_cached_tokens: int = 0           # prefix cache 命中的 token 数

        # metrics
        self._first_token_step: int | None = None  # 首 token 产出的 step 编号
        self._finish_step: int | None = None

    # ── 便利属性 ──────────────────────────────────────────────────────────

    def num_tokens(self) -> int:
        """prompt + 已生成 output 的总 token 数。"""
        return len(self.request.prompt_ids) + len(self.output_ids)

    def last_token(self) -> int:
        """当前序列末尾的 token id（用于 decode 阶段单步输入）。"""
        if self.output_ids:
            return self.output_ids[-1]
        return self.request.prompt_ids[-1]

    def is_finished(self) -> bool:
        return self.status == SeqStatus.FINISHED
