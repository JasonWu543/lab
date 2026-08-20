"""request.py — Request / Sequence 数据结构（Phase 4.0，学生实现文件）

闯关顺序：
  U1  BlockManager（block_manager.py）
  U2  Scheduler   （scheduler.py）
  U3  Engine.step  主循环（engine.py）
  U4  prefix cache（block_manager.py 的 match_prefix / allocate / free 完整版）
  U5  metrics      （engine.py metrics()）

运行测试：
  cd m4-inference && python3 -m pytest tests/test_minivllm.py -x -q

本文件全部由 Agent 给出（纯结构，无核心算法），无需修改。
卡住 30 分钟以上再看：reference/4.0-minivllm/*_solution.py

── M7 并发开发玩法 ─────────────────────────────────────────────────────────────
接口已冻结（见 docs/4.0-minivllm/SPEC.md §3）。你可以派多个 Agent 分工实现：
  Agent-A → block_manager.py + prefix cache
  Agent-B → scheduler.py + engine.py 主循环
  Agent-C → sampler.py + metrics
然后自己做 review + 集成测试。度量记录写进 POSTMORTEM §7.1。
────────────────────────────────────────────────────────────────────────────────
"""
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
    """一个请求的运行态。

    属性：
      request          : 原始 Request
      status           : SeqStatus
      output_ids       : list[int]，已生成的 token（不含 prompt）
      block_table      : list[int]，已分配的物理 block id（由 BlockManager 维护）
      num_cached_tokens: int，prefix cache 命中的 token 数（allocate 时写入）

    方法：
      num_tokens()  → int   prompt + output 的总 token 数
      last_token()  → int   序列末尾 token id（decode 阶段单步输入）
      is_finished() → bool
    """

    def __init__(self, request: Request) -> None:
        self.request = request
        self.status = SeqStatus.WAITING
        self.output_ids: list[int] = []
        self.block_table: list[int] = []
        self.num_cached_tokens: int = 0

        # metrics 内部字段（Engine 负责填写）
        self._first_token_step: int | None = None
        self._finish_step: int | None = None

    def num_tokens(self) -> int:
        """prompt + output 的总 token 数。"""
        return len(self.request.prompt_ids) + len(self.output_ids)

    def last_token(self) -> int:
        """序列末尾 token id（有 output 则取最后一个 output，否则取 prompt 最后一个）。"""
        if self.output_ids:
            return self.output_ids[-1]
        return self.request.prompt_ids[-1]

    def is_finished(self) -> bool:
        return self.status == SeqStatus.FINISHED
