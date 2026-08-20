"""engine.py — 推理引擎主体（Phase 4.0，学生实现文件）

U3 任务（U1+U2 完成后来这里）：
  实现 Engine.step() 的核心逻辑，让 T2/T3/T4 全绿。

U5 任务（U3 完成后）：
  实现 metrics()，让 T7 全绿。

────────────────────────────────────────────────────────────────────────────────
step() 的阶段划分（必须按顺序）：

  1. schedule  →  (prefill_seqs, decode_seqs) = self.scheduler.schedule()
  2. forward   →  对 prefill_seqs 和 decode_seqs 各跑模型前向
                  允许分两次前向（先 prefill，后 decode），但必须在同一 step 内
  3. sample    →  用 sampler.sample() 得到各 seq 的 next token
  4. update    →  append token 到 output_ids，调用 append_slot，
                  检查终止条件（max_new_tokens 或 EOS），
                  调用 scheduler.finish() 完成的 seq

────────────────────────────────────────────────────────────────────────────────
分页 KV 存储 shape（已由 __init__ 创建，无需修改）：
  self.paged_kv: [num_layers, 2, num_blocks, block_size, n_kv_heads, head_dim]

模型前向调用方式（DynamicCache 示例）：
  past_kv = gather_past_kv(self.paged_kv, [seq])          # gather 历史 KV
  out = model(input_ids, past_key_values=past_kv, use_cache=True)
  scatter_new_kv(self.paged_kv, out.past_key_values, [seq], [cached_len])  # 写回

────────────────────────────────────────────────────────────────────────────────
metrics() 字段说明：
  ttft_steps : 首 token 产出的 step 编号 - arrival_step
               （用 step 计数作时钟，不是真实秒数）
  tpot_steps : (finish_step - first_token_step) / (n_out - 1)
               （每个 decode token 平均耗费的步数）
  n_out      : 生成的 token 数（不含 prompt）

  提示：seq._first_token_step / seq._finish_step 由 step() 在适当时机写入。

────────────────────────────────────────────────────────────────────────────────
引导问题：

  Q1. prefill 时，已命中 prefix cache 的 token 不再计算 —— input_ids 应从哪里开始截？
      past_key_values 里已有的 KV 长度（cached_len）怎么确定？

  Q2. decode 时，input_ids 只有 1 个 token（last_token），
      past_key_values 里应有多少 token 的历史 KV？

  Q3. scatter_new_kv 的 new_token_positions 参数表示什么？
      prefill 和 decode 场景下分别传什么值？

  Q4. 终止条件有哪两种？两者都要处理。

────────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import torch

from .request import Request, Sequence, SeqStatus
from .block_manager import BlockManager
from .scheduler import Scheduler
from .sampler import sample
from .cache_adapter import gather_past_kv, scatter_new_kv, _cached_kv_len


class Engine:
    def __init__(
        self,
        model,
        num_blocks: int = 256,
        block_size: int = 16,
        max_batch_tokens: int = 2048,
        generator: torch.Generator | None = None,
    ) -> None:
        self.model = model
        self.generator = generator

        # 从 model.config 读取 KV 维度（无需修改）
        cfg = model.config
        self.num_layers: int = cfg.num_hidden_layers
        self.n_kv_heads: int = cfg.num_key_value_heads
        self.head_dim: int = cfg.hidden_size // cfg.num_attention_heads
        self.block_size = block_size

        # 分页 KV 存储（无需修改）
        self.paged_kv = torch.zeros(
            self.num_layers, 2, num_blocks, block_size,
            self.n_kv_heads, self.head_dim,
            dtype=torch.float32,
        )

        self.block_manager = BlockManager(num_blocks, block_size)
        self.scheduler = Scheduler(self.block_manager, max_batch_tokens)

        self._step_count: int = 0
        self._finished_seqs: list[Sequence] = []

    # ── 公开接口 ──────────────────────────────────────────────────────────

    def add_request(self, req: Request) -> None:
        """把 Request 包装成 Sequence 放入调度器。已给出，无需修改。"""
        seq = Sequence(req)
        self.scheduler.add(seq)

    def step(self) -> list[Sequence]:
        """一个 iteration：schedule → forward → sample → update → 返回完成的 seq。"""
        # TODO（U3）：参考上方阶段划分，逐步实现
        # 阶段 1：调度
        # 阶段 2a：prefill 前向（逐 seq，调用 _prefill_one）
        # 阶段 2b：decode 前向（逐 seq，调用 _decode_one）
        # 阶段 3&4：采样 + 更新（已封装在 _prefill_one / _decode_one 中）
        # 最后：self._step_count += 1；返回本步 finished
        raise NotImplementedError

    def run(self) -> dict[int, list[int]]:
        """跑到全部完成，返回 {req_id: output_ids}。已给出框架，无需修改。"""
        results: dict[int, list[int]] = {}
        while self.scheduler.num_waiting > 0 or self.scheduler.num_running > 0:
            finished = self.step()
            for seq in finished:
                results[seq.request.req_id] = list(seq.output_ids)
        return results

    def metrics(self) -> dict[int, dict]:
        """{req_id: {"ttft_steps": int, "tpot_steps": float, "n_out": int}}"""
        # TODO（U5）
        raise NotImplementedError

    # ── 内部：prefill 前向 ────────────────────────────────────────────────

    def _prefill_one(self, seq: Sequence) -> int:
        """单条 prefill（含 prefix cache gather）。返回第一个 output token id。"""
        # TODO（U3）：
        #   1. 确定 cached_len = seq.num_cached_tokens
        #   2. input_ids = prompt_ids[cached_len:]（形状 [1, effective_len]）
        #   3. past_kv = gather_past_kv(self.paged_kv, [seq])（若 cached_len>0）
        #   4. model 前向
        #   5. scatter_new_kv（new_token_positions=[cached_len]）
        #   6. 取 out.logits[0, -1, :] 采样
        raise NotImplementedError

    # ── 内部：decode 前向 ─────────────────────────────────────────────────

    def _decode_one(self, seq: Sequence) -> int:
        """单步 decode。返回 next token id。"""
        # TODO（U3）：
        #   1. cached_len = seq.num_tokens() - 1（上一步结束后 paged_kv 里已有的 token 数）
        #   2. input_ids = [[seq.last_token()]]
        #   3. past_kv = gather_past_kv(self.paged_kv, [seq])
        #   4. model 前向
        #   5. scatter_new_kv（new_token_positions=[cached_len]）
        #   6. 取 out.logits[0, -1, :] 采样
        raise NotImplementedError

    # ── 辅助 ──────────────────────────────────────────────────────────────

    def _eos_token_id(self) -> int:
        """读取 model.config.eos_token_id（已给出，无需修改）。"""
        eos = getattr(self.model.config, "eos_token_id", None)
        if isinstance(eos, list):
            return eos[0]
        return eos if eos is not None else -1
