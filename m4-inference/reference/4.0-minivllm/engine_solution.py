"""engine_solution.py — 推理引擎主体（参考实现）

设计决策：
- 混合 prefill+decode 时分两次前向（先 prefill，后 decode）——简化 attention mask 处理
- paged KV 存储 shape: [num_layers, 2, num_blocks, block_size, n_kv_heads, head_dim]
- cache_adapter 负责 gather（分页→DynamicCache）和 scatter（DynamicCache→分页）
- 正确性锚点：与手写串行 greedy 循环完全一致（串行循环用 oracle_generate 验证）
"""
from __future__ import annotations

import torch
from torch import Tensor

from .request_solution import Request, Sequence, SeqStatus
from .block_manager_solution import BlockManager
from .scheduler_solution import Scheduler
from .sampler_solution import sample
from .cache_adapter_solution import (
    gather_past_kv,
    scatter_new_kv,
    _cached_kv_len,
)


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

        # 从 model config 读取 KV 维度
        cfg = model.config
        self.num_layers: int = cfg.num_hidden_layers
        self.n_kv_heads: int = cfg.num_key_value_heads
        self.head_dim: int = cfg.hidden_size // cfg.num_attention_heads

        self.block_size = block_size
        self.num_blocks = num_blocks

        # 分页 KV 存储：[num_layers, 2, num_blocks, block_size, n_kv_heads, head_dim]
        self.paged_kv = torch.zeros(
            self.num_layers, 2, num_blocks, block_size, self.n_kv_heads, self.head_dim,
            dtype=torch.float32,
        )

        self.block_manager = BlockManager(num_blocks, block_size)
        self.scheduler = Scheduler(self.block_manager, max_batch_tokens)

        self._step_count: int = 0
        # metrics 存储：req_id → dict
        self._metrics: dict[int, dict] = {}
        # 已完成的序列缓存（用于 metrics 查询）
        self._finished_seqs: list[Sequence] = []

    # ── 公开接口 ──────────────────────────────────────────────────────────

    def add_request(self, req: Request) -> None:
        seq = Sequence(req)
        self.scheduler.add(seq)

    def step(self) -> list[Sequence]:
        """一个 iteration：schedule → prefill → decode → 采样 → 更新 → 返回完成的 seq。"""
        prefill_seqs, decode_seqs = self.scheduler.schedule()

        if not prefill_seqs and not decode_seqs:
            return []

        finished: list[Sequence] = []

        # ── prefill 前向 ──────────────────────────────────────────────────
        if prefill_seqs:
            prefill_tokens = self._run_prefill(prefill_seqs)
            for seq, tok in zip(prefill_seqs, prefill_tokens):
                seq.output_ids.append(tok)
                self.block_manager.append_slot(seq)
                seq._first_token_step = self._step_count
                if seq.output_ids[-1] == self._eos_token_id() or \
                        len(seq.output_ids) >= seq.request.max_new_tokens:
                    self.scheduler.finish(seq)
                    seq._finish_step = self._step_count
                    finished.append(seq)

        # ── decode 前向 ──────────────────────────────────────────────────
        # 过滤掉刚 prefill 完并 finish 的（若某 seq prefill 时恰好结束）
        active_decode = [s for s in decode_seqs if not s.is_finished()]
        if active_decode:
            decode_tokens = self._run_decode(active_decode)
            for seq, tok in zip(active_decode, decode_tokens):
                seq.output_ids.append(tok)
                self.block_manager.append_slot(seq)
                if tok == self._eos_token_id() or \
                        len(seq.output_ids) >= seq.request.max_new_tokens:
                    self.scheduler.finish(seq)
                    seq._finish_step = self._step_count
                    finished.append(seq)

        self._step_count += 1

        for seq in finished:
            self._finished_seqs.append(seq)

        return finished

    def run(self) -> dict[int, list[int]]:
        """跑到全部完成，返回 {req_id: output_ids}。"""
        results: dict[int, list[int]] = {}
        while self.scheduler.num_waiting > 0 or self.scheduler.num_running > 0:
            finished = self.step()
            for seq in finished:
                results[seq.request.req_id] = list(seq.output_ids)
        return results

    def metrics(self) -> dict[int, dict]:
        """{req_id: {"ttft_steps": int, "tpot_steps": float, "n_out": int}}"""
        out = {}
        for seq in self._finished_seqs:
            rid = seq.request.req_id
            arr = seq.request.arrival_step
            ttft = (seq._first_token_step - arr) if seq._first_token_step is not None else -1
            n_out = len(seq.output_ids)
            if n_out > 1 and seq._finish_step is not None and seq._first_token_step is not None:
                tpot = (seq._finish_step - seq._first_token_step) / (n_out - 1)
            else:
                tpot = 0.0
            out[rid] = {"ttft_steps": ttft, "tpot_steps": tpot, "n_out": n_out}
        return out

    # ── 内部：prefill 前向 ────────────────────────────────────────────────

    def _run_prefill(self, seqs: list[Sequence]) -> list[int]:
        """
        对每个 prefill seq 独立前向（逐条，简化 padding 逻辑）。
        返回各 seq 的第一个 output token。
        """
        tokens = []
        for seq in seqs:
            tok = self._prefill_one(seq)
            tokens.append(tok)
        return tokens

    def _prefill_one(self, seq: Sequence) -> int:
        """单条 prefill：带 prefix cache 的前向。"""
        prompt_ids = seq.request.prompt_ids
        cached_len = seq.num_cached_tokens  # prefix cache 命中数

        input_ids = torch.tensor(
            [prompt_ids[cached_len:]], dtype=torch.long
        )  # [1, effective_prompt_len]

        past_kv = None
        if cached_len > 0:
            # gather prefix cache KV
            past_kv = gather_past_kv(self.paged_kv, [seq])

        with torch.no_grad():
            out = self.model(
                input_ids,
                past_key_values=past_kv,
                use_cache=True,
            )

        # scatter 新 KV 回 paged_kv
        scatter_new_kv(
            self.paged_kv,
            out.past_key_values,
            [seq],
            new_token_positions=[cached_len],
        )

        # 取最后一步的 logit 采样
        last_logit = out.logits[0, -1:, :]  # [1, vocab]
        tok = sample(last_logit, [seq], generator=self.generator)[0]
        return tok

    # ── 内部：decode 前向 ─────────────────────────────────────────────────

    def _run_decode(self, seqs: list[Sequence]) -> list[int]:
        """
        批量 decode 前向（每条 seq 输入 1 token）。
        为简化 attention mask，逐条前向（batching 留给学生扩展）。
        """
        tokens = []
        for seq in seqs:
            tok = self._decode_one(seq)
            tokens.append(tok)
        return tokens

    def _decode_one(self, seq: Sequence) -> int:
        """单步 decode：gather KV → 前向 1 token → scatter KV → sample。"""
        cached_len = seq.num_tokens() - 1  # 上一步结束后的 KV 长度

        input_ids = torch.tensor([[seq.last_token()]], dtype=torch.long)  # [1, 1]

        past_kv = gather_past_kv(self.paged_kv, [seq])

        with torch.no_grad():
            out = self.model(
                input_ids,
                past_key_values=past_kv,
                use_cache=True,
            )

        scatter_new_kv(
            self.paged_kv,
            out.past_key_values,
            [seq],
            new_token_positions=[cached_len],
        )

        last_logit = out.logits[0, -1:, :]  # [1, vocab]
        tok = sample(last_logit, [seq], generator=self.generator)[0]
        return tok

    # ── 辅助 ──────────────────────────────────────────────────────────────

    def _eos_token_id(self) -> int:
        eos = getattr(self.model.config, "eos_token_id", None)
        if isinstance(eos, list):
            return eos[0]
        return eos if eos is not None else -1
