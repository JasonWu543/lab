"""block_manager_solution.py — 分页 KV block 管理 + prefix cache（参考实现）

设计：
  - 物理 block 状态：
      "cached"  : refcount == 0，但在 prefix_table 里，内容可复用；
                  按 LRU 淘汰（本实现简化为队列末尾淘汰，无严格 LRU）。
      "free"    : refcount == 0，不在 prefix_table，可直接分配。
      "active"  : refcount >= 1。
  - 分配优先从"free"取；free 耗尽时从"cached"驱逐最旧的。
  - prefix cache 命中：matched blocks 的 refcount += 1（激活）。
  - free 时：全块满 → 保留在 prefix_table（状态变为 cached）；
             未满块 → 彻底回收（状态变为 free）。
"""
from __future__ import annotations

from collections import OrderedDict

from .request_solution import Sequence


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int) -> None:
        self.num_blocks = num_blocks
        self.block_size = block_size

        self._refcount: list[int] = [0] * num_blocks
        # block_id → prefix hash（None 表示不在缓存表）
        self._block_prefix_hash: list[int | None] = [None] * num_blocks

        # prefix cache 表：hash → block_id（refcount 可以为 0，但内容有效）
        # 用 OrderedDict 模拟简单 LRU（insertions order = 最老在前）
        self._prefix_table: OrderedDict[int, int] = OrderedDict()

        # 纯 free 池（没有 prefix cache 价值的块）
        self._free_list: list[int] = list(range(num_blocks))

    # ── 基础统计 ──────────────────────────────────────────────────────────

    @property
    def num_free_blocks(self) -> int:
        """可立即用的 free 块 + 可驱逐的 cached 块之和。"""
        cached_count = sum(
            1 for bid in self._prefix_table.values() if self._refcount[bid] == 0
        )
        return len(self._free_list) + cached_count

    # ── 内部辅助 ──────────────────────────────────────────────────────────

    def _alloc_block(self) -> int:
        """取一个可用 block（refcount 置 1）。
        优先从 free_list 取；耗尽时驱逐最旧的 cached block（LRU 近似）。
        """
        if self._free_list:
            bid = self._free_list.pop()
        else:
            # 驱逐最旧的 cached block（prefix_table 里 refcount==0 的）
            bid = self._evict_cached()

        self._refcount[bid] = 1
        return bid

    def _evict_cached(self) -> int:
        """从 cached 块（prefix_table 中 refcount==0）驱逐最旧的一个。"""
        for h, bid in self._prefix_table.items():
            if self._refcount[bid] == 0:
                del self._prefix_table[h]
                self._block_prefix_hash[bid] = None
                return bid
        raise RuntimeError("No free or cached blocks available")

    def _release_block(self, bid: int, token_ids: list[int], block_idx: int) -> None:
        """refcount - 1；归零时：若整块满则保留 cached，否则回收为 free。"""
        self._refcount[bid] -= 1
        if self._refcount[bid] > 0:
            return
        # refcount 归零
        end = (block_idx + 1) * self.block_size
        if end <= len(token_ids):
            # 整块满：登记/更新 prefix cache，状态 = cached
            h = self._prefix_hash(token_ids, block_idx, self.block_size)
            old_bid = self._prefix_table.get(h)
            if old_bid is not None and old_bid != bid:
                # 移除旧条目（旧 block 的 hash 被覆盖）
                self._block_prefix_hash[old_bid] = None
                if self._refcount[old_bid] == 0:
                    self._free_list.append(old_bid)
            self._prefix_table[h] = bid
            self._prefix_table.move_to_end(h)  # 标记为最近使用
            self._block_prefix_hash[bid] = h
            # 不回收到 free_list，保留内容（cached 状态）
        else:
            # 未满块：彻底回收
            old_h = self._block_prefix_hash[bid]
            if old_h is not None and self._prefix_table.get(old_h) == bid:
                del self._prefix_table[old_h]
            self._block_prefix_hash[bid] = None
            self._free_list.append(bid)

    @staticmethod
    def _prefix_hash(token_ids: list[int], block_idx: int, block_size: int) -> int:
        end = (block_idx + 1) * block_size
        return hash(tuple(token_ids[:end]))

    # ── 准入检查 ──────────────────────────────────────────────────────────

    def can_allocate(self, seq: Sequence) -> bool:
        _, cached_tokens = self.match_prefix(seq.request.prompt_ids)
        total_tokens = len(seq.request.prompt_ids)
        total_blocks = (total_tokens + self.block_size - 1) // self.block_size
        cached_blocks = cached_tokens // self.block_size
        new_blocks_needed = total_blocks - cached_blocks
        return new_blocks_needed <= self.num_free_blocks

    def can_append(self, seq: Sequence) -> bool:
        n = seq.num_tokens()
        if n % self.block_size == 0:
            return self.num_free_blocks >= 1
        return True

    # ── 分配与释放 ────────────────────────────────────────────────────────

    def allocate(self, seq: Sequence) -> None:
        """prefill 分配：prefix cache 命中的块 refcount+1，其余新分配。"""
        token_ids = seq.request.prompt_ids
        total_tokens = len(token_ids)
        total_blocks = (total_tokens + self.block_size - 1) // self.block_size

        cached_blocks, cached_tokens = self.match_prefix(token_ids)
        seq.num_cached_tokens = cached_tokens

        block_table: list[int] = []
        for bid in cached_blocks:
            self._refcount[bid] += 1
            block_table.append(bid)

        for _ in range(len(cached_blocks), total_blocks):
            bid = self._alloc_block()
            block_table.append(bid)

        seq.block_table = block_table

    def append_slot(self, seq: Sequence) -> None:
        """decode 后更新 block_table；跨块边界时分配新块。"""
        n = seq.num_tokens()
        block_idx = (n - 1) // self.block_size
        if block_idx >= len(seq.block_table):
            bid = self._alloc_block()
            seq.block_table.append(bid)

    def free(self, seq: Sequence) -> None:
        """refcount-1；满块保留 cached，未满块回收为 free。"""
        token_ids = seq.request.prompt_ids + seq.output_ids
        for i, bid in enumerate(seq.block_table):
            self._release_block(bid, token_ids, i)
        seq.block_table = []

    # ── prefix cache 查询 ─────────────────────────────────────────────────

    def match_prefix(self, token_ids: list[int]) -> tuple[list[int], int]:
        """逐块匹配；返回 (matched block ids, 命中 token 数)。"""
        matched: list[int] = []
        num_full_blocks = len(token_ids) // self.block_size

        for i in range(num_full_blocks):
            h = self._prefix_hash(token_ids, i, self.block_size)
            bid = self._prefix_table.get(h)
            if bid is None:
                break
            matched.append(bid)

        cached_tokens = len(matched) * self.block_size
        return matched, cached_tokens
