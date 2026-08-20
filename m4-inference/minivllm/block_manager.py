"""block_manager.py — 分页 KV block 管理 + prefix cache（Phase 4.0，学生实现文件）

U1 任务（先完成这里，再去 scheduler.py）：
  实现 BlockManager 的全部方法，让 T1 全绿。

U4 任务（U2/U3 完成后回来）：
  完善 prefix cache：match_prefix / allocate 的命中路径 / free 的缓存保留逻辑，
  让 T5 全绿。

────────────────────────────────────────────────────────────────────────────────
接口约定（不得改变方法签名）：

  num_free_blocks  : int property
                     可用 block 总数 = 纯空闲块 + 可驱逐的 cached 块
  can_allocate(seq): 检查 prefill 准入（prefix 命中后实际需要的新块数 ≤ 空闲数）。
                     若 prefill 后还要 decode，需为首个输出 token 的 KV slot
                     预留容量；can_allocate 与 allocate 必须使用相同计数
  allocate(seq)    : 分配 block_table；命中前缀的块直接复用（refcount+1）
  can_append(seq)  : decode 准入（跨块边界时需 1 个空闲块）
  append_slot(seq) : decode 后更新 block_table；跨块边界时分配新块
  free(seq)        : refcount-1；归零时：整块满 → cached，未满块 → 彻底回收
  match_prefix(ids): 返回 (可复用 block ids, 命中 token 数（block_size 整数倍）)

────────────────────────────────────────────────────────────────────────────────
数据结构提示：

  self._refcount : list[int]，每个 block 的引用计数
  self._free_list: list[int]，完全空闲（refcount==0，不在 prefix table）的 block id
  self._prefix_table : dict[hash_key → block_id]，prefix cache 索引
                       建议用 OrderedDict 以支持 LRU 近似驱逐

  块粒度 hash：key = hash(tuple(token_ids[: (block_idx+1)*block_size]))
               即「该块及之前所有 token 的元组」的 Python hash。

────────────────────────────────────────────────────────────────────────────────
引导问题（动手前想清楚，不要直接看答案）：

  Q1. 一个 block 能被 free（真正回收）的充要条件是什么？
      "整块满"与"未满块"的处理为何不同？

  Q2. prefix cache 命中时 refcount 要加几？为什么不能直接跳过？

  Q3. 当 free_list 为空时，如何腾出新块？驱逐 cached block 的正确顺序是什么？

  Q4. match_prefix 中，如果某个 hash 在 prefix_table 里但 refcount==0，
      能算作命中吗？（提示：cached 块内容仍有效）

────────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

from collections import OrderedDict

from .request import Sequence


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int) -> None:
        self.num_blocks = num_blocks
        self.block_size = block_size

        # ── 你需要初始化的数据结构 ────────────────────────────────────────
        self._refcount: list[int] = [0] * num_blocks
        self._block_prefix_hash: list[int | None] = [None] * num_blocks
        self._prefix_table: OrderedDict[int, int] = OrderedDict()
        self._free_list: list[int] = list(range(num_blocks))

    # ── 基础统计 ──────────────────────────────────────────────────────────

    @property
    def num_free_blocks(self) -> int:
        # TODO（U1）：返回可用块数 = 纯空闲块 + refcount==0 的 cached 块
        raise NotImplementedError

    # ── 内部辅助（建议先实现这里，再写公开方法）───────────────────────────

    def _alloc_block(self) -> int:
        """取一个可用 block（refcount 置 1）。
        优先 free_list；耗尽时驱逐最旧的 cached block。
        """
        # TODO（U1）
        raise NotImplementedError

    def _evict_cached(self) -> int:
        """从 prefix_table 中驱逐 refcount==0 的最旧 block，返回其 id。"""
        # TODO（U4）：遍历 _prefix_table（有序字典）找到第一个 refcount==0 的条目驱逐
        raise NotImplementedError

    def _release_block(self, bid: int, token_ids: list[int], block_idx: int) -> None:
        """refcount - 1；归零时：整块满 → 保留 cached，未满块 → 回收为 free。"""
        # TODO（U1 实现基础版不含 cached 逻辑；U4 完善 cached 保留）
        raise NotImplementedError

    @staticmethod
    def _prefix_hash(token_ids: list[int], block_idx: int, block_size: int) -> int:
        """该块及之前所有 token 组成前缀的 hash。已给出，无需实现。"""
        end = (block_idx + 1) * block_size
        return hash(tuple(token_ids[:end]))

    # ── 准入检查 ──────────────────────────────────────────────────────────

    def can_allocate(self, seq: Sequence) -> bool:
        """prefill 准入：考虑 prefix 命中后实际需要的新块数 ≤ num_free_blocks。"""
        # TODO（U1）
        raise NotImplementedError

    def can_append(self, seq: Sequence) -> bool:
        """decode 准入：跨块边界时需 1 个空闲块。"""
        # TODO（U1）
        raise NotImplementedError

    # ── 分配与释放 ────────────────────────────────────────────────────────

    def allocate(self, seq: Sequence) -> None:
        """为 prefill 分配 block_table；命中前缀的块直接复用（refcount+1）。
        同时设置 seq.num_cached_tokens。
        """
        # TODO（U1 先实现无 prefix cache 版本；U4 加入命中逻辑）
        raise NotImplementedError

    def append_slot(self, seq: Sequence) -> None:
        """decode 后更新 block_table；跨块边界时分配新块。
        注意：调用时 seq.output_ids 已经 append 了新 token，
        所以 seq.num_tokens() 反映的是最新状态。
        """
        # TODO（U1）
        raise NotImplementedError

    def free(self, seq: Sequence) -> None:
        """refcount-1 并回收；同时更新 prefix cache（整块满的块保留为 cached）。"""
        # TODO（U1 先实现彻底回收；U4 加入 cached 保留逻辑）
        raise NotImplementedError

    # ── prefix cache 查询 ─────────────────────────────────────────────────

    def match_prefix(self, token_ids: list[int]) -> tuple[list[int], int]:
        """逐块从头匹配，遇到缺失立即停止。
        返回 (matched block ids, 命中 token 数)。
        """
        # TODO（U4）：U1 阶段可先返回 ([], 0) 让其他测试能跑
        return [], 0
