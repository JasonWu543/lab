"""Phase 6.0 参考答案 — MinHash + LSH（不得在备课前发给学生）

通用哈希族：h_i(x) = (a_i * x + b_i) mod p，p = 2^61 - 1（梅森素数）。
shingle 哈希：hashlib.blake2b(digest_size=8) → uint64，跨进程稳定。
"""
from __future__ import annotations

import hashlib
import struct
from collections import defaultdict

import numpy as np

# 梅森素数
_P = (1 << 61) - 1
_MAX_UINT64 = np.uint64(0xFFFF_FFFF_FFFF_FFFF)


def _stable_hash(s: str) -> int:
    """blake2b 截断到 8 字节 → uint64（与 Python 进程无关，跨进程稳定）。"""
    h = hashlib.blake2b(s.encode(), digest_size=8).digest()
    return struct.unpack(">Q", h)[0]  # big-endian uint64


def shingle(text: str, k: int = 3) -> set[str]:
    """小写化 + 空白分词后的词级 k-gram 集合（k 个词用空格连接）。"""
    words = text.lower().split()
    if len(words) < k:
        return set()
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


class MinHasher:
    """通用哈希族 MinHash 签名器。

    实现约定（SPEC §3 冻结）：
        h_i(x) = (a_i * x + b_i) mod p   (mod 2^61-1)
    a_i, b_i 由 seed 确定性生成，范围 [1, p-1] / [0, p-1]。
    """

    def __init__(self, num_perm: int = 128, seed: int = 42):
        self.num_perm = num_perm
        rng = np.random.default_rng(seed)
        # a ∈ [1, p-1], b ∈ [0, p-1]；用 uint64 采样后转 Python int 列表避免 numpy overflow
        # 注：numpy 不支持 dtype=object 的 integers()，改用 uint64 采样再转 list[int]
        _a_raw = rng.integers(1, _P, size=num_perm, dtype=np.uint64)
        _b_raw = rng.integers(0, _P, size=num_perm, dtype=np.uint64)
        self._a: list[int] = [int(x) for x in _a_raw]
        self._b: list[int] = [int(x) for x in _b_raw]

    def signature(self, shingles: set[str]) -> np.ndarray:
        """(num_perm,) uint64。空集合返回全 uint64 最大值。"""
        if not shingles:
            return np.full(self.num_perm, _MAX_UINT64, dtype=np.uint64)

        # 所有 shingle 的稳定哈希值（Python int 列表，避免 numpy 截断）
        hashes = [_stable_hash(s) for s in shingles]

        sig: list[int] = [_P] * self.num_perm
        for h in hashes:
            for i in range(self.num_perm):
                v = (self._a[i] * h + self._b[i]) % _P
                if v < sig[i]:
                    sig[i] = v

        return np.array(sig, dtype=np.uint64)


def jaccard_estimate(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    """签名逐位相等比例。"""
    return float(np.mean(sig_a == sig_b))


class LSHIndex:
    """Banded LSH 索引。

    rows = num_perm // bands；同一 band 内所有行相同才触发候选。
    """

    def __init__(self, num_perm: int = 128, bands: int = 32):
        self.num_perm = num_perm
        self.bands = bands
        self.rows = num_perm // bands
        # bucket: band_id -> {band_hash -> [doc_id]}
        self._buckets: list[dict[bytes, list[int]]] = [
            defaultdict(list) for _ in range(bands)
        ]

    def _band_hash(self, band_sig: np.ndarray) -> bytes:
        """将一个 band 的 uint64 向量序列化为字节，作为 bucket key。"""
        return band_sig.tobytes()

    def add(self, doc_id: int, sig: np.ndarray) -> None:
        for b in range(self.bands):
            band_sig = sig[b * self.rows : (b + 1) * self.rows]
            key = self._band_hash(band_sig)
            self._buckets[b][key].append(doc_id)

    def candidates(self, sig: np.ndarray) -> set[int]:
        """至少一个 band 完全相同的已入库 doc_id 集合。"""
        result: set[int] = set()
        for b in range(self.bands):
            band_sig = sig[b * self.rows : (b + 1) * self.rows]
            key = self._band_hash(band_sig)
            if key in self._buckets[b]:
                result.update(self._buckets[b][key])
        return result


def dedup(
    docs: list[str],
    threshold: float = 0.8,
    num_perm: int = 128,
    bands: int = 32,
    k: int = 3,
    seed: int = 42,
) -> tuple[list[int], list[tuple[int, int]]]:
    """近重复去重。

    返回：
        kept_indices  — 保留的 doc 下标（保序，重复簇保留最早的）
        dup_pairs     — 判定为重复的 (早, 晚) 对列表
    """
    hasher = MinHasher(num_perm=num_perm, seed=seed)
    index = LSHIndex(num_perm=num_perm, bands=bands)

    sigs: list[np.ndarray] = []
    nonempty: list[bool] = []
    for doc in docs:
        sh = shingle(doc, k=k)
        nonempty.append(bool(sh))
        sigs.append(hasher.signature(sh))

    removed: set[int] = set()
    dup_pairs: list[tuple[int, int]] = []

    for i, sig_i in enumerate(sigs):
        # 空 shingle 集合的全 MAX 签名只是哨兵，不表示短文档彼此相似。
        # 否则任意两篇少于 k 个词的文档都会被误判为完全重复。
        if not nonempty[i]:
            continue
        cands = index.candidates(sig_i)
        for j in cands:
            if j >= i:
                continue  # j 已经早于 i（索引里只有 < i 的）
            est = jaccard_estimate(sigs[j], sig_i)
            if est >= threshold:
                removed.add(i)
                dup_pairs.append((j, i))
                break  # i 已被标记，不再继续
        index.add(i, sig_i)

    kept = [i for i in range(len(docs)) if i not in removed]
    return kept, dup_pairs
