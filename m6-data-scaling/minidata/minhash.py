"""Phase 6.0 — MinHash + LSH 去重（学生实现文件）

你的任务：实现 shingle / MinHasher / LSHIndex / dedup，让 T2–T5 全绿。

闯关顺序（U2 MinHash → U3 LSH/dedup）：
  Step 1  shingle(text, k)：小写化 + 空白分词 → 词级 k-gram 集合
           → 先过 T3 的 test_t3_shingle_stability
  Step 2  MinHasher.__init__：用 seed 初始化通用哈希族参数 a_i, b_i
           哈希族公式（SPEC 冻结）：h_i(x) = (a_i * x + b_i) mod p，p = 2^61-1
           a_i ∈ [1, p-1]，b_i ∈ [0, p-1]，由 numpy.random.default_rng(seed) 生成
           → 过 T3 的 test_t3_same_seed_same_sig、test_t3_no_builtin_hash
  Step 3  MinHasher.signature：计算 (num_perm,) uint64 签名
           - shingle 稳定哈希：hashlib.blake2b(s.encode(), digest_size=8) 截断到 uint64
             （严禁用 Python 内置 hash()，它跨进程不稳定）
           - 为什么取 min 就是无偏估计？先想清楚再写
           - 空集返回全 uint64 最大值（np.iinfo(np.uint64).max）
           → 过 T2 全部、T3 全部
  Step 4  jaccard_estimate：签名逐位相等比例（一行就够）
           → 辅助 T2 的误差测试
  Step 5  LSHIndex：Banded LSH 索引（add / candidates）
           - rows = num_perm // bands
           - 同一 band 内所有行相同 → 触发候选
           - 每个 band 用 bytes 作 bucket key（band_sig.tobytes()）
           → 过 T4 全部
  Step 6  dedup：签名 → LSH 候选 → 复核 → 保留最早
           - 遍历文档，先查 candidates，再用 jaccard_estimate 复核 ≥ threshold
           - 重复簇保留下标最小者，结果保序
           → 过 T5 全部

运行测试：
  cd m6-data-scaling && python3 -m pytest tests/test_data.py::TestT2Unbiasedness tests/test_data.py::TestT3Stability tests/test_data.py::TestT4LSH tests/test_data.py::TestT5Dedup -x -q
"""
from __future__ import annotations

import numpy as np

# 梅森素数（通用哈希族模数，SPEC 冻结）
_P = (1 << 61) - 1
_MAX_UINT64 = np.uint64(0xFFFF_FFFF_FFFF_FFFF)


def shingle(text: str, k: int = 3) -> set[str]:
    """小写化 + 空白分词后的词级 k-gram 集合（k 个词用空格连接）。"""
    raise NotImplementedError("U2 Step 1：先实现 shingle，见文件头 Step 1")


class MinHasher:
    """通用哈希族 MinHash 签名器。

    哈希族公式（SPEC §3 冻结）：h_i(x) = (a_i * x + b_i) mod p
    p = 2^61-1（梅森素数），a_i ∈ [1, p-1]，b_i ∈ [0, p-1]。
    """

    def __init__(self, num_perm: int = 128, seed: int = 42):
        self.num_perm = num_perm
        # TODO U2 Step 2：用 numpy.random.default_rng(seed) 初始化 a_i, b_i
        # 注意：numpy 不支持 dtype=object 的 integers()；
        #   建议用 uint64 采样后转 list[int]，以便后续 Python int 大数运算不溢出
        raise NotImplementedError("U2 Step 2：初始化哈希族参数")

    def signature(self, shingles: set[str]) -> np.ndarray:
        """(num_perm,) uint64。空集合返回全 uint64 最大值。

        关键问题：为什么对每个哈希函数取 shingle 哈希值的最小值，
        就能无偏估计 Jaccard 相似度？先想清楚这个问题再动手写。
        """
        raise NotImplementedError("U2 Step 3：实现 MinHash 签名")


def jaccard_estimate(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    """签名逐位相等比例（一行实现）。"""
    raise NotImplementedError("U2 Step 4：一行即可")


class LSHIndex:
    """Banded LSH 索引。

    rows = num_perm // bands；同一 band 内所有行相同才触发候选。
    数据结构建议：list[dict[bytes, list[int]]]，外层按 band，内层按 band 哈希值。
    """

    def __init__(self, num_perm: int = 128, bands: int = 32):
        self.num_perm = num_perm
        self.bands = bands
        self.rows = num_perm // bands
        # TODO U3 Step 5：初始化 bucket 结构
        raise NotImplementedError("U3 Step 5：初始化 LSH bucket 结构")

    def add(self, doc_id: int, sig: np.ndarray) -> None:
        # TODO：将 sig 切成 bands 段，每段 tobytes() 作 key，加入对应 bucket
        raise NotImplementedError("U3 Step 5：实现 add")

    def candidates(self, sig: np.ndarray) -> set[int]:
        """至少一个 band 完全相同的已入库 doc_id 集合。"""
        raise NotImplementedError("U3 Step 5：实现 candidates")


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

    流程提示：
        1. 依次计算每篇文档的签名
        2. 对每篇文档：先查 LSHIndex 候选 → 再用 jaccard_estimate 复核 ≥ threshold
        3. 重复的后者标记为去除，保留前者（下标更小的）
        4. 查完候选后再把当前文档加入 index（流式去重）
    """
    raise NotImplementedError("U3 Step 6：实现端到端 dedup")
