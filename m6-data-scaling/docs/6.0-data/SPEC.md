# SPEC — Phase 6.0: 数据工程（质量过滤 / MinHash 去重 / 污染检测）

> 状态：FROZEN（接口已冻结）
> 模式：Copilot —— MinHash/LSH 与污染检测核心手写；过滤规则与消融脚手架给足
> 算力：correctness 全本地 CPU；数据消融实验 M 级（W10 副线/W12）
> 工期：约 0.5 周

## 1. 问题

搭一条最小但真实的预训练数据管线：Gopher 风格规则过滤 →
MinHash+LSH 近重复去重 → 对评测集的 n-gram 污染检测，
然后做一个「只改数据、其余全固定」的消融实验。

学完必须能回答（写进 POSTMORTEM）：
- MinHash 为什么能无偏估计 Jaccard？num_perm 决定什么？
- LSH 的 bands/rows 怎么决定「相似度阈值曲线」的形状？（画出 S 曲线）
- 为什么污染检测用 8-gram 而不是 3-gram 或整句匹配？
- 去重对 validation loss 的影响是正是负？先预测再看消融结果。

## 2. 范围与非目标

范围：文档级规则过滤（规则集冻结见 §3）、词级 shingle + MinHash 签名 +
banded LSH、8-gram 污染率、TinyStories 上的三组消融（raw/filtered/deduped）。
非目标：不做模型质量分类器过滤、不做语言识别、不做 exact-dedup 的
suffix array（近重复足够）、不做分布式。

## 3. 冻结接口（minidata/）

```python
# minidata/filters.py —— 规则集冻结（Gopher 子集，针对英文语料）
@dataclass
class FilterStats:
    kept: int = 0
    dropped: int = 0
    drop_reasons: dict[str, int] = field(default_factory=dict)  # 规则名→计数

def quality_filter(doc: str) -> tuple[bool, str | None]:
    """按顺序检查，第一条违反的规则即返回 (False, 规则名)：
      word_count      : 词数（空白切分）在 [20, 100_000]
      mean_word_len   : 平均词长在 [2, 12]
      symbol_ratio    : '#' 与 '…'/'...' 出现次数 / 词数 < 0.1
      bullet_ratio    : 以 '-' 或 '*' 开头的行占比 < 0.9
      dup_line_ratio  : 重复行占比（重复出现的行数/总行数）< 0.3
      alpha_ratio     : 含字母的词占比 > 0.6
    全过返回 (True, None)。"""

def apply_filters(docs: list[str]) -> tuple[list[str], FilterStats]: ...

# minidata/minhash.py
def shingle(text: str, k: int = 3) -> set[str]:
    """小写化 + 空白分词后的词级 k-gram 集合（k 个词用空格连接）。"""

class MinHasher:
    def __init__(self, num_perm: int = 128, seed: int = 42): ...
    def signature(self, shingles: set[str]) -> np.ndarray:
        """(num_perm,) uint64。空集合返回全 uint64 最大值。
        实现约定：用 (a_i * h + b_i) mod p 的通用哈希族（p 取梅森素数
        2^61-1），h 为 shingle 的稳定哈希（如 blake2b 截断——
        不许用 Python 内置 hash()，它跨进程不稳定）。"""

def jaccard_estimate(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    """签名逐位相等比例。"""

class LSHIndex:
    def __init__(self, num_perm: int = 128, bands: int = 32): ...
    def add(self, doc_id: int, sig: np.ndarray) -> None: ...
    def candidates(self, sig: np.ndarray) -> set[int]:
        """至少一个 band 完全相同的已入库 doc_id 集合。"""

def dedup(docs: list[str], threshold: float = 0.8,
          num_perm: int = 128, bands: int = 32, k: int = 3,
          seed: int = 42) -> tuple[list[int], list[tuple[int, int]]]:
    """返回 (保留的 doc 下标（保序，重复簇保留最早的）,
             判定为重复的 (早, 晚) 对列表)。
    流程：签名 → LSH 候选 → 候选对用 jaccard_estimate 复核 ≥ threshold。
    空 shingle 集合的全最大值签名仅是哨兵；这类文档之间没有可用于近重复
    判定的 shingle，必须分别保留。"""

# minidata/contamination.py
def ngram_set(text: str, n: int = 8) -> set[str]:
    """小写词级 n-gram；不足 n 个词返回空集。"""

def contamination_rate(train_doc: str, eval_ngrams: set[str], n: int = 8) -> float:
    """train_doc 的 n-gram 中出现在 eval_ngrams 里的比例（空集返回 0）。"""

def flag_contaminated(train_docs: list[str], eval_docs: list[str],
                      n: int = 8, threshold: float = 0.1) -> list[int]:
    """返回污染率 > threshold 的 train doc 下标。"""

# scripts/ablate_data.py —— 消融脚手架（完整）：TinyStories 采样固定 token 预算，
#   raw / filtered / filtered+deduped 三组，同一 tiny 模型与超参，比 val loss
```

## 4. 验收标准（tests/test_data.py，CPU）

| 编号 | 通过条件 |
| --- | --- |
| T1 | 六条过滤规则逐条构造违例文档，返回的规则名精确匹配；干净文档全过；drop_reasons 计数正确 |
| T2 | MinHash 无偏性：构造已知真实 Jaccard 的文档对（如 0.2/0.5/0.8），num_perm=256 时估计误差 < 0.08；同一文档估计 = 1.0；不相交 ≈ 0 |
| T3 | 签名稳定性：同 seed 两次构建 MinHasher 签名逐位相等（跨进程稳定哈希的回归测试）|
| T4 | LSH 召回/过滤：Jaccard ≥ 0.9 的对必为候选（该参数下漏检率理论上 <1e-6）；Jaccard ≤ 0.1 的对不为候选（构造 20 对无一误报）|
| T5 | dedup 端到端：100 篇文档中埋 10 组近重复（轻微改词），全部检出且保留最早者；无近重复的干净集零误杀 |
| T6 | 污染检测：把 eval 句子原样/截断嵌进 train doc 能检出；无重叠文档污染率为 0；不足 8 词的边界不崩 |

## 5. 产物

- `minidata/*.py` 全绿 + 消融实验报告（预测 vs 实际）
- `docs/6.0-data/POSTMORTEM.md`（含 LSH S 曲线手绘/手算）
