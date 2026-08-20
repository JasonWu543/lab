"""Phase 6.0 — n-gram 污染检测（学生实现文件）

你的任务：实现 ngram_set / contamination_rate / flag_contaminated，让 T6 全绿。

闯关顺序（U4 contamination）：
  Step 1  ngram_set(text, n)：小写词级 n-gram 集合
           - 空白分词后，不足 n 个词返回空集
           → 过 T6 的 test_t6_ngram_set_basic、test_t6_short_doc_no_crash
  Step 2  contamination_rate(train_doc, eval_ngrams, n)：
           - 计算 train_doc 的 n-gram 与 eval_ngrams 的交集占比
           - train_doc n-gram 为空时返回 0.0
           → 过 T6 的 test_t6_exact_overlap_detected、test_t6_no_overlap_rate_zero
             test_t6_contamination_rate_full_overlap
  Step 3  flag_contaminated(train_docs, eval_docs, n, threshold)：
           - 合并 eval_docs 所有 n-gram 为一个大集合
           - 逐篇 train_doc 计算污染率，> threshold 则标记
           → 过 T6 的 test_t6_flag_contaminated_batch、test_t6_truncated_eval_detected

思考题（写进 POSTMORTEM）：
  - 为什么污染检测用 8-gram 而不是 3-gram 或整句匹配？
    提示：3-gram 误报率如何？整句匹配的召回率如何？8-gram 的设计权衡是什么？

运行测试：
  cd m6-data-scaling && python3 -m pytest tests/test_data.py::TestT6Contamination -x -q
"""
from __future__ import annotations


def ngram_set(text: str, n: int = 8) -> set[str]:
    """小写词级 n-gram；不足 n 个词返回空集。"""
    raise NotImplementedError("U4 Step 1：实现 ngram_set")


def contamination_rate(train_doc: str, eval_ngrams: set[str], n: int = 8) -> float:
    """train_doc 的 n-gram 中出现在 eval_ngrams 里的比例（空集返回 0）。"""
    raise NotImplementedError("U4 Step 2：实现 contamination_rate")


def flag_contaminated(
    train_docs: list[str],
    eval_docs: list[str],
    n: int = 8,
    threshold: float = 0.1,
) -> list[int]:
    """返回污染率 > threshold 的 train doc 下标。"""
    raise NotImplementedError("U4 Step 3：实现 flag_contaminated")
