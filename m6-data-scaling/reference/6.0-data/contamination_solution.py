"""Phase 6.0 参考答案 — contamination（不得在备课前发给学生）

n-gram 污染检测：词级 n-gram 集合交集占比。
"""
from __future__ import annotations


def ngram_set(text: str, n: int = 8) -> set[str]:
    """小写词级 n-gram；不足 n 个词返回空集。"""
    words = text.lower().split()
    if len(words) < n:
        return set()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def contamination_rate(train_doc: str, eval_ngrams: set[str], n: int = 8) -> float:
    """train_doc 的 n-gram 中出现在 eval_ngrams 里的比例（空集返回 0）。"""
    train_ngrams = ngram_set(train_doc, n=n)
    if not train_ngrams:
        return 0.0
    overlap = train_ngrams & eval_ngrams
    return len(overlap) / len(train_ngrams)


def flag_contaminated(
    train_docs: list[str],
    eval_docs: list[str],
    n: int = 8,
    threshold: float = 0.1,
) -> list[int]:
    """返回污染率 > threshold 的 train doc 下标。"""
    # 合并 eval 所有 n-gram 到一个大集合
    eval_ngrams: set[str] = set()
    for doc in eval_docs:
        eval_ngrams |= ngram_set(doc, n=n)

    flagged: list[int] = []
    for idx, doc in enumerate(train_docs):
        rate = contamination_rate(doc, eval_ngrams, n=n)
        if rate > threshold:
            flagged.append(idx)
    return flagged
