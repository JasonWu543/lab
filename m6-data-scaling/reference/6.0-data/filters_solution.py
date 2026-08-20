"""Phase 6.0 参考答案 — filters（不得在备课前发给学生）

规则集：Gopher 风格六条，详见 SPEC §3。
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FilterStats:
    kept: int = 0
    dropped: int = 0
    drop_reasons: dict[str, int] = field(default_factory=dict)


def quality_filter(doc: str) -> tuple[bool, str | None]:
    """按顺序检查，第一条违反的规则即返回 (False, 规则名)；全过返回 (True, None)。"""
    words = doc.split()
    n_words = len(words)

    # 1. word_count
    if not (20 <= n_words <= 100_000):
        return False, "word_count"

    # 2. mean_word_len
    mean_len = sum(len(w) for w in words) / n_words
    if not (2.0 <= mean_len <= 12.0):
        return False, "mean_word_len"

    # 3. symbol_ratio — '#' 与 '…'/'...' 出现次数 / 词数 < 0.1
    symbol_count = doc.count("#") + doc.count("…") + doc.count("...")
    if symbol_count / n_words >= 0.1:
        return False, "symbol_ratio"

    # 4. bullet_ratio — 以 '-' 或 '*' 开头的行占比 < 0.9
    lines = doc.splitlines()
    if lines:
        bullet_lines = sum(1 for l in lines if l.lstrip().startswith(("-", "*")))
        if bullet_lines / len(lines) >= 0.9:
            return False, "bullet_ratio"

    # 5. dup_line_ratio — 重复行占比 < 0.3
    if lines:
        from collections import Counter
        counts = Counter(lines)
        dup_line_count = sum(cnt for cnt in counts.values() if cnt > 1)
        if dup_line_count / len(lines) >= 0.3:
            return False, "dup_line_ratio"

    # 6. alpha_ratio — 含字母的词占比 > 0.6
    alpha_words = sum(1 for w in words if any(c.isalpha() for c in w))
    if alpha_words / n_words <= 0.6:
        return False, "alpha_ratio"

    return True, None


def apply_filters(docs: list[str]) -> tuple[list[str], FilterStats]:
    stats = FilterStats()
    kept: list[str] = []
    for doc in docs:
        ok, reason = quality_filter(doc)
        if ok:
            kept.append(doc)
            stats.kept += 1
        else:
            stats.dropped += 1
            stats.drop_reasons[reason] = stats.drop_reasons.get(reason, 0) + 1
    return kept, stats
