"""Phase 6.0 — 质量过滤（学生实现文件）

你的任务：实现 quality_filter 与 apply_filters，让 tests/test_data.py 的 T1 全绿。

闯关顺序（U1 filters）：
  Step 1  word_count 规则：空白切分后词数在 [20, 100_000]
           → 先过 T1 的 test_t1_word_count_*
  Step 2  mean_word_len 规则：平均词长在 [2, 12]
           → 过 T1 的 test_t1_mean_word_len_*
  Step 3  symbol_ratio 规则：'#' 与 '…'/'...' 出现次数 / 词数 < 0.1
           → 过 T1 的 test_t1_symbol_ratio
  Step 4  bullet_ratio 规则：以 '-' 或 '*' 开头的行占比 < 0.9
           → 过 T1 的 test_t1_bullet_ratio
  Step 5  dup_line_ratio 规则：重复行占比（重复出现的行数 / 总行数）< 0.3
           → 过 T1 的 test_t1_dup_line_ratio
  Step 6  alpha_ratio 规则：含字母的词占比 > 0.6
           → 过 T1 的 test_t1_alpha_ratio
  Step 7  apply_filters 批量接口 + FilterStats 统计
           → 过 T1 的 test_t1_apply_filters_stats

关键约定：
- quality_filter 按顺序检查，第一条违反的规则即返回 (False, 规则名)，全过返回 (True, None)
- 规则名必须与接口文档完全匹配（用于测试精确比对）
- symbol_ratio 同时计数 '…'（Unicode 省略号）和 '...'（三个 ASCII 点，作为整体一次计数）

运行测试：
  cd m6-data-scaling && python3 -m pytest tests/test_data.py::TestT1Filters -x -q
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FilterStats:
    kept: int = 0
    dropped: int = 0
    drop_reasons: dict[str, int] = field(default_factory=dict)


def quality_filter(doc: str) -> tuple[bool, str | None]:
    """按顺序检查六条规则，第一条违反即返回 (False, 规则名)；全过返回 (True, None)。

    规则集（Gopher 子集，英文语料）：
      word_count      : 词数（空白切分）在 [20, 100_000]
      mean_word_len   : 平均词长在 [2, 12]
      symbol_ratio    : '#' 与 '…'/'...' 出现次数 / 词数 < 0.1
      bullet_ratio    : 以 '-' 或 '*' 开头的行占比 < 0.9
      dup_line_ratio  : 重复行占比（重复出现的行数/总行数）< 0.3
      alpha_ratio     : 含字母的词占比 > 0.6
    """
    raise NotImplementedError(
        "U1 Step 1–6：依次实现六条规则，见文件头闯关顺序"
    )


def apply_filters(docs: list[str]) -> tuple[list[str], FilterStats]:
    """对文档列表批量应用 quality_filter，收集统计信息。

    提示：遍历 docs，调用 quality_filter，分别累积 kept/dropped 与 drop_reasons。
    """
    raise NotImplementedError(
        "U1 Step 7：先实现 quality_filter，再来写 apply_filters"
    )
