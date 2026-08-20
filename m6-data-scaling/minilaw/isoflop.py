"""Phase 6.1 — isoFLOP 谷底提取（学生实现文件）

你的任务：实现 isoflop_minima，让 tests/test_scaling.py 的 T6 全绿。

闯关顺序（U3）：
  Step 1  实现分组逻辑（按 C 分组，过滤点数 < 3 的组）
  Step 2  对每组做抛物线拟合取谷底
           → 过 T6（谷底回收误差 < 5%；点数不足的组跳过不崩）

思路提示：
  - 对每个 C 组，用 np.polyfit(log_N, L, 2) 拟合二次多项式
    系数 [a, b, c]，即 L ≈ a·x² + b·x + c（x = log N）
  - 对这个二次式求导：什么条件下驻点才是谷底？
  - 从 log N 空间变回 N 空间时需要做什么变换？
  - 如果拟合曲线没有谷底，怎样给出不越过观测范围的退化结果？

运行测试：
  cd m6-data-scaling && python3 -m pytest tests/test_scaling.py::TestT6IsoflopMinima -x -q

卡住 30 分钟以上再看参考答案：reference/6.1-scaling/isoflop_solution.py
"""

from __future__ import annotations

import numpy as np


def isoflop_minima(runs: list[dict]) -> list[dict]:
    """runs: [{"C": float, "N": float, "L": float}, ...]（同一 C 多个 N）。
    每个 C 组内对 (log N, L) 做抛物线拟合取谷底，
    返回 [{"C", "N_opt", "L_min"}, ...]。组内点数 < 3 时跳过该组。
    """
    # TODO U3-Step1: 按 C 分组
    # TODO U3-Step2: 对每组拟合二次曲线，并从曲线性质推导谷底
    raise NotImplementedError("TODO: isoflop_minima")
