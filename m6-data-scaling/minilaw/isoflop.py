"""Phase 6.1 — isoFLOP 谷底提取（学生实现文件）

你的任务：实现 isoflop_minima，让 tests/test_scaling.py 的 T6 全绿。

闯关顺序（U3）：
  Step 1  实现分组逻辑（按 C 分组，过滤点数 < 3 的组）
  Step 2  对每组做抛物线拟合取谷底
           → 过 T6（谷底回收误差 < 5%；点数不足的组跳过不崩）

思路提示：
  - 对每个 C 组，用 np.polyfit(log_N, L, 2) 拟合二次多项式
    系数 [a, b, c]，即 L ≈ a·x² + b·x + c（x = log N）
  - 顶点（谷底）在 x_opt = -b/(2a)，即 N_opt = exp(x_opt)
  - 注意 a > 0 才有最小值（开口向上）；否则退化处理
  - np.polyval(coeffs, x_opt) 给出 L_min

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
    # TODO U3-Step2: 对每组做 np.polyfit(log_N, L, 2)，提取谷底 N_opt 和 L_min
    raise NotImplementedError("TODO: isoflop_minima")
