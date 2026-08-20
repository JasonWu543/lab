"""Phase 6.1 参考答案 — isoflop.py（学生不可提前查看）

isoFLOP 谷底提取：对每个 FLOPs 预算 C，
在 (log N, L) 上做二次多项式拟合（np.polyfit），
取顶点 N_opt = exp(-b/(2a))，对应 L_min = poly(log N_opt)。
组内点数 < 3 时跳过（无法做二次拟合）。
"""

from __future__ import annotations

import numpy as np


def isoflop_minima(runs: list[dict]) -> list[dict]:
    """每个 C 组内对 (log N, L) 做抛物线拟合取谷底。

    Args:
        runs: [{"C": float, "N": float, "L": float}, ...]

    Returns:
        [{"C": float, "N_opt": float, "L_min": float}, ...]，已按 C 排序。
        组内点数 < 3 的 C 直接跳过。
    """
    # 按 C 分组
    groups: dict[float, list] = {}
    for r in runs:
        c = float(r["C"])
        groups.setdefault(c, []).append(r)

    results = []
    for c in sorted(groups.keys()):
        pts = groups[c]
        if len(pts) < 3:
            continue  # 点数不足，跳过

        log_N = np.array([np.log(r["N"]) for r in pts])
        L_vals = np.array([float(r["L"]) for r in pts])

        # 二次多项式拟合：L ≈ a·(log N)^2 + b·(log N) + c_const
        coeffs = np.polyfit(log_N, L_vals, 2)  # [a, b, c_const]
        a_coef, b_coef, c_coef = coeffs

        if a_coef <= 0:
            # 开口向下，没有最小值——退化为端点最小
            idx = int(np.argmin(L_vals))
            N_opt = float(np.exp(log_N[idx]))
            L_min = float(L_vals[idx])
        else:
            log_N_opt = -b_coef / (2.0 * a_coef)
            N_opt = float(np.exp(log_N_opt))
            L_min = float(np.polyval(coeffs, log_N_opt))

        results.append({"C": c, "N_opt": N_opt, "L_min": L_min})

    return results
