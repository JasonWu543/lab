"""Phase 6.1 参考答案 — optimal.py（学生不可提前查看）

核心推导（这是 POSTMORTEM 的手推作业，骨架绝不出现下列公式）：

目标：min_{N,D}  L = E + A/N^α + B/D^β
约束：          6ND = C   →  D = C/(6N)

代入消元（从此只剩 N）：
  L(N) = E + A·N^{-α} + B·[C/(6N)]^{-β}
       = E + A·N^{-α} + B·(6/C)^β · N^β

对 N 求导并令 dL/dN = 0：
  -α·A·N^{-α-1} + β·B·(6/C)^β · N^{β-1} = 0
  β·B·(6/C)^β · N^{β-1} = α·A·N^{-α-1}
  N^{α+β} = (αA) / (βB) · (C/6)^β

∴  N* = [(αA)/(βB)]^{1/(α+β)} · (C/6)^{β/(α+β)}
       = G · (C/6)^{β/(α+β)}    其中 G = [(αA)/(βB)]^{1/(α+β)}

   D* = C / (6 · N*)

指数：a = β/(α+β),  b = α/(α+β),  a+b = 1  ✓
"""

from __future__ import annotations

from .fit_solution import ScalingParams


def compute_optimal(C: float, p: ScalingParams) -> tuple[float, float]:
    """min_{N,D} L s.t. 6ND = C 的闭式解 (N*, D*)。

    解析式（不许数值搜索）：
      G   = (α·A / (β·B))^{1/(α+β)}
      N*  = G · (C/6)^{β/(α+β)}
      D*  = C / (6·N*)
    """
    alpha, beta = p.alpha, p.beta
    A, B = p.A, p.B
    exp_sum = alpha + beta
    G = (alpha * A / (beta * B)) ** (1.0 / exp_sum)
    N_star = G * (C / 6.0) ** (beta / exp_sum)
    D_star = C / (6.0 * N_star)
    return float(N_star), float(D_star)


def optimal_exponents(p: ScalingParams) -> tuple[float, float]:
    """返回 (a, b)：N* ∝ C^a, D* ∝ C^b。

    a = β/(α+β),  b = α/(α+β)
    验证：a + b = 1（理论上严格成立）。
    """
    alpha, beta = p.alpha, p.beta
    denom = alpha + beta
    a = beta / denom
    b = alpha / denom
    return float(a), float(b)
