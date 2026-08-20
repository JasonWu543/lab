"""Phase 6.1 参考答案 — fit.py（学生不可提前查看）

实现要点（也是 POSTMORTEM 考点）：
  1. 为何 log 参数化：保证 E/A/B > 0，同时让优化景观更平滑。
  2. 为何用 Huber loss（delta=1e-3）：对 log-residual 的大离群点降权，
     模拟 Chinchilla Approach 3 的鲁棒回归。
  3. 多起点 + L-BFGS-B：损失面多峰，单点易陷局部极值。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass
class ScalingParams:
    E: float
    A: float
    alpha: float
    B: float
    beta: float


def predict_loss(N, D, p: ScalingParams):
    """L = E + A/N^α + B/D^β。N/D 可为标量或 np.ndarray。"""
    return p.E + p.A / np.asarray(N, dtype=float) ** p.alpha + p.B / np.asarray(D, dtype=float) ** p.beta


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------

def _huber(r, delta=1e-3):
    """逐元素 Huber on 残差 r，均值标量。"""
    abs_r = np.abs(r)
    return np.where(abs_r <= delta,
                    0.5 * r ** 2,
                    delta * (abs_r - 0.5 * delta)).mean()


def _loss_fn(theta, log_N, log_D, log_L_obs, delta=1e-3):
    """目标函数：Huber on (log L_pred − log L_obs)。

    theta = [log_E, log_A, alpha, log_B, beta]
    """
    log_E, log_A, alpha, log_B, beta = theta
    E = np.exp(log_E)
    A = np.exp(log_A)
    B = np.exp(log_B)
    L_pred = E + A * np.exp(-alpha * log_N) + B * np.exp(-beta * log_D)
    # 防止 log(负数)：若 L_pred <= 0 返回大值
    if np.any(L_pred <= 0):
        return 1e10
    log_L_pred = np.log(L_pred)
    residual = log_L_pred - log_L_obs
    return _huber(residual, delta)


def fit_scaling_law(
    N: np.ndarray,
    D: np.ndarray,
    L: np.ndarray,
    n_starts: int = 16,
    seed: int = 0,
) -> ScalingParams:
    """Chinchilla Approach 3 风格的多起点 L-BFGS-B 拟合。

    参数空间：theta = [log_E, log_A, alpha, log_B, beta]
    目标：min Huber(delta=1e-3) on (log L_pred − log L_obs)
    确定性：同 seed 同输入必须同输出。
    """
    log_N = np.log(np.asarray(N, dtype=float))
    log_D = np.log(np.asarray(D, dtype=float))
    log_L = np.log(np.asarray(L, dtype=float))

    # 参数边界（防止发散）
    bounds = [
        (-5.0, 5.0),    # log_E：E ∈ [e^-5, e^5]
        (-5.0, 15.0),   # log_A：A ∈ [e^-5, e^15]
        (0.01, 2.5),    # alpha
        (-5.0, 15.0),   # log_B
        (0.01, 2.5),    # beta
    ]

    rng = np.random.default_rng(seed)

    best_val = np.inf
    best_theta = None

    for _ in range(n_starts):
        # 随机起点
        log_E0 = rng.uniform(-1.0, 2.0)
        log_A0 = rng.uniform(2.0, 8.0)
        alpha0 = rng.uniform(0.1, 1.0)
        log_B0 = rng.uniform(2.0, 8.0)
        beta0 = rng.uniform(0.1, 1.0)
        theta0 = [log_E0, log_A0, alpha0, log_B0, beta0]

        res = minimize(
            _loss_fn,
            theta0,
            args=(log_N, log_D, log_L),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 2000, "ftol": 1e-15, "gtol": 1e-8},
        )

        if res.fun < best_val:
            best_val = res.fun
            best_theta = res.x

    log_E, log_A, alpha, log_B, beta = best_theta
    return ScalingParams(
        E=float(np.exp(log_E)),
        A=float(np.exp(log_A)),
        alpha=float(alpha),
        B=float(np.exp(log_B)),
        beta=float(beta),
    )
