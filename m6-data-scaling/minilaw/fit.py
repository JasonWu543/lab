"""Phase 6.1 — Scaling Law 拟合（学生实现文件）

你的任务：实现 ScalingParams、predict_loss、fit_scaling_law，
让 tests/test_scaling.py 的 T1–T3 全绿。

闯关顺序（U1）：
  Step 1  实现 predict_loss（L = E + A/N^α + B/D^β）
           → 先过 T1 的标量/数组对拍
  Step 2  实现 fit_scaling_law 的核心优化循环
           → 过 T2（参数回收）
  Step 3  确保确定性（同 seed 同输出）+ 抗离群点
           → 过 T3

工程提示（这是脚手架知识，不是考点，可以直接参考）：
  - 参数化：用 (log_E, log_A, alpha, log_B, beta) 保证 E/A/B > 0
  - 优化器：scipy.optimize.minimize(method="L-BFGS-B")
  - 目标：Huber(delta=1e-3) 作用在 (log L_pred − log L_obs) 上
  - 多起点：用 np.random.default_rng(seed) 生成 n_starts 个随机起点
  - 取目标值最小的结果返回
  - bounds 示例：alpha/beta ∈ [0.01, 2.5]，log_A/log_B ∈ [-5, 15]，log_E ∈ [-5, 5]

运行测试：
  cd m6-data-scaling && python3 -m pytest tests/test_scaling.py::TestT1PredictLoss -x -q
  cd m6-data-scaling && python3 -m pytest tests/test_scaling.py::TestT2ParamRecovery -x -q
  cd m6-data-scaling && python3 -m pytest tests/test_scaling.py::TestT3Determinism -x -q

卡住 30 分钟以上再看参考答案：reference/6.1-scaling/fit_solution.py
（看完要能说出你的版本差在哪）
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ScalingParams:
    E: float
    A: float
    alpha: float
    B: float
    beta: float


def predict_loss(N, D, p: ScalingParams):
    """L = E + A/N^α + B/D^β。N/D 可为标量或 np.ndarray。"""
    # TODO U1-Step1: 实现闭式计算
    # 提示：np.asarray(N, dtype=float) 将标量和数组统一处理
    raise NotImplementedError("TODO: predict_loss")


def fit_scaling_law(
    N: np.ndarray,
    D: np.ndarray,
    L: np.ndarray,
    n_starts: int = 16,
    seed: int = 0,
) -> ScalingParams:
    """Chinchilla Approach 3 风格：
    在 log 空间参数化（拟合 log A、log B、log E 保证正性），
    目标 = Huber(delta=1e-3) on (log L_pred − log L_obs)，
    多起点（对 α,β ∈ [0,2.5]、logA/logB 网格随机）取最优。
    确定性：同 seed 同输入必须同输出。
    """
    # TODO U1-Step2: 实现多起点 L-BFGS-B 优化
    # 步骤：
    #   1. 将 N, D, L 转换为 log 空间（log_N, log_D, log_L）
    #   2. 定义内部目标函数 _loss_fn(theta, ...)
    #      theta = [log_E, log_A, alpha, log_B, beta]
    #   3. 设置 bounds（防止发散）
    #   4. 用 np.random.default_rng(seed) 生成 n_starts 个起点
    #   5. 对每个起点调用 scipy.optimize.minimize(method="L-BFGS-B")
    #   6. 返回目标值最小的结果（转换回 ScalingParams）
    raise NotImplementedError("TODO: fit_scaling_law")
