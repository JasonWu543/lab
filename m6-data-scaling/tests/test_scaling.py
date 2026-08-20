"""Phase 6.1 验收测试 — T1–T7（CPU，全合成数据，≤90s）

运行：
  cd m6-data-scaling && python3 -m pytest tests/test_scaling.py -x -q

闯关顺序建议（配合骨架 TODO）：
  U1  predict_loss / fit_scaling_law  →  T1, T2, T3
  U2  compute_optimal / optimal_exponents  →  T4, T5
  U3  isoflop_minima  →  T6
  U4  端到端外推  →  T7
"""

import math

import numpy as np
import pytest

# ── 学生实现（minilaw/）────────────────────────────────────────────────────
from minilaw.fit import ScalingParams, predict_loss, fit_scaling_law
from minilaw.optimal import compute_optimal, optimal_exponents
from minilaw.isoflop import isoflop_minima


# ── 共用的 Chinchilla 真参数 ──────────────────────────────────────────────
TRUE_PARAMS = ScalingParams(E=1.69, A=406.4, alpha=0.34, B=410.7, beta=0.28)
NOISE_SIGMA = 0.02   # 乘性噪声标准差
N_POINTS    = 40     # 合成数据点数


def _synth_data(params: ScalingParams = TRUE_PARAMS,
                n: int = N_POINTS,
                seed: int = 42,
                noise_sigma: float = NOISE_SIGMA,
                N_range=(1e7, 1e10),
                D_range=(1e7, 1e10)):
    """生成合成 (N, D, L) 数据（对数均匀 + 乘性噪声）。"""
    rng = np.random.default_rng(seed)
    N = np.exp(rng.uniform(math.log(N_range[0]), math.log(N_range[1]), n))
    D = np.exp(rng.uniform(math.log(D_range[0]), math.log(D_range[1]), n))
    L_clean = predict_loss(N, D, params)
    L = L_clean * np.exp(rng.normal(0, noise_sigma, n))
    return N, D, L


# ════════════════════════════════════════════════════════════════════════════
# T1 — predict_loss：闭式对拍（标量 + 数组广播）
# ════════════════════════════════════════════════════════════════════════════
class TestT1PredictLoss:
    def test_scalar(self):
        p = TRUE_PARAMS
        N, D = 1e9, 1e10
        expected = p.E + p.A / N ** p.alpha + p.B / D ** p.beta
        assert abs(predict_loss(N, D, p) - expected) < 1e-10

    def test_array_shape(self):
        p = TRUE_PARAMS
        N = np.array([1e8, 1e9, 1e10])
        D = np.array([1e9, 1e10, 1e11])
        L = predict_loss(N, D, p)
        assert L.shape == (3,)

    def test_array_values(self):
        p = TRUE_PARAMS
        N = np.array([1e8, 1e9])
        D = np.array([1e9, 1e10])
        expected = p.E + p.A / N ** p.alpha + p.B / D ** p.beta
        np.testing.assert_allclose(predict_loss(N, D, p), expected, rtol=1e-10)

    def test_monotone_in_N(self):
        """L 关于 N 单调递减（更大模型 → 更低损失，给定 D）。"""
        p = TRUE_PARAMS
        D = 1e10
        N_small = 1e8
        N_large = 1e10
        assert predict_loss(N_small, D, p) > predict_loss(N_large, D, p)

    def test_monotone_in_D(self):
        """L 关于 D 单调递减（更多数据 → 更低损失，给定 N）。"""
        p = TRUE_PARAMS
        N = 1e9
        assert predict_loss(N, 1e8, p) > predict_loss(N, 1e12, p)


# ════════════════════════════════════════════════════════════════════════════
# T2 — 参数回收：已知参数生成数据，拟合后 α/β 误差 < 0.05、E 误差 < 0.2
# ════════════════════════════════════════════════════════════════════════════
class TestT2ParamRecovery:
    @pytest.fixture(scope="class")
    def fitted(self):
        N, D, L = _synth_data()
        return fit_scaling_law(N, D, L, n_starts=32, seed=0)

    def test_alpha_recovery(self, fitted):
        assert abs(fitted.alpha - TRUE_PARAMS.alpha) < 0.05, (
            f"alpha={fitted.alpha:.4f}，真值={TRUE_PARAMS.alpha}"
        )

    def test_beta_recovery(self, fitted):
        assert abs(fitted.beta - TRUE_PARAMS.beta) < 0.05, (
            f"beta={fitted.beta:.4f}，真值={TRUE_PARAMS.beta}"
        )

    def test_E_recovery(self, fitted):
        # E（不可逆熵底）与 A/B 高度相关，容差适当放宽
        assert abs(fitted.E - TRUE_PARAMS.E) < 0.2, (
            f"E={fitted.E:.4f}，真值={TRUE_PARAMS.E}"
        )


# ════════════════════════════════════════════════════════════════════════════
# T3 — 拟合确定性 & 抗离群点（Huber 的意义）
# ════════════════════════════════════════════════════════════════════════════
class TestT3Determinism:
    def test_same_seed_same_result(self):
        N, D, L = _synth_data()
        p1 = fit_scaling_law(N, D, L, n_starts=16, seed=7)
        p2 = fit_scaling_law(N, D, L, n_starts=16, seed=7)
        assert p1.alpha == p2.alpha
        assert p1.beta == p2.beta
        assert p1.E == p2.E

    def test_robust_to_outliers(self):
        """污染 2 个 10× 离群 L 后，α/β 仍在容差 0.05 内。"""
        N, D, L = _synth_data(seed=99)
        L_corrupt = L.copy()
        L_corrupt[0] *= 10.0
        L_corrupt[1] *= 10.0
        fitted = fit_scaling_law(N, D, L_corrupt, n_starts=32, seed=0)
        assert abs(fitted.alpha - TRUE_PARAMS.alpha) < 0.05, (
            f"outlier 后 alpha={fitted.alpha:.4f}"
        )
        assert abs(fitted.beta - TRUE_PARAMS.beta) < 0.05, (
            f"outlier 后 beta={fitted.beta:.4f}"
        )


# ════════════════════════════════════════════════════════════════════════════
# T4 — 闭式 vs 数值网格搜索（相对误差 < 1%）
# ════════════════════════════════════════════════════════════════════════════
class TestT4ClosedFormVsGrid:
    def _grid_search(self, C, p, n_grid=10_000):
        """暴力网格搜索最优 N（10^4 点）作为数值基准。"""
        N_vals = np.logspace(
            math.log10(1e6),
            math.log10(C / 6.0),  # D ≥ 1
            n_grid
        )
        D_vals = C / (6.0 * N_vals)
        # 过滤 D < 1
        mask = D_vals >= 1.0
        N_vals, D_vals = N_vals[mask], D_vals[mask]
        L_vals = predict_loss(N_vals, D_vals, p)
        idx = np.argmin(L_vals)
        return N_vals[idx]

    @pytest.mark.parametrize("C", [1e17, 1e20, 1e23])
    def test_N_star_vs_grid(self, C):
        p = TRUE_PARAMS
        N_analytic, D_analytic = compute_optimal(C, p)
        N_grid = self._grid_search(C, p)
        rel_err = abs(N_analytic - N_grid) / N_grid
        assert rel_err < 0.01, (
            f"C={C:.0e}: 解析 N*={N_analytic:.3e}, 网格 N*={N_grid:.3e}, "
            f"相对误差={rel_err:.4f}"
        )
        # D* 必须满足约束恒等式 6·N*·D* = C（抓 D* 少除 6 之类的笔误）
        assert abs(6.0 * N_analytic * D_analytic - C) / C < 1e-8, (
            f"D* 违反约束：6·N*·D*={6.0 * N_analytic * D_analytic:.3e} != C={C:.0e}"
        )


# ════════════════════════════════════════════════════════════════════════════
# T5 — optimal_exponents：a+b=1；α=β → a=b=0.5
# ════════════════════════════════════════════════════════════════════════════
class TestT5Exponents:
    def test_sum_to_one(self):
        a, b = optimal_exponents(TRUE_PARAMS)
        assert abs(a + b - 1.0) < 1e-9, f"a+b={a+b}"

    def test_equal_alpha_beta(self):
        p = ScalingParams(E=1.69, A=406.4, alpha=0.5, B=410.7, beta=0.5)
        a, b = optimal_exponents(p)
        assert abs(a - 0.5) < 1e-9, f"α=β 时 a={a}"
        assert abs(b - 0.5) < 1e-9, f"α=β 时 b={b}"

    def test_chinchilla_values(self):
        """Chinchilla 真参数下 a ≈ β/(α+β) = 0.28/0.62 ≈ 0.452。"""
        a, b = optimal_exponents(TRUE_PARAMS)
        expected_a = TRUE_PARAMS.beta / (TRUE_PARAMS.alpha + TRUE_PARAMS.beta)
        assert abs(a - expected_a) < 1e-9

    def test_N_star_exponent_consistent(self):
        """验证 N* ∝ C^a：C 增大 10×，N* 应增大 10^a 倍。"""
        C1, C2 = 1e18, 1e19
        p = TRUE_PARAMS
        N1, _ = compute_optimal(C1, p)
        N2, _ = compute_optimal(C2, p)
        a, _ = optimal_exponents(p)
        expected_ratio = (C2 / C1) ** a
        actual_ratio = N2 / N1
        assert abs(actual_ratio - expected_ratio) / expected_ratio < 1e-6


# ════════════════════════════════════════════════════════════════════════════
# T6 — isoflop_minima：抛物线谷底回收；点数不足跳过不崩
# ════════════════════════════════════════════════════════════════════════════
class TestT6IsoflopMinima:
    def _make_parabola_runs(self, C, N_opt_true, L_min_true, noise=0.0, n=7, seed=1):
        """在 (log N, L) 上合成抛物线数据（可选加噪声）。"""
        rng = np.random.default_rng(seed)
        log_N_opt = math.log(N_opt_true)
        log_N_vals = np.linspace(log_N_opt - 1.5, log_N_opt + 1.5, n)
        # L = L_min + k*(log N - log N_opt)^2
        k = 0.3
        L_vals = L_min_true + k * (log_N_vals - log_N_opt) ** 2
        if noise > 0:
            L_vals += rng.normal(0, noise, n)
        return [{"C": C, "N": math.exp(x), "L": float(l)}
                for x, l in zip(log_N_vals, L_vals)]

    def test_single_group_no_noise(self):
        C = 1e20
        N_opt_true = 5e9
        L_min_true = 2.5
        runs = self._make_parabola_runs(C, N_opt_true, L_min_true)
        result = isoflop_minima(runs)
        assert len(result) == 1
        r = result[0]
        assert abs(r["N_opt"] - N_opt_true) / N_opt_true < 0.05, (
            f"N_opt={r['N_opt']:.3e} vs 真值 {N_opt_true:.3e}"
        )
        assert abs(r["L_min"] - L_min_true) < 0.05 * L_min_true, (
            f"L_min={r['L_min']:.4f} vs 真值 {L_min_true:.4f}"
        )

    def test_multiple_groups(self):
        runs = []
        truths = {}
        for i, C in enumerate([1e18, 1e20, 1e22]):
            N_opt = 1e8 * (10 ** i)
            L_min = 3.0 - 0.3 * i
            truths[C] = (N_opt, L_min)
            runs.extend(self._make_parabola_runs(C, N_opt, L_min, seed=i))

        results = isoflop_minima(runs)
        assert len(results) == 3
        for r in results:
            N_true, L_true = truths[r["C"]]
            assert abs(r["N_opt"] - N_true) / N_true < 0.05
            assert abs(r["L_min"] - L_true) < 0.05 * L_true

    def test_skip_small_group(self):
        """点数 < 3 的组直接跳过，不崩溃。"""
        runs = [
            {"C": 1e18, "N": 1e9, "L": 2.5},
            {"C": 1e18, "N": 2e9, "L": 2.6},
            # 只有 2 个点，不够二次拟合
            {"C": 1e20, "N": 1e10, "L": 2.3},
            {"C": 1e20, "N": 2e10, "L": 2.2},
            {"C": 1e20, "N": 3e10, "L": 2.1},
        ]
        result = isoflop_minima(runs)
        Cs = [r["C"] for r in result]
        assert 1e18 not in Cs, "点数不足的组不应出现在结果中"
        assert 1e20 in Cs

    def test_empty_runs(self):
        """空输入直接返回空列表，不崩溃。"""
        assert isoflop_minima([]) == []

    def test_monotone_group_uses_observed_endpoint(self):
        """近线性曲率的数值噪声不得把谷底外推到 inf。"""
        runs = [
            {"C": 1e18, "N": 1e6, "L": 2.2},
            {"C": 1e18, "N": 1e7, "L": 2.1},
            {"C": 1e18, "N": 1e8, "L": 2.0},
        ]
        result = isoflop_minima(runs)
        assert result == [{"C": 1e18, "N_opt": pytest.approx(1e8), "L_min": 2.0}]


# ════════════════════════════════════════════════════════════════════════════
# T7 — 端到端外推：小算力拟合 → 预测 C=1e19
# ════════════════════════════════════════════════════════════════════════════
class TestT7Extrapolation:
    def test_extrapolate_N_star(self):
        """只用 C ≤ 1e18 的合成数据拟合，外推 C=1e19 的 N*，误差 < 10%。"""
        # 生成更宽范围的小算力数据（确保 6ND <= 1e18）
        rng = np.random.default_rng(123)
        n = 60
        # 在 C = 6ND ≤ 1e18 区域采样
        log_C_vals = rng.uniform(math.log(1e15), math.log(1e18), n)
        frac = rng.uniform(0.1, 0.9, n)   # N 占 C/6 的比例（对数空间均匀）
        C_vals = np.exp(log_C_vals)
        N_arr = (C_vals / 6.0) ** frac
        D_arr = C_vals / (6.0 * N_arr)

        # 确保 N, D 在合理范围
        mask = (N_arr >= 1e6) & (D_arr >= 1e6) & (N_arr <= 1e12) & (D_arr <= 1e12)
        N_arr, D_arr, C_vals = N_arr[mask], D_arr[mask], C_vals[mask]

        L_arr = predict_loss(N_arr, D_arr, TRUE_PARAMS) * np.exp(
            rng.normal(0, NOISE_SIGMA, len(N_arr))
        )

        fitted = fit_scaling_law(N_arr, D_arr, L_arr, n_starts=32, seed=0)

        # 外推 C=1e19
        C_test = 1e19
        N_pred, _ = compute_optimal(C_test, fitted)
        N_true, _ = compute_optimal(C_test, TRUE_PARAMS)

        rel_err = abs(N_pred - N_true) / N_true
        assert rel_err < 0.10, (
            f"外推 N*={N_pred:.3e}，真值={N_true:.3e}，相对误差={rel_err:.4f}"
        )

    def test_extrapolate_loss(self):
        """用拟合参数预测 C=1e19 最优配置下的 L，误差 < 10%。"""
        rng = np.random.default_rng(456)
        n = 60
        log_C_vals = rng.uniform(math.log(1e15), math.log(1e18), n)
        frac = rng.uniform(0.1, 0.9, n)
        C_vals = np.exp(log_C_vals)
        N_arr = (C_vals / 6.0) ** frac
        D_arr = C_vals / (6.0 * N_arr)
        mask = (N_arr >= 1e6) & (D_arr >= 1e6) & (N_arr <= 1e12) & (D_arr <= 1e12)
        N_arr, D_arr = N_arr[mask], D_arr[mask]
        L_arr = predict_loss(N_arr, D_arr, TRUE_PARAMS) * np.exp(
            rng.normal(0, NOISE_SIGMA, len(N_arr))
        )

        fitted = fit_scaling_law(N_arr, D_arr, L_arr, n_starts=32, seed=0)

        C_test = 1e19
        N_pred, D_pred = compute_optimal(C_test, fitted)
        L_pred = predict_loss(N_pred, D_pred, fitted)

        N_true, D_true = compute_optimal(C_test, TRUE_PARAMS)
        L_true = predict_loss(N_true, D_true, TRUE_PARAMS)

        rel_err = abs(L_pred - L_true) / L_true
        assert rel_err < 0.10, (
            f"外推 L={L_pred:.4f}，真值={L_true:.4f}，相对误差={rel_err:.4f}"
        )
