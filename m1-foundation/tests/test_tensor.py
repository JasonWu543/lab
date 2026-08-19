"""Phase 1.1 — Tensor 与 Autograd 验收测试（SPEC §5 T1–T9）。

测试先于实现存在：当前骨架下全红是预期状态。
只使用冻结接口（SPEC §4），不触碰实现内部属性。

Oracle 策略：同一随机数据分别构建 minilm Tensor 和 torch.tensor，
前向逐 op 对比、backward 后 grad 对比，容差 rtol=1e-8。
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from minilm.tensor.tensor import Tensor, no_grad
from minilm.tensor import nn as mnn
from minilm.tensor.optim import SGD

# ---------- 工具函数 ----------

RNG = np.random.default_rng(0)


def rand(*shape) -> np.ndarray:
    """固定 rng 的随机数组，值域 (0, 2) 避免 log/div 的数值问题。"""
    return RNG.random(shape).astype(np.float64) + 0.5


def allclose(a: np.ndarray, b: np.ndarray, rtol: float = 1e-8) -> bool:
    return np.allclose(a, b, rtol=rtol, atol=1e-10)


def t2np(t: torch.Tensor) -> np.ndarray:
    return t.detach().numpy().astype(np.float64)


def make_pair(data: np.ndarray, requires_grad: bool = True):
    """同一数据建 minilm Tensor 和 torch tensor（float64）。"""
    mt = Tensor(data.copy(), requires_grad=requires_grad)
    tt = torch.tensor(data.copy(), dtype=torch.float64, requires_grad=requires_grad)
    return mt, tt


# ============================================================
# T1 — 前向对齐
# ============================================================

class TestT1Forward:
    """随机 shape 下所有 op 的前向结果与 PyTorch 一致（rtol 1e-8）。"""

    def test_add_broadcast(self):
        a_np = rand(3, 1)
        b_np = rand(1, 4)
        ma, ta = make_pair(a_np)
        mb, tb = make_pair(b_np)
        mr = ma + mb
        tr = ta + tb
        assert allclose(mr.data, t2np(tr))

    def test_sub(self):
        a_np, b_np = rand(4), rand(4)
        ma, ta = make_pair(a_np)
        mb, tb = make_pair(b_np)
        assert allclose((ma - mb).data, t2np(ta - tb))

    def test_mul(self):
        a_np, b_np = rand(3, 4), rand(3, 4)
        ma, ta = make_pair(a_np)
        mb, tb = make_pair(b_np)
        assert allclose((ma * mb).data, t2np(ta * tb))

    def test_div(self):
        a_np, b_np = rand(3, 4), rand(3, 4)
        ma, ta = make_pair(a_np)
        mb, tb = make_pair(b_np)
        assert allclose((ma / mb).data, t2np(ta / tb))

    def test_neg(self):
        a_np = rand(3, 3)
        ma, ta = make_pair(a_np)
        assert allclose((-ma).data, t2np(-ta))

    def test_pow(self):
        a_np = rand(4, 4)
        ma, ta = make_pair(a_np)
        assert allclose((ma ** 2.5).data, t2np(ta ** 2.5))

    def test_matmul(self):
        a_np = rand(4, 5)
        b_np = rand(5, 6)
        ma, ta = make_pair(a_np)
        mb, tb = make_pair(b_np)
        assert allclose((ma @ mb).data, t2np(ta @ tb))

    def test_exp(self):
        a_np = rand(3, 4) - 0.5   # 值域 (0,1.5) → exp 不溢出
        ma, ta = make_pair(a_np)
        assert allclose(ma.exp().data, t2np(ta.exp()))

    def test_log(self):
        a_np = rand(3, 4)
        ma, ta = make_pair(a_np)
        assert allclose(ma.log().data, t2np(ta.log()))

    def test_sigmoid(self):
        a_np = rand(3, 4) - 0.5
        ma, ta = make_pair(a_np)
        assert allclose(ma.sigmoid().data, t2np(ta.sigmoid()))

    def test_maximum_scalar(self):
        a_np = rand(3, 4) - 0.8   # 含负值
        ma, ta = make_pair(a_np)
        assert allclose(ma.maximum(0).data, t2np(torch.clamp(ta, min=0)))

    def test_sum_all(self):
        a_np = rand(3, 4)
        ma, ta = make_pair(a_np)
        assert allclose(ma.sum().data, t2np(ta.sum()))

    def test_sum_dim_keepdim(self):
        a_np = rand(3, 4)
        ma, ta = make_pair(a_np)
        assert allclose(ma.sum(dim=1, keepdim=True).data, t2np(ta.sum(dim=1, keepdim=True)))

    def test_mean_dim(self):
        a_np = rand(3, 4)
        ma, ta = make_pair(a_np)
        assert allclose(ma.mean(dim=0).data, t2np(ta.mean(dim=0)))

    def test_reshape(self):
        a_np = rand(3, 4)
        ma, ta = make_pair(a_np)
        assert allclose(ma.reshape(2, 6).data, t2np(ta.reshape(2, 6)))

    def test_transpose(self):
        a_np = rand(3, 4)
        ma, ta = make_pair(a_np)
        assert allclose(ma.transpose(0, 1).data, t2np(ta.transpose(0, 1)))

    def test_getitem(self):
        a_np = rand(4, 4)
        ma, ta = make_pair(a_np)
        assert allclose(ma[1:3, :].data, t2np(ta[1:3, :]))

    def test_scalar_add(self):
        a_np = rand(3, 3)
        ma, ta = make_pair(a_np)
        assert allclose((ma + 2.0).data, t2np(ta + 2.0))
        assert allclose((2.0 + ma).data, t2np(2.0 + ta))

    def test_scalar_mul(self):
        a_np = rand(3, 3)
        ma, ta = make_pair(a_np)
        assert allclose((ma * 3.0).data, t2np(ta * 3.0))


# ============================================================
# T2 — 反向对齐
# ============================================================

class TestT2Backward:
    """同一计算图，minilm backward 后 .grad 与 PyTorch 一致（rtol 1e-8）。"""

    def _check_grads(self, m_pairs, t_pairs, m_result, t_result):
        """触发 backward 并逐参数检查梯度。"""
        m_result.backward()
        t_result.backward()
        for (mt, tt) in zip(m_pairs, t_pairs):
            assert allclose(mt.grad, t2np(tt.grad)), \
                f"grad mismatch:\nminilm={mt.grad}\ntorch={t2np(tt.grad)}"

    def test_add_backward(self):
        a = rand(3)
        ma, ta = make_pair(a)
        mb, tb = make_pair(rand(3))
        self._check_grads([ma, mb], [ta, tb], (ma + mb).mean(), (ta + tb).mean())

    def test_mul_backward(self):
        a_np, b_np = rand(3, 4), rand(3, 4)
        ma, ta = make_pair(a_np)
        mb, tb = make_pair(b_np)
        self._check_grads([ma, mb], [ta, tb], (ma * mb).sum(), (ta * tb).sum())

    def test_pow_backward(self):
        a_np = rand(4)
        ma, ta = make_pair(a_np)
        self._check_grads([ma], [ta], (ma ** 3).sum(), (ta ** 3).sum())

    def test_matmul_backward(self):
        a_np = rand(4, 5)
        b_np = rand(5, 6)
        ma, ta = make_pair(a_np)
        mb, tb = make_pair(b_np)
        r_m = (ma @ mb).sum()
        r_t = (ta @ tb).sum()
        self._check_grads([ma, mb], [ta, tb], r_m, r_t)

    def test_exp_log_backward(self):
        a_np = rand(4)
        ma, ta = make_pair(a_np)
        self._check_grads([ma], [ta], ma.exp().log().sum(), ta.exp().log().sum())

    def test_sigmoid_backward(self):
        a_np = rand(4) - 0.5
        ma, ta = make_pair(a_np)
        self._check_grads([ma], [ta], ma.sigmoid().sum(), ta.sigmoid().sum())

    def test_sum_keepdim_backward(self):
        a_np = rand(3, 4)
        ma, ta = make_pair(a_np)
        self._check_grads([ma], [ta],
                          ma.sum(dim=1, keepdim=True).sum(),
                          ta.sum(dim=1, keepdim=True).sum())

    def test_mean_backward(self):
        a_np = rand(3, 4)
        ma, ta = make_pair(a_np)
        self._check_grads([ma], [ta], ma.mean(), ta.mean())

    def test_chain_backward(self):
        """链式：((a * b).exp() + c.sigmoid()).mean()"""
        a_np, b_np, c_np = rand(4), rand(4), rand(4)
        ma, ta = make_pair(a_np)
        mb, tb = make_pair(b_np)
        mc, tc = make_pair(c_np)
        r_m = ((ma * mb).exp() + mc.sigmoid()).mean()
        r_t = ((ta * tb).exp() + tc.sigmoid()).mean()
        self._check_grads([ma, mb, mc], [ta, tb, tc], r_m, r_t)


# ============================================================
# T3 — 有限差分 gradcheck
# ============================================================

class TestT3Gradcheck:
    """中心差分数值梯度 vs 解析梯度，eps=1e-6，rtol=1e-4。

    复合表达式：f = (a @ b + a.sigmoid() * c).mean()
    """

    def _numerical_grad(self, f, x_np: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """对 x_np 的每个元素做中心差分。"""
        grad = np.zeros_like(x_np)
        it = np.nditer(x_np, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            orig = x_np[idx]
            x_np[idx] = orig + eps
            fp = f(x_np)
            x_np[idx] = orig - eps
            fm = f(x_np)
            x_np[idx] = orig
            grad[idx] = (fp - fm) / (2 * eps)
            it.iternext()
        return grad

    def test_composite_expr(self):
        # f = (a @ b + sigmoid(a @ b) * c).mean()
        # a: (4,3), b: (3,2) → a@b: (4,2), c: (4,2)  → shapes align
        a_np = rand(4, 3)
        b_np = rand(3, 2)
        c_np = rand(4, 2)

        def f_scalar(a_data):
            """返回标量值（numpy float）。"""
            a = Tensor(a_data.copy(), requires_grad=False)
            b = Tensor(b_np.copy(), requires_grad=False)
            c = Tensor(c_np.copy(), requires_grad=False)
            ab = a @ b
            return float((ab + ab.sigmoid() * c).mean().data)

        # 数值梯度
        num_grad = self._numerical_grad(f_scalar, a_np.copy())

        # 解析梯度
        ma = Tensor(a_np.copy(), requires_grad=True)
        mb = Tensor(b_np.copy(), requires_grad=True)
        mc = Tensor(c_np.copy(), requires_grad=True)
        mab = ma @ mb
        r = (mab + mab.sigmoid() * mc).mean()
        r.backward()

        assert np.allclose(ma.grad, num_grad, rtol=1e-4, atol=1e-8), \
            f"gradcheck failed:\nanalytic={ma.grad}\nnumeric={num_grad}"


# ============================================================
# T4 — broadcast 梯度
# ============================================================

class TestT4Broadcast:
    """broadcast 梯度的 shape 和数值都正确。"""

    def test_broadcast_3x1_plus_1x4(self):
        a_np = rand(3, 1)
        b_np = rand(1, 4)
        ma, ta = make_pair(a_np)
        mb, tb = make_pair(b_np)
        r_m = (ma + mb).sum()
        r_t = (ta + tb).sum()
        r_m.backward()
        r_t.backward()
        assert ma.grad.shape == (3, 1), f"wrong shape {ma.grad.shape}"
        assert mb.grad.shape == (1, 4), f"wrong shape {mb.grad.shape}"
        assert allclose(ma.grad, t2np(ta.grad))
        assert allclose(mb.grad, t2np(tb.grad))

    def test_scalar_plus_matrix(self):
        a_np = rand(1)[0]          # 标量
        b_np = rand(3, 4)
        ma = Tensor(float(a_np), requires_grad=True)
        ta = torch.tensor(float(a_np), dtype=torch.float64, requires_grad=True)
        mb, tb = make_pair(b_np)
        r_m = (ma + mb).sum()
        r_t = (ta + tb).sum()
        r_m.backward()
        r_t.backward()
        assert allclose(np.array(ma.grad), t2np(ta.grad))
        assert allclose(mb.grad, t2np(tb.grad))

    def test_mul_broadcast(self):
        a_np = rand(4, 1)
        b_np = rand(1, 5)
        ma, ta = make_pair(a_np)
        mb, tb = make_pair(b_np)
        r_m = (ma * mb).mean()
        r_t = (ta * tb).mean()
        r_m.backward()
        r_t.backward()
        assert allclose(ma.grad, t2np(ta.grad))
        assert allclose(mb.grad, t2np(tb.grad))

    def test_keepdim_sum_backward(self):
        """sum(dim=1, keepdim=True) 的梯度 shape 应保留该维度。"""
        a_np = rand(3, 4)
        ma, ta = make_pair(a_np)
        r_m = ma.sum(dim=1, keepdim=True).sum()
        r_t = ta.sum(dim=1, keepdim=True).sum()
        r_m.backward()
        r_t.backward()
        assert ma.grad.shape == (3, 4)
        assert allclose(ma.grad, t2np(ta.grad))


# ============================================================
# T5 — 梯度累积
# ============================================================

class TestT5GradAccum:
    """同一 leaf 进入多条路径，梯度自动累积。"""

    def test_multi_path(self):
        a_np = rand(3)
        ma = Tensor(a_np.copy(), requires_grad=True)
        # a 同时出现在两条路径
        r = (ma * ma + ma).sum()
        r.backward()
        grad_once = ma.grad.copy()

        # 不 zero_grad，新建相同结构图再 backward → 梯度叠加（翻倍）
        # 注意：必须重新建图，旧图重复 backward 会因图中节点 grad 已填充产生 3x 效应
        ma2 = Tensor(a_np.copy(), requires_grad=True)
        r2a = (ma2 * ma2 + ma2).sum()
        r2a.backward()
        # 再建一次图（共享 ma2 leaf），不 zero_grad 直接 backward
        r2b = (ma2 * ma2 + ma2).sum()
        r2b.backward()
        assert allclose(ma2.grad, 2 * grad_once), \
            "两次 backward（不 zero_grad）应使梯度翻倍"

    def test_zero_grad_clears(self):
        a_np = rand(3)
        ma = Tensor(a_np.copy(), requires_grad=True)
        r = ma.sum()
        r.backward()
        ma.zero_grad()
        assert ma.grad is None or np.all(ma.grad == 0), "zero_grad 后 grad 应为 None 或 0"

    def test_diamond_graph(self):
        """菱形图：a → b, a → c, b+c → d。a 的梯度应是两条路径的和。"""
        a_np = rand(4)
        ma = Tensor(a_np.copy(), requires_grad=True)
        ta = torch.tensor(a_np.copy(), dtype=torch.float64, requires_grad=True)
        b_m = ma * 2.0
        c_m = ma ** 2
        d_m = (b_m + c_m).sum()
        b_t = ta * 2.0
        c_t = ta ** 2
        d_t = (b_t + c_t).sum()
        d_m.backward()
        d_t.backward()
        assert allclose(ma.grad, t2np(ta.grad))


# ============================================================
# T6 — 非连续（transpose 后运算）
# ============================================================

class TestT6NonContiguous:
    """transpose 后参与 matmul/加法，前后向均正确。"""

    def test_transpose_matmul(self):
        a_np = rand(4, 5)
        b_np = rand(4, 6)
        ma, ta = make_pair(a_np)
        mb, tb = make_pair(b_np)
        # a.T @ b  → shape (5, 6) 不需要 contiguous
        r_m = (ma.transpose(0, 1) @ mb).sum()
        r_t = (ta.transpose(0, 1) @ tb).sum()
        r_m.backward()
        r_t.backward()
        assert allclose(ma.grad, t2np(ta.grad))
        assert allclose(mb.grad, t2np(tb.grad))

    def test_transpose_add(self):
        # 两个 (3,4) 矩阵各自 transpose 后相加，结果 shape (4,3)
        a_np = rand(3, 4)
        b_np = rand(3, 4)
        ma, ta = make_pair(a_np)
        mb, tb = make_pair(b_np)
        r_m = (ma.transpose(0, 1) + mb.transpose(0, 1)).sum()
        r_t = (ta.transpose(0, 1) + tb.transpose(0, 1)).sum()
        r_m.backward()
        r_t.backward()
        assert allclose(ma.grad, t2np(ta.grad))
        assert allclose(mb.grad, t2np(tb.grad))


# ============================================================
# T7 — 图与 no_grad
# ============================================================

class TestT7Graph:
    """no_grad / detach / 非标量错误处理。"""

    def test_no_grad_no_graph(self):
        a_np = rand(3)
        with no_grad():
            ma = Tensor(a_np, requires_grad=True)
            mb = Tensor(rand(3), requires_grad=True)
            r = ma + mb
        # no_grad 下结果不应有梯度
        assert not r.requires_grad, "no_grad 下结果 requires_grad 应为 False"

    def test_no_grad_no_backward(self):
        a_np = rand(3)
        with no_grad():
            ma = Tensor(a_np, requires_grad=True)
            r = ma.sum()
        # 在 no_grad 作用域外调用 backward 应该失败或者无梯度
        # （实现约定：no_grad 下的输出 requires_grad=False）
        assert not r.requires_grad

    def test_detach_cuts_grad(self):
        a_np = rand(3)
        ma = Tensor(a_np.copy(), requires_grad=True)
        b = ma * 2.0
        b_det = b.detach()
        c = b_det * 3.0
        # c 的计算图不连接到 ma，c.sum().backward() 后 ma.grad 应为 None
        c.sum().backward()
        assert ma.grad is None, "detach 后梯度不应传回原 leaf"

    def test_nonscalar_backward_raises(self):
        a_np = rand(3)
        ma = Tensor(a_np, requires_grad=True)
        r = ma * 2.0     # 非标量
        with pytest.raises((RuntimeError, ValueError)):
            r.backward()  # 不传 grad 应报错

    def test_nonscalar_backward_with_grad(self):
        """非标量传入 grad 时可以正常工作。"""
        a_np = rand(3)
        ma = Tensor(a_np.copy(), requires_grad=True)
        ta = torch.tensor(a_np.copy(), dtype=torch.float64, requires_grad=True)
        r_m = ma * 2.0
        r_t = ta * 2.0
        upstream = np.ones(3)
        r_m.backward(grad=upstream)
        r_t.backward(torch.ones(3, dtype=torch.float64))
        assert allclose(ma.grad, t2np(ta.grad))


# ============================================================
# T8 — nn 层对齐
# ============================================================

class TestT8NN:
    """Linear / RMSNorm / SwiGLU 前反向与 PyTorch 等价实现对齐。"""

    def _linear_torch(self, weight_np, bias_np, x_np):
        """用 torch 做等价 Linear 前反向。"""
        W = torch.tensor(weight_np, dtype=torch.float64, requires_grad=True)
        b = torch.tensor(bias_np, dtype=torch.float64, requires_grad=True)
        x = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)
        r = x @ W.T + b
        r.sum().backward()
        return r.detach().numpy(), W.grad.numpy(), b.grad.numpy(), x.grad.numpy()

    def test_linear_forward_backward(self):
        in_f, out_f = 5, 3
        batch = 4
        layer = mnn.Linear(in_f, out_f, bias=True)
        # 用 layer 内部的 weight/bias 作为 oracle 输入
        w_np = layer.weight.data.copy()
        b_np = layer.bias.data.copy()
        x_np = rand(batch, in_f)

        # minilm
        x_m = Tensor(x_np.copy(), requires_grad=True)
        r_m = layer(x_m)
        r_m.sum().backward()

        # torch oracle
        r_ref, dW_ref, db_ref, dx_ref = self._linear_torch(w_np, b_np, x_np)

        assert allclose(r_m.data, r_ref), "Linear forward mismatch"
        assert allclose(layer.weight.grad, dW_ref), "Linear weight grad mismatch"
        assert allclose(layer.bias.grad, db_ref), "Linear bias grad mismatch"
        assert allclose(x_m.grad, dx_ref), "Linear input grad mismatch"

    def _rmsnorm_torch(self, weight_np, x_np, eps=1e-6):
        """手写 torch 版 RMSNorm。"""
        x = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)
        w = torch.tensor(weight_np, dtype=torch.float64, requires_grad=True)
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(eps).sqrt()
        r = x / rms * w
        r.sum().backward()
        return r.detach().numpy(), w.grad.numpy(), x.grad.numpy()

    def test_rmsnorm_forward_backward(self):
        dim = 6
        batch = 4
        norm = mnn.RMSNorm(dim)
        w_np = norm.weight.data.copy()
        x_np = rand(batch, dim) - 0.5

        x_m = Tensor(x_np.copy(), requires_grad=True)
        r_m = norm(x_m)
        r_m.sum().backward()

        r_ref, dw_ref, dx_ref = self._rmsnorm_torch(w_np, x_np)
        assert allclose(r_m.data, r_ref, rtol=1e-7), "RMSNorm forward mismatch"
        assert allclose(norm.weight.grad, dw_ref, rtol=1e-7), "RMSNorm weight grad"
        assert allclose(x_m.grad, dx_ref, rtol=1e-7), "RMSNorm input grad"

    def _swiglu_torch(self, w1_np, w2_np, w3_np, x_np):
        """手写 torch 版 SwiGLU：output = W2(silu(W1 x) * (W3 x))，无 bias。"""
        W1 = torch.tensor(w1_np, dtype=torch.float64, requires_grad=True)
        W2 = torch.tensor(w2_np, dtype=torch.float64, requires_grad=True)
        W3 = torch.tensor(w3_np, dtype=torch.float64, requires_grad=True)
        x = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)
        gate = torch.nn.functional.silu(x @ W1.T)
        value = x @ W3.T
        hidden = gate * value
        r = hidden @ W2.T
        r.sum().backward()
        return r.detach().numpy(), W1.grad.numpy(), W2.grad.numpy(), W3.grad.numpy(), x.grad.numpy()

    def test_swiglu_forward_backward(self):
        dim, hidden = 4, 6
        batch = 3
        swiglu = mnn.SwiGLU(dim, hidden)
        w1_np = swiglu.w1.weight.data.copy()
        w2_np = swiglu.w2.weight.data.copy()
        w3_np = swiglu.w3.weight.data.copy()
        x_np = rand(batch, dim) - 0.5

        x_m = Tensor(x_np.copy(), requires_grad=True)
        r_m = swiglu(x_m)
        r_m.sum().backward()

        r_ref, dw1, dw2, dw3, dx_ref = self._swiglu_torch(w1_np, w2_np, w3_np, x_np)
        assert allclose(r_m.data, r_ref, rtol=1e-7), "SwiGLU forward mismatch"
        assert allclose(swiglu.w1.weight.grad, dw1, rtol=1e-7), "SwiGLU W1 grad"
        assert allclose(swiglu.w2.weight.grad, dw2, rtol=1e-7), "SwiGLU W2 grad"
        assert allclose(swiglu.w3.weight.grad, dw3, rtol=1e-7), "SwiGLU W3 grad"
        assert allclose(x_m.grad, dx_ref, rtol=1e-7), "SwiGLU input grad"

    def test_parameters_collects_all(self):
        """Module.parameters() 应递归收集所有 Tensor。"""
        swiglu = mnn.SwiGLU(4, 6)
        params = swiglu.parameters()
        # w1/w2/w3 各一个 weight，共 3 个
        assert len(params) == 3, f"SwiGLU 应有 3 个参数，得到 {len(params)}"


# ============================================================
# T9 — MLP 收敛
# ============================================================

class TestT9Convergence:
    """两层 MLP + SGD 在合成二分类数据上 500 步 loss < 0.05。"""

    @staticmethod
    def make_data(n=64, seed=42):
        rng = np.random.default_rng(seed)
        # 两个高斯团，中心分别是 (1,1) 和 (-1,-1)
        x0 = rng.normal(loc=[1.0, 1.0], scale=0.4, size=(n // 2, 2))
        x1 = rng.normal(loc=[-1.0, -1.0], scale=0.4, size=(n // 2, 2))
        X = np.vstack([x0, x1]).astype(np.float64)
        y = np.array([1.0] * (n // 2) + [0.0] * (n // 2), dtype=np.float64)
        return X, y

    def _make_mlp(self):
        """两层 MLP：Linear(2→8) → SwiGLU(8,16) → Linear(8→1)。

        注意：SwiGLU 输出维度 = dim（= 8），所以最后一层输入也是 8。
        """
        class MLP(mnn.Module):
            def __init__(self):
                self.l1 = mnn.Linear(2, 8)
                self.act = mnn.SwiGLU(8, 16)
                self.l2 = mnn.Linear(8, 1)

            def forward(self, x):
                return self.l2(self.act(self.l1(x)))

            def parameters(self):
                return self.l1.parameters() + self.act.parameters() + self.l2.parameters()

        return MLP()

    def test_convergence(self):
        X, y = self.make_data()
        model = self._make_mlp()
        opt = SGD(model.parameters(), lr=0.05)

        final_loss = None
        y_col = y.reshape(-1, 1)
        for step in range(500):
            x_t = Tensor(X, requires_grad=False)
            y_t = Tensor(y_col.copy(), requires_grad=False)

            logits = model(x_t)              # shape (64, 1)
            # MSE loss：简单，梯度清晰，收敛稳定
            diff = logits - y_t
            loss = (diff * diff).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()
            final_loss = float(loss.data)

        assert final_loss < 0.05, f"500 步后 loss={final_loss:.4f}，未收敛到 < 0.05"

    def test_convergence_relu_mlp(self):
        """备用测试：Linear-ReLU(maximum(0)) 版 MLP，更简单，应更快收敛。"""
        X, y = self.make_data()

        class ReluMLP(mnn.Module):
            def __init__(self):
                self.l1 = mnn.Linear(2, 16)
                self.l2 = mnn.Linear(16, 1)

            def forward(self, x):
                return self.l2(self.l1(x).maximum(0))

            def parameters(self):
                return self.l1.parameters() + self.l2.parameters()

        model = ReluMLP()
        opt = SGD(model.parameters(), lr=0.1)
        y_col = y.reshape(-1, 1)

        final_loss = None
        for _ in range(500):
            x_t = Tensor(X, requires_grad=False)
            y_t = Tensor(y_col.copy(), requires_grad=False)
            logits = model(x_t)
            diff = logits - y_t
            loss = (diff * diff).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            final_loss = float(loss.data)

        assert final_loss < 0.05, f"ReLU MLP 未收敛，loss={final_loss:.4f}"
