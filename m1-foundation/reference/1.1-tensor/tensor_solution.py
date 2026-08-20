"""Phase 1.1 — Tensor 参考答案（教师版）

关键梯度公式推导（以 2D 为主，括号内为批量扩展时的规则）：

  add:      dA = upstream,    dB = upstream（broadcast 后归约到原 shape）
  sub:      dA = upstream,    dB = -upstream
  mul:      dA = B * upstream, dB = A * upstream
  div:      dA = upstream / B, dB = -A / B^2 * upstream
  neg:      dA = -upstream
  pow:      dA = n * A^(n-1) * upstream
  matmul:   dA = upstream @ B.T,  dB = A.T @ upstream
            推导：L = f(C), C = A @ B
            dL/dA_ij = sum_k dL/dC_ik * dC_ik/dA_ij = sum_k dL/dC_ik * B_jk
                     = (dC @ B.T)_ij   ✓
            dL/dB_jk = sum_i dL/dC_ik * A_ij = (A.T @ dC)_jk  ✓
  exp:      dA = exp(A) * upstream  （即 result.data * upstream）
  log:      dA = upstream / A
  sigmoid:  dA = σ(A)*(1-σ(A)) * upstream  （= result.data*(1-result.data)*upstream）
  maximum:  dA = (A > scalar) * upstream   （边界取 0）
  sum(dim): dA = upstream broadcast 回 self.shape
            若 keepdim=False：先对 upstream 在 dim 方向 expand_dims，再 broadcast
  mean(dim):dA = upstream / n，其中 n = self.data.shape[dim] 或 self.data.size（全局）
  reshape:  dA = upstream.reshape(self.shape)
  transpose:dA = np.swapaxes(upstream, dim0, dim1)   （交换是自逆的）
  getitem:  dA 全 0，dA[idx] += upstream（np.add.at 处理重复索引）
"""
from __future__ import annotations

from contextlib import contextmanager

import numpy as np

# ---------- 全局 no_grad 标志 ----------

_NO_GRAD_MODE = False


@contextmanager
def no_grad():
    """上下文管理器：禁止建计算图。"""
    global _NO_GRAD_MODE
    prev = _NO_GRAD_MODE
    _NO_GRAD_MODE = True
    try:
        yield
    finally:
        _NO_GRAD_MODE = prev


# ---------- broadcast 梯度归约工具 ----------

def _reduce_to(grad: np.ndarray, target_shape: tuple) -> np.ndarray:
    """将 grad 归约（sum）到 target_shape，处理 broadcast 反向传播。

    逻辑：
      1. 如果 target_shape 维数更少（被前缀 broadcast），先对多余的前缀 dim sum
      2. 再对值为 1 的维度 sum（keepdim=True），最后 reshape
    """
    # 处理标量 target（shape == () 或 shape == (1,) 边界情况）
    out = grad
    # 补齐前缀维度差
    if out.ndim > len(target_shape):
        n_extra = out.ndim - len(target_shape)
        out = out.sum(axis=tuple(range(n_extra)))
    # 对原本为 1 的维度归约（keepdim，保留维度数）
    for i, (s, g) in enumerate(zip(target_shape, out.shape)):
        if s == 1 and g != 1:
            out = out.sum(axis=i, keepdims=True)
    return out.reshape(target_shape)


# ---------- Tensor ----------

class Tensor:
    """numpy 底层，支持 reverse-mode autograd 的 Tensor。"""

    def __init__(self, data, requires_grad: bool = False):
        if isinstance(data, np.ndarray):
            # Preserve numpy view semantics when the input is already float64.
            self.data = data.astype(np.float64, copy=False)
        else:
            self.data = np.array(data, dtype=np.float64)
        self.requires_grad = requires_grad
        self.grad: np.ndarray | None = None
        # 计算图内部状态（普通用户不应访问）
        self._parents: set[Tensor] = set()
        self._backward = lambda: None   # 默认无操作

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    # ------ 梯度辅助 ------

    def _accum_grad(self, g: np.ndarray) -> None:
        """将梯度 g 累加到 self.grad（保证 shape 对齐）。"""
        g = _reduce_to(g, self.shape)
        if self.grad is None:
            self.grad = g.copy()
        else:
            self.grad = self.grad + g

    # ------ 图操作 ------

    def backward(self, grad: np.ndarray | None = None) -> None:
        if not self.requires_grad:
            raise RuntimeError("不能从 requires_grad=False 的 Tensor 发起 backward()")
        if grad is None:
            if self.data.shape == () or self.data.size == 1:
                grad = np.ones_like(self.data)
            else:
                raise RuntimeError(
                    "非标量 Tensor 调用 backward() 必须传入 grad 参数（与 PyTorch 行为一致）"
                )
        # 拓扑排序（后序 DFS）
        topo: list[Tensor] = []
        visited: set[int] = set()

        def dfs(node: Tensor):
            if id(node) in visited:
                return
            visited.add(id(node))
            for p in node._parents:
                dfs(p)
            topo.append(node)

        dfs(self)
        # A second backward through the same graph must not reuse stale
        # intermediate gradients.  Leaf gradients, however, accumulate.
        for node in topo:
            if node._parents:
                node.grad = None
        if self._parents:
            self.grad = np.asarray(grad, dtype=np.float64)
        else:
            self._accum_grad(np.asarray(grad, dtype=np.float64))
        # 从输出节点向输入节点传播
        for node in reversed(topo):
            node._backward()

    def zero_grad(self) -> None:
        self.grad = None

    def detach(self) -> "Tensor":
        """共享 data，切断梯度图。"""
        t = Tensor.__new__(Tensor)
        t.data = self.data          # 共享存储
        t.requires_grad = False
        t.grad = None
        t._parents = set()
        t._backward = lambda: None
        return t

    # ------ 内部：创建结果 Tensor 并注册 backward ------

    def _make_result(self, data: np.ndarray, parents: set["Tensor"]) -> "Tensor":
        """工厂：创建结果 Tensor，根据 no_grad 模式决定是否建图。"""
        req_grad = (not _NO_GRAD_MODE) and any(p.requires_grad for p in parents)
        result = Tensor(data, requires_grad=req_grad)
        if req_grad:
            result._parents = parents
        return result

    # ------ 逐元素 ops ------

    def __add__(self, other) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        result = self._make_result(self.data + other.data, {self, other})

        def _bwd():
            if self.requires_grad:
                self._accum_grad(result.grad)
            if other.requires_grad:
                other._accum_grad(result.grad)

        result._backward = _bwd
        return result

    def __radd__(self, other) -> "Tensor":
        return self.__add__(other)

    def __sub__(self, other) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        result = self._make_result(self.data - other.data, {self, other})

        def _bwd():
            if self.requires_grad:
                self._accum_grad(result.grad)
            if other.requires_grad:
                other._accum_grad(-result.grad)

        result._backward = _bwd
        return result

    def __rsub__(self, other) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other.__sub__(self)

    def __mul__(self, other) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        # 保存前向值（用于梯度计算）
        self_data = self.data
        other_data = other.data
        result = self._make_result(self_data * other_data, {self, other})

        def _bwd():
            if self.requires_grad:
                self._accum_grad(other_data * result.grad)
            if other.requires_grad:
                other._accum_grad(self_data * result.grad)

        result._backward = _bwd
        return result

    def __rmul__(self, other) -> "Tensor":
        return self.__mul__(other)

    def __truediv__(self, other) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        self_data = self.data
        other_data = other.data
        result = self._make_result(self_data / other_data, {self, other})

        def _bwd():
            if self.requires_grad:
                # dL/dA = upstream / B
                self._accum_grad(result.grad / other_data)
            if other.requires_grad:
                # dL/dB = -A / B^2 * upstream
                other._accum_grad(-self_data / (other_data ** 2) * result.grad)

        result._backward = _bwd
        return result

    def __rtruediv__(self, other) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other.__truediv__(self)

    def __neg__(self) -> "Tensor":
        result = self._make_result(-self.data, {self})

        def _bwd():
            if self.requires_grad:
                self._accum_grad(-result.grad)

        result._backward = _bwd
        return result

    def __pow__(self, exponent) -> "Tensor":
        # exponent 只支持标量
        self_data = self.data
        result = self._make_result(self_data ** exponent, {self})

        def _bwd():
            if self.requires_grad:
                # d/dx x^n = n * x^(n-1)
                self._accum_grad(exponent * self_data ** (exponent - 1) * result.grad)

        result._backward = _bwd
        return result

    def __matmul__(self, other) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        self_data = self.data
        other_data = other.data
        result = self._make_result(self_data @ other_data, {self, other})

        def _bwd():
            # dA = dC @ B.T,   dB = A.T @ dC
            if self.requires_grad:
                self._accum_grad(result.grad @ other_data.swapaxes(-1, -2))
            if other.requires_grad:
                other._accum_grad(self_data.swapaxes(-1, -2) @ result.grad)

        result._backward = _bwd
        return result

    def __getitem__(self, idx) -> "Tensor":
        result = self._make_result(self.data[idx], {self})

        def _bwd():
            if self.requires_grad:
                grad_self = np.zeros_like(self.data)
                # np.add.at 处理重复索引（如 fancy indexing）
                np.add.at(grad_self, idx, result.grad)
                self._accum_grad(grad_self)

        result._backward = _bwd
        return result

    # ------ 数学函数 ------

    def exp(self) -> "Tensor":
        out = np.exp(self.data)
        result = self._make_result(out, {self})

        def _bwd():
            if self.requires_grad:
                # d/dx exp(x) = exp(x)
                self._accum_grad(out * result.grad)

        result._backward = _bwd
        return result

    def log(self) -> "Tensor":
        self_data = self.data
        result = self._make_result(np.log(self_data), {self})

        def _bwd():
            if self.requires_grad:
                # d/dx log(x) = 1/x
                self._accum_grad(result.grad / self_data)

        result._backward = _bwd
        return result

    def sigmoid(self) -> "Tensor":
        out = 1.0 / (1.0 + np.exp(-self.data))
        result = self._make_result(out, {self})

        def _bwd():
            if self.requires_grad:
                # dσ/dx = σ(x)*(1-σ(x))
                self._accum_grad(out * (1.0 - out) * result.grad)

        result._backward = _bwd
        return result

    def maximum(self, scalar) -> "Tensor":
        mask = (self.data > scalar).astype(np.float64)
        result = self._make_result(np.maximum(self.data, scalar), {self})

        def _bwd():
            if self.requires_grad:
                self._accum_grad(mask * result.grad)

        result._backward = _bwd
        return result

    def sum(self, dim=None, keepdim: bool = False) -> "Tensor":
        out = np.sum(self.data, axis=dim, keepdims=keepdim)
        result = self._make_result(out, {self})
        self_shape = self.data.shape

        def _bwd():
            if self.requires_grad:
                g = result.grad
                # 若 keepdim=False 且 dim 不为 None，需在 dim 方向恢复维度再 broadcast
                if dim is not None and not keepdim:
                    g = np.expand_dims(g, axis=dim)
                # broadcast 到 self.shape
                self._accum_grad(np.broadcast_to(g, self_shape).copy())

        result._backward = _bwd
        return result

    def mean(self, dim=None, keepdim: bool = False) -> "Tensor":
        out = np.mean(self.data, axis=dim, keepdims=keepdim)
        result = self._make_result(out, {self})
        self_shape = self.data.shape
        # 归约的元素数
        if dim is None:
            n = self.data.size
        else:
            n = self.data.shape[dim]

        def _bwd():
            if self.requires_grad:
                g = result.grad / n
                if dim is not None and not keepdim:
                    g = np.expand_dims(g, axis=dim)
                self._accum_grad(np.broadcast_to(g, self_shape).copy())

        result._backward = _bwd
        return result

    def reshape(self, *shape) -> "Tensor":
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        self_shape = self.data.shape
        result = self._make_result(self.data.reshape(shape), {self})

        def _bwd():
            if self.requires_grad:
                self._accum_grad(result.grad.reshape(self_shape))

        result._backward = _bwd
        return result

    def transpose(self, dim0: int, dim1: int) -> "Tensor":
        result = self._make_result(np.swapaxes(self.data, dim0, dim1), {self})

        def _bwd():
            if self.requires_grad:
                # 交换是自逆的：再交换一次还原
                self._accum_grad(np.swapaxes(result.grad, dim0, dim1))

        result._backward = _bwd
        return result

    def __repr__(self) -> str:
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"
