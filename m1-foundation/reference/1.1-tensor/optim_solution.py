"""Phase 1.1 — 优化器参考答案（教师版）

SGD 含可选 momentum。

实现要点：
  - step() 遍历 params，对 requires_grad=True 且 grad 不为 None 的参数更新
  - momentum 版本：v_t = momentum * v_{t-1} + lr * grad_t；param -= v_t
    （这是 classical momentum，与 PyTorch SGD(momentum, dampening=0) 一致）
  - zero_grad() 调用每个 param 的 .zero_grad()
"""
from __future__ import annotations

# 注意：验证时 Tensor 来自参考答案模块
from minilm.tensor.tensor import Tensor


class SGD:
    """随机梯度下降（含可选 momentum）。"""

    def __init__(self, params: list[Tensor], lr: float, momentum: float = 0.0):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        # 为每个参数维护速度（用 id 作 key，避免 Tensor 不可哈希问题）
        self._velocities: dict[int, object] = {}   # id → np.ndarray

    def step(self) -> None:
        for p in self.params:
            if not p.requires_grad or p.grad is None:
                continue
            grad = p.grad
            if self.momentum == 0.0:
                # 普通 SGD
                p.data = p.data - self.lr * grad
            else:
                # momentum SGD
                pid = id(p)
                if pid not in self._velocities:
                    self._velocities[pid] = self.lr * grad
                else:
                    self._velocities[pid] = (
                        self.momentum * self._velocities[pid] + self.lr * grad
                    )
                p.data = p.data - self._velocities[pid]

    def zero_grad(self) -> None:
        for p in self.params:
            p.zero_grad()
