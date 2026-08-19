"""Phase 1.1 — 优化器骨架（学生实现文件）

实现 SGD（含可选 momentum）。

提示：
  - step() 遍历 params，对每个 requires_grad=True 的 Tensor：
      param.data -= lr * param.grad
  - momentum 版本：维护速度项 v，v = momentum * v + lr * param.grad，
    然后 param.data -= v
  - zero_grad() 调用每个 param 的 .zero_grad()

运行测试：
  python3 -m pytest tests/test_tensor.py -k T9 -q
"""
from __future__ import annotations

from minilm.tensor.tensor import Tensor


class SGD:
    """随机梯度下降（含可选 momentum）。

    参数：
        params   : 参数列表（来自 model.parameters()）
        lr       : 学习率
        momentum : 动量系数（默认 0，即普通 SGD）
    """

    def __init__(self, params: list[Tensor], lr: float, momentum: float = 0.0):
        """提示：保存 params/lr/momentum；若 momentum > 0，初始化速度字典。"""
        raise NotImplementedError

    def step(self) -> None:
        """用当前 .grad 更新 .data。

        提示：
          for p in self.params:
              if p.grad is None:
                  continue
              # 普通 SGD：p.data -= self.lr * p.grad
              # momentum SGD：v = momentum * v + lr * grad; p.data -= v
        """
        raise NotImplementedError

    def zero_grad(self) -> None:
        """将所有参数的梯度清零。"""
        raise NotImplementedError
