"""Phase 1.1 — 优化器骨架（学生实现文件）

实现 SGD（含可选 momentum）。

提示：
  - step() 应跳过没有梯度的参数；先从普通 SGD 的更新含义推导更新方向
  - momentum 版本需要保存跨 step 的状态；思考速度如何组合历史与当前梯度
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
              # 普通 SGD 与 momentum 的更新式请从课程材料推导
        """
        raise NotImplementedError

    def zero_grad(self) -> None:
        """将所有参数的梯度清零。"""
        raise NotImplementedError
