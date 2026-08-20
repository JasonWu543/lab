"""Muon 优化器的矩阵更新：学生练习。"""

import torch


def newton_schulz_msign(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """用 SPEC 公开的五次迭代公式近似矩阵符号函数。

    如何统一处理高矩阵与宽矩阵？哪些计算必须提升到 fp32？
    """
    raise NotImplementedError("请实现 Newton--Schulz msign")


class Muon(torch.optim.Optimizer):
    """仅接受二维参数的最小 Muon。"""
    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95,
                 nesterov: bool = True):
        raise NotImplementedError("请校验参数并初始化优化器状态")

    @torch.no_grad()
    def step(self, closure=None):
        """执行 momentum、可选 Nesterov、msign、形状缩放和参数更新。"""
        raise NotImplementedError("请实现一次 Muon 更新")
