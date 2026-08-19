"""minilm.tensor 子包。

导出：
    Tensor   — 核心 Tensor 类（带 autograd）
    no_grad  — 上下文管理器，禁止建计算图
    nn       — 神经网络模块（Module / Linear / RMSNorm / SwiGLU）
    optim    — 优化器（SGD）
"""
from minilm.tensor.tensor import Tensor, no_grad
from minilm.tensor import nn, optim

__all__ = ["Tensor", "no_grad", "nn", "optim"]
