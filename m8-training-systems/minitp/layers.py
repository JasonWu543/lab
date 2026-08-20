"""Student exercises: Megatron-style tensor parallel linear layers."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class _CopyToTP(torch.autograd.Function):
    """列切边界的 autograd 原语：两个方向各需要什么通信语义？"""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("复制到 TP 区域的前向应改变张量吗？")

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        raise NotImplementedError("列切后的输入梯度应如何跨 rank 合并？")


class _ReduceFromTP(torch.autograd.Function):
    """行切边界的 autograd 原语：两个方向各需要什么通信语义？"""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("行切产生的局部部分积如何组成完整输出？")

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        raise NotImplementedError("前向求和原语的反向应如何处理完整梯度？")


class ColumnParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, world_size, rank, seed):
        super().__init__()
        raise NotImplementedError("请确定列切权重的形状、初始化和本 rank 切片")

    def forward(self, x):
        raise NotImplementedError("列切前向需要哪个 autograd 通信原语？")


class RowParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, world_size, rank, seed):
        super().__init__()
        raise NotImplementedError("请确定行切权重的形状、初始化和本 rank 切片")

    def forward(self, x):
        raise NotImplementedError("局部线性部分积之后需要哪种 collective？")


class TPMLP(nn.Module):
    """Column(d -> 4d, GELU) followed by Row(4d -> d)."""

    def __init__(self, dim: int, world_size: int, rank: int, seed: int):
        super().__init__()
        raise NotImplementedError("请组合列切、GELU 与行切两层")

    def forward(self, x):
        raise NotImplementedError("列切和行切如何串联而不 gather 中间激活？")
