"""Hyper-Connections：学生练习。"""

import torch
from torch import nn


class HyperConnection(nn.Module):
    """静态 Hyper-Connection；冻结构造对应恒等初始化。

    想一想：宽度混合、层输入读出和层输出写回各自改变哪一个轴？怎样的参数
    起点能让展开后的网络与普通 residual 网络完全一致？
    """
    def __init__(self, dim: int, expand_n: int, layer_id: int):
        super().__init__()
        raise NotImplementedError("请定义并初始化 alpha、read_weight 与 beta")

    def width_mix(self, h: torch.Tensor) -> torch.Tensor:
        """混合 ``(B,T,n,C)`` 的 stream 轴。"""
        raise NotImplementedError("请实现 stream 宽度混合")

    def read(self, h: torch.Tensor) -> torch.Tensor:
        """从多条 stream 读出层输入。"""
        raise NotImplementedError("请实现层输入读出")

    def write(self, h: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        """把层输出按深度权重写回 streams。"""
        raise NotImplementedError("请实现层输出写回")


def expand_stream(x: torch.Tensor, n: int) -> torch.Tensor:
    """把一条 residual stream 复制为 ``n`` 条；哪个新轴代表 stream？"""
    raise NotImplementedError("请实现 stream 展开")


def collapse_stream(h: torch.Tensor) -> torch.Tensor:
    """将多条 stream 均值聚合；应沿哪个轴？"""
    raise NotImplementedError("请实现 stream 聚合")
