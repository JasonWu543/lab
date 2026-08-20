"""学生任务：把 delta rule 封装为支持流式解码的完整注意力层。"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import KDAConfig


class DeltaAttention(nn.Module):
    """q/k/v、写入门、遗忘门投影和输出投影组成的完整层。

    思考：forward 和 step 怎样共享参数并保证相同的更新/读取顺序？step 除固定
    大小矩阵状态外是否还需要保存历史 token？
    """

    def __init__(self, cfg: KDAConfig):
        super().__init__()
        raise NotImplementedError("请定义投影与门，并保存流式计算所需配置")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """处理完整序列，返回与 x 同形状的输出。"""
        raise NotImplementedError("请用 chunkwise 核心实现整段前向")

    def step(
        self, x_t: torch.Tensor, state: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """处理一个 token；state 只能是每个 head 的 D×D 矩阵。"""
        raise NotImplementedError("请实现与 forward 数值一致的单步更新")


def state_bytes(cfg) -> int:
    """单请求的 decode 状态字节数，与序列长度无关。

    思考：每个 head 保存多少个元素？注意口径：递推按 SPEC 是 fp32 累加，
    decode 状态也以 fp32 持有（与输入 dtype 无关）——每元素几个字节？
    """
    raise NotImplementedError("请写出固定状态大小的闭式")
