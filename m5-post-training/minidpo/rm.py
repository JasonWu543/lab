"""minidpo/rm.py — 学生实现：RewardModel + Bradley-Terry loss。

接口（已冻结）：
    class RewardModel(nn.Module):
        __init__(self, base_model)
        forward(self, input_ids: Tensor, seq_lens: Tensor) -> Tensor  # (B,)

    def bt_loss(chosen_rewards, rejected_rewards) -> Tensor

实现要点（思考问题）：
    RewardModel：
        1. base_model.model(input_ids=...) 调用 backbone（不含 lm_head），
           得到 last_hidden_state (B, T, H)。
        2. 加一个 Linear(H, 1) 作为 value_head，得到 (B, T) 的分数。
        3. 「取每条序列最后一个非 pad token」——已知 seq_lens，
           问：索引是多少？如何用 .gather() 取对应位置？

    bt_loss：
        Bradley-Terry 模型：P(c > r) = σ(r_c - r_r)
        loss = -log P(c > r) 对 batch 求均值。
        问：F.logsigmoid 和 -log σ(x) 是什么关系？
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class RewardModel(nn.Module):
    """base 最后隐层 → Linear(H, 1)，取每条序列最后一个非 pad token 的标量。"""

    def __init__(self, base_model):
        super().__init__()
        # TODO: 保存 base_model，初始化 value_head = Linear(H, 1)
        raise NotImplementedError("RewardModel.__init__ 尚未实现")

    def forward(self, input_ids: Tensor, seq_lens: Tensor) -> Tensor:  # (B,)
        # TODO: backbone → value_head → gather 最后非 pad 位置
        raise NotImplementedError("RewardModel.forward 尚未实现")


def bt_loss(chosen_rewards: Tensor, rejected_rewards: Tensor) -> Tensor:
    """Bradley-Terry：-log σ(r_c - r_r)，返回 batch 均值。

    TODO: 学生实现。
    """
    raise NotImplementedError("bt_loss 尚未实现，请完成 TODO")
