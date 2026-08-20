"""参考答案 5.1 — RewardModel + Bradley-Terry loss。"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class RewardModel(nn.Module):
    """base 最后隐层 → Linear(H, 1)，取每条序列最后一个非 pad token 的标量。"""

    def __init__(self, base_model):
        super().__init__()
        self.base = base_model
        hidden = base_model.config.hidden_size
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, input_ids: Tensor, seq_lens: Tensor) -> Tensor:  # (B,)
        out = self.base.model(input_ids=input_ids)          # base backbone (no lm_head)
        hidden = out.last_hidden_state                       # (B, T, H)
        scores = self.value_head(hidden).squeeze(-1)         # (B, T)
        # 取每条序列最后一个非 pad token：索引 = seq_len - 1
        idx = (seq_lens - 1).clamp(min=0)                    # (B,)
        return scores.gather(1, idx.unsqueeze(1)).squeeze(1)  # (B,)


def bt_loss(chosen_rewards: Tensor, rejected_rewards: Tensor) -> Tensor:
    """Bradley-Terry：-log σ(r_c - r_r)，返回 batch 均值。"""
    return -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
