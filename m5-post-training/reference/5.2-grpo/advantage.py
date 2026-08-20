"""参考答案 5.2 — 组内标准化 advantage。"""
from __future__ import annotations

import torch
from torch import Tensor


def group_advantages(rewards: Tensor, eps: float = 1e-6) -> Tensor:
    """组内标准化：(r - mean_g) / (std_g + eps)，逐组计算。
    全组同分（std=0）→ 该组 advantage 全 0（分子=0）。返回 (n_prompts, G)。
    """
    mean = rewards.mean(dim=1, keepdim=True)              # (n_prompts, 1)
    std = rewards.std(dim=1, keepdim=True, unbiased=False)  # (n_prompts, 1)
    return (rewards - mean) / (std + eps)
