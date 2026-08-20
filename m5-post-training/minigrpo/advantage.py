"""minigrpo/advantage.py — 学生实现：组内标准化 advantage。

接口（已冻结）：
    group_advantages(rewards: Tensor, eps: float = 1e-6) -> Tensor
    输入：(n_prompts, G)，输出：(n_prompts, G)

公式：A_i = (r_i - mean_g) / (std_g + eps)，逐组计算。

关键边界：
    - 用 keepdim=True 保持维度对齐（广播）。
    - 全组同分（std=0）时，分子 = 0，advantage 全 0（自然满足）。
    - 用 unbiased=False 计算 std（与组内 N 个样本的最大似然估计一致）。

思考问题：
    标准化后每组的均值应是多少？std 应是多少（eps 趋近于 0 时）？
    用这个性质手算验证你的实现。
"""
from __future__ import annotations

import torch
from torch import Tensor


def group_advantages(rewards: Tensor,        # (n_prompts, G)
                     eps: float = 1e-6) -> Tensor:
    """组内标准化：(r - mean_g) / (std_g + eps)，逐组计算。
    全组同分（std=0）→ 该组 advantage 全 0。返回 (n_prompts, G)。

    TODO: 学生实现。
    """
    raise NotImplementedError("group_advantages 尚未实现，请完成 TODO")
