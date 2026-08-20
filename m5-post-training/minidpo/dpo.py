"""minidpo/dpo.py — 学生实现：DPO loss。

接口（已冻结）：
    dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta=0.1)
    -> (loss: Tensor, chosen_implicit_reward: Tensor, rejected_implicit_reward: Tensor)

核心公式（请查阅 DPO 论文 Eq.7）：
    loss = -E[ log σ(β * ((log π_c - log π_r) - (log ref_c - log ref_r))) ]

隐式 reward = β * (log π - log ref)，用于监控 margin。

思考问题：
    1. delta_policy = ? delta_ref = ?  logits 传给 F.logsigmoid 的是什么？
    2. 隐式 reward 监控用，不参与梯度——如何让它 detach？
    3. 函数应返回 (loss, chosen_reward, rejected_reward) 三元组。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def dpo_loss(policy_chosen_logps: Tensor, policy_rejected_logps: Tensor,
             ref_chosen_logps: Tensor, ref_rejected_logps: Tensor,
             beta: float = 0.1) -> tuple[Tensor, Tensor, Tensor]:
    """返回 (loss 均值, chosen_implicit_reward, rejected_implicit_reward)。

    TODO: 学生实现。
    隐式 reward = beta * (policy_logp - ref_logp)，detach 后返回（监控用）。
    """
    raise NotImplementedError("dpo_loss 尚未实现，请完成 TODO")
