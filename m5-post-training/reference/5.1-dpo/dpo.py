"""参考答案 5.1 — DPO loss。"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def dpo_loss(policy_chosen_logps: Tensor, policy_rejected_logps: Tensor,
             ref_chosen_logps: Tensor, ref_rejected_logps: Tensor,
             beta: float = 0.1) -> tuple[Tensor, Tensor, Tensor]:
    """返回 (loss 均值, chosen_implicit_reward, rejected_implicit_reward)。

    delta_policy = policy_chosen - policy_rejected
    delta_ref    = ref_chosen    - ref_rejected
    loss = -log σ(beta * (delta_policy - delta_ref))
    隐式 reward = beta * (policy_logp - ref_logp)  (detach，用于监控 margin)
    """
    delta_policy = policy_chosen_logps - policy_rejected_logps
    delta_ref = ref_chosen_logps - ref_rejected_logps
    logits = beta * (delta_policy - delta_ref)
    loss = -F.logsigmoid(logits).mean()

    chosen_reward = beta * (policy_chosen_logps - ref_chosen_logps).detach()
    rejected_reward = beta * (policy_rejected_logps - ref_rejected_logps).detach()
    return loss, chosen_reward, rejected_reward
