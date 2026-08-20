"""参考答案 5.2 — GRPO token-level clipped surrogate + k3 KL。"""
from __future__ import annotations

import torch
from torch import Tensor


def grpo_loss(logps: Tensor,        # (B, T) 当前策略 per-token logp
              old_logps: Tensor,    # (B, T) rollout 时的策略
              ref_logps: Tensor,    # (B, T) 冻结参考策略
              advantages: Tensor,   # (B,)   序列级，广播到 token
              mask: Tensor,         # (B, T) completion token 为 1
              clip_eps: float = 0.2,
              kl_coef: float = 0.04) -> tuple[Tensor, dict]:
    """loss = -E[min(ratio*A, clip(ratio,1±eps)*A)] + kl_coef * E[k3]
    k3 = exp(ref-logp) - (ref-logp) - 1（无偏且非负的 KL 估计）。
    按 mask 内 token 平均。返回 (loss, {"pg_loss","kl","clip_frac"})。
    """
    adv = advantages.unsqueeze(1)  # (B, 1) 广播到 token
    ratio = torch.exp(logps - old_logps)                       # (B, T)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    pg_token = -torch.min(unclipped, clipped)                  # (B, T)

    # k3 KL 估计（policy 对 ref）
    diff = ref_logps - logps                                   # (B, T)
    k3 = torch.exp(diff) - diff - 1.0                          # (B, T) >= 0

    mask_f = mask.float()
    denom = mask_f.sum().clamp(min=1.0)
    pg_loss = (pg_token * mask_f).sum() / denom
    kl = (k3 * mask_f).sum() / denom
    loss = pg_loss + kl_coef * kl

    # clip_frac：ratio 被截断（超出 1±eps）的 masked token 比例
    clipped_flag = ((ratio > 1.0 + clip_eps) | (ratio < 1.0 - clip_eps)).float()
    clip_frac = (clipped_flag * mask_f).sum() / denom

    stats = {
        "pg_loss": pg_loss.detach(),
        "kl": kl.detach(),
        "clip_frac": clip_frac.detach(),
    }
    return loss, stats
