"""minigrpo/loss.py — 学生实现：GRPO token-level clipped surrogate + k3 KL。

接口（已冻结）：
    grpo_loss(logps, old_logps, ref_logps, advantages, mask,
              clip_eps=0.2, kl_coef=0.04)
    -> (loss: Tensor, stats: dict)
    stats 包含："pg_loss"、"kl"、"clip_frac"（均已 detach）

两个主要组件：

1. Clipped surrogate（Policy Gradient 部分）：
   ratio = exp(logps - old_logps)
   思考：ratio 超出 1±eps 时你希望梯度是什么？
   advantages 是序列级 (B,)，需要广播到 token 级 (B, T)。

2. k3 KL 估计（正则化部分）：
   这是 KL(π || ref) 的一种无偏估计。
   思考：k3 为什么非负？当 π == ref 时 k3 = ?

mask 处理：
   按 mask 内的 token 数做平均（分母 = mask.sum().clamp(min=1)）。
   clip_frac = mask 内 ratio 超出 [1-eps, 1+eps] 的 token 比例（detach）。
"""
from __future__ import annotations

import torch
from torch import Tensor


def grpo_loss(logps: Tensor,                 # (B, T) 当前策略 per-token logp
              old_logps: Tensor,             # (B, T) rollout 时的策略
              ref_logps: Tensor,             # (B, T) 冻结参考策略
              advantages: Tensor,            # (B,)   序列级，广播到 token
              mask: Tensor,                  # (B, T) completion token 为 1
              clip_eps: float = 0.2,
              kl_coef: float = 0.04) -> tuple[Tensor, dict]:
    """token-level clipped surrogate + k3 KL 估计。
    按 mask 内 token 平均。返回 (loss, {"pg_loss","kl","clip_frac"})。

    TODO: 学生实现。
    """
    raise NotImplementedError("grpo_loss 尚未实现，请完成 TODO")
