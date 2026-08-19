"""参考实现：Router + SwiGLU expert + MoELayer（dropless gather-scatter）。

隐藏了红线设计（仅供答案对照）：
- bias 只加在 selection（选 top-k），不进 gating 权重
- update_bias：负载高 -> bias 减小，负载低 -> bias 增大
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .config_solution import MoEConfig
except ImportError:  # 允许作为独立模块导入
    from config_solution import MoEConfig


class SwiGLU(nn.Module):
    """SwiGLU FFN expert: down(silu(gate(x)) * up(x))."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Router(nn.Module):
    """aux-loss-free 路由器。

    - logits = x @ weight.T          weight: (n_experts, hidden)
    - score  = sigmoid(logits)       (N, n_experts)
    - selection：用 score + bias 选 top-k（bias 是 buffer，不参与梯度）
    - gating ：用 **不加 bias** 的 score 在被选位置归一化到和为 1
    """

    def __init__(self, cfg: MoEConfig):
        super().__init__()
        self.cfg = cfg
        self.n_experts = cfg.n_routed_experts
        self.top_k = cfg.top_k
        self.gamma = cfg.bias_update_speed
        self.weight = nn.Parameter(torch.empty(self.n_experts, cfg.hidden_size))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        # bias 是 buffer，不是 Parameter -> 无梯度
        self.register_buffer("bias", torch.zeros(self.n_experts))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (N, H)
        logits = x @ self.weight.T          # (N, n_experts)
        score = torch.sigmoid(logits)       # (N, n_experts)

        # selection：加 bias 后取 top-k index（红线：bias 仅影响选择）
        biased = score + self.bias          # (N, n_experts)
        topk_idx = torch.topk(biased, self.top_k, dim=-1).indices  # (N, top_k)

        # gating：取**不加 bias** 的 score，在被选位置归一化
        topk_score = torch.gather(score, 1, topk_idx)              # (N, top_k)
        topk_weight = topk_score / (topk_score.sum(dim=-1, keepdim=True) + 1e-20)
        return topk_idx, topk_weight

    @torch.no_grad()
    def update_bias(self, topk_idx: torch.Tensor) -> None:
        # 统计本 batch 各 expert 负载
        load = torch.bincount(topk_idx.flatten(), minlength=self.n_experts).float()
        mean = load.sum() / self.n_experts
        # 红线：过载 -> bias 减小（降低被选概率）；欠载 -> bias 增大
        self.bias += self.gamma * (load < mean).float()
        self.bias -= self.gamma * (load > mean).float()


class MoELayer(nn.Module):
    """fine-grained routed experts + always-on shared experts, dropless。"""

    def __init__(self, cfg: MoEConfig):
        super().__init__()
        self.cfg = cfg
        self.router = Router(cfg)
        self.experts = nn.ModuleList(
            [SwiGLU(cfg.hidden_size, cfg.moe_intermediate_size) for _ in range(cfg.n_routed_experts)]
        )
        # shared expert：n_shared_experts 个宽度合并成一个（等价的宽 SwiGLU）
        self.shared = SwiGLU(
            cfg.hidden_size, cfg.moe_intermediate_size * cfg.n_shared_experts
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, H = x.shape
        x_flat = x.reshape(-1, H)                        # (N, H)
        N = x_flat.shape[0]

        topk_idx, topk_weight = self.router(x_flat)      # (N, k), (N, k)
        if self.training:
            self.router.update_bias(topk_idx)

        out = torch.zeros_like(x_flat)                   # (N, H)

        # dropless gather-scatter：逐 expert 处理命中它的 token
        for e in range(self.cfg.n_routed_experts):
            # 哪些 (token, slot) 命中 expert e
            hit = topk_idx == e                          # (N, k) bool
            token_mask = hit.any(dim=-1)                 # (N,)
            if not token_mask.any():
                continue  # 空 expert：跳过，不产生 NaN
            token_ids = token_mask.nonzero(as_tuple=True)[0]
            xe = x_flat[token_ids]                        # (m, H)
            ye = self.experts[e](xe)                      # (m, H)
            # 该 token 分给 expert e 的 gating 权重（top_k 里可能占某个 slot）
            w = (topk_weight * hit.float()).sum(dim=-1)[token_ids]  # (m,)
            out.index_add_(0, token_ids, ye * w.unsqueeze(-1))

        # shared expert：所有 token 恒定参与
        out = out + self.shared(x_flat)
        return out.reshape(B, T, H)
