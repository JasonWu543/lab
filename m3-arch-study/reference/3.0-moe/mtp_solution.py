"""参考实现：深度 1 的 Multi-Token Prediction（V3 风格顺序模块）。

隐藏红线：梯度如何流回主干（merge 输入的 h 不 detach，因此 MTP loss
会通过 h -> 主干反传）。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .config_solution import MoEConfig
    from .mla_solution import RMSNorm
    from .moe_solution import SwiGLU
except ImportError:
    from config_solution import MoEConfig
    from mla_solution import RMSNorm
    from moe_solution import SwiGLU


class TransformerBlock(nn.Module):
    """MTP 内部用的标准 block：pre_norm + MHA + post_norm + SwiGLU（残差）。"""

    def __init__(self, cfg: MoEConfig):
        super().__init__()
        H = cfg.hidden_size
        self.pre_norm = RMSNorm(H, cfg.rms_norm_eps)
        self.attn = nn.MultiheadAttention(H, cfg.num_heads, batch_first=True)
        self.post_norm = RMSNorm(H, cfg.rms_norm_eps)
        self.ffn = SwiGLU(H, cfg.intermediate_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[1]
        causal = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )
        h = self.pre_norm(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=causal, need_weights=False)
        x = x + attn_out
        x = x + self.ffn(self.post_norm(x))
        return x


class MTPHead(nn.Module):
    """h'_t = TransformerBlock( merge( RMSNorm(h_t), RMSNorm(emb(next token)) ) )
    用共享 embedding / 输出头预测 t+2。"""

    def __init__(self, cfg: MoEConfig, embed: nn.Embedding, head: nn.Linear):
        super().__init__()
        self.cfg = cfg
        self.embed = embed  # 共享主干 embedding
        self.head = head    # 共享主干输出头
        H = cfg.hidden_size
        self.merge_proj = nn.Linear(2 * H, H, bias=False)
        self.norm_h = RMSNorm(H, cfg.rms_norm_eps)
        self.norm_e = RMSNorm(H, cfg.rms_norm_eps)
        self.transformer_block = TransformerBlock(cfg)

    def forward(self, h: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        # h: (B, T, H) 主干隐层；input_ids: (B, T)
        # 红线：h 不 detach -> 梯度经 merge 回流主干
        h_normed = self.norm_h(h[:, :-1, :])                  # (B, T-1, H)
        e_normed = self.norm_e(self.embed(input_ids[:, 1:]))  # (B, T-1, H)
        merged = self.merge_proj(torch.cat([h_normed, e_normed], dim=-1))
        h_prime = self.transformer_block(merged)              # (B, T-1, H)
        logits = self.head(h_prime)                           # (B, T-1, V)
        return logits
