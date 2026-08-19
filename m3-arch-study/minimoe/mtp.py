"""学生任务：深度 1 的 Multi-Token Prediction（V3 风格顺序模块）。

TransformerBlock 的结构给到（标准 MHA + SwiGLU 残差），MTPHead 的核心由学生写。

思考题：MTP 的梯度流回主干的路径是什么？（关系到 forward 里怎么用 h）
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from minimoe.config import MoEConfig
from minimoe.mla import RMSNorm
from minimoe.moe import SwiGLU


class TransformerBlock(nn.Module):
    """MTP 内部的标准 block：pre_norm + causal MHA + post_norm + SwiGLU（均残差）。

    子模块：
      pre_norm  : RMSNorm(H)
      attn      : nn.MultiheadAttention(H, num_heads, batch_first=True)
      post_norm : RMSNorm(H)
      ffn       : SwiGLU(H, intermediate_size)
    """

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
    """深度 1 MTP：用位置 t 的主干隐层 h_t 与 t+1 token 的 embedding，预测 t+2。

    __init__ 需要构造（embed / head 由主干传入并共享）：
      self.embed, self.head           # 共享，直接存引用
      self.merge_proj : Linear(2*H, H, bias=False)
      self.norm_h     : RMSNorm(H)
      self.norm_e     : RMSNorm(H)
      self.transformer_block : TransformerBlock(cfg)

    forward 流程提示：
      h_normed = norm_h(h[:, :-1, :])                 # (B, T-1, H)
      e_normed = norm_e(embed(input_ids[:, 1:]))      # (B, T-1, H)
      merged   = merge_proj(cat([h_normed, e_normed], dim=-1))
      h_prime  = transformer_block(merged)
      logits   = head(h_prime)                        # (B, T-1, V)

    RED LINE：h 要不要 detach？（决定 MTP loss 能否反传到主干 —— T6 会检查）
    """

    def __init__(self, cfg: MoEConfig, embed: nn.Embedding, head: nn.Linear):
        super().__init__()
        self.cfg = cfg
        # TODO(学生): 存 embed/head 引用；构造 merge_proj / norm_h / norm_e / transformer_block
        raise NotImplementedError

    def forward(self, h: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        # h: (B, T, H) 主干最后隐层；input_ids: (B, T) -> 返回 (B, T-1, V)
        # TODO(学生): 实现上面的 merge -> block -> head 流程
        raise NotImplementedError
