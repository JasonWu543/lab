"""学生任务：MiniMoELM 组装。

主要是「把前面写好的模块接起来」。结构框架给足，重点是 tied head、
RoPE buffer、KV cache 的位置管理、use_mtp 分支。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from minimoe.config import MoEConfig
from minimoe.mla import MLA, MLACache, RMSNorm, precompute_rope
from minimoe.moe import MoELayer, SwiGLU
from minimoe.mtp import MTPHead


class Block(nn.Module):
    """pre-norm 残差块：x + MLA(norm(x)) ；x + FFN(norm(x))。
    use_moe=True -> ffn 是 MoELayer；否则是 dense SwiGLU（对照组）。
    """

    def __init__(self, cfg: MoEConfig, use_moe: bool):
        super().__init__()
        H = cfg.hidden_size
        self.attn_norm = RMSNorm(H, cfg.rms_norm_eps)
        self.attn = MLA(cfg)
        self.ffn_norm = RMSNorm(H, cfg.rms_norm_eps)
        # TODO(学生): 按 use_moe 选择 self.ffn = MoELayer(cfg) 或 SwiGLU(H, intermediate_size)
        raise NotImplementedError

    def forward(self, x, cos, sin, positions, kv_cache=None):
        # TODO(学生): 两个残差子层（attn 需要传 cos/sin/positions/kv_cache）
        raise NotImplementedError


class MiniMoELM(nn.Module):
    """embedding -> [Block] x L -> norm -> head（与 embedding tied）。

    __init__(cfg, use_moe=True, use_mtp=False)
    forward(input_ids, kv_caches=None) -> logits；use_mtp 时额外返回 mtp_logits。

    结构提示：
      self.embed  : nn.Embedding(vocab, H)
      self.blocks : ModuleList[Block] x num_layers
      self.norm   : RMSNorm(H)
      self.head   : Linear(H, vocab, bias=False)；self.head.weight = self.embed.weight  # tied
      RoPE：precompute_rope(qk_rope_head_dim, max_seq_len, rope_theta) 存成 buffer
      use_mtp：self.mtp = MTPHead(cfg, self.embed, self.head)

    forward 位置管理：
      有 kv_caches 时，起始位置 = caches[0].seq_len（decode 时要接着算 positions）
      逐层把对应的 cache 传给 block
    """

    def __init__(self, cfg: MoEConfig, use_moe: bool = True, use_mtp: bool = False):
        super().__init__()
        self.cfg = cfg
        self.use_moe = use_moe
        self.use_mtp = use_mtp
        # TODO(学生): 构造 embed / blocks / norm / head(tied) / RoPE buffer / (可选) mtp
        raise NotImplementedError

    def make_caches(self, batch_size, max_seq_len, device=None, dtype=torch.float32):
        # 已给：为每层创建一个 MLACache
        device = device or self.embed.weight.device
        return [
            MLACache(self.cfg, batch_size, max_seq_len, device, dtype)
            for _ in range(self.cfg.num_layers)
        ]

    def forward(self, input_ids, kv_caches=None):
        # TODO(学生):
        #   x = embed(input_ids)
        #   positions = arange(...)（有 cache 时从 caches[0].seq_len 起算）
        #   逐 block 前向（传对应 cache）
        #   h = norm(x); logits = head(h)
        #   use_mtp -> 额外算 mtp_logits 并一并返回
        raise NotImplementedError
