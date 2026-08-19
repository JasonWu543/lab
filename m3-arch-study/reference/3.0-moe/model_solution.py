"""参考实现：MiniMoELM 组装。

embedding -> [MLA + (MoELayer 或 dense SwiGLU)] x L -> norm -> head（tied）。
"""
from __future__ import annotations

import torch
import torch.nn as nn

try:
    from .config_solution import MoEConfig
    from .mla_solution import MLA, MLACache, RMSNorm, precompute_rope
    from .moe_solution import MoELayer, SwiGLU
    from .mtp_solution import MTPHead
except ImportError:
    from config_solution import MoEConfig
    from mla_solution import MLA, MLACache, RMSNorm, precompute_rope
    from moe_solution import MoELayer, SwiGLU
    from mtp_solution import MTPHead


class Block(nn.Module):
    def __init__(self, cfg: MoEConfig, use_moe: bool):
        super().__init__()
        H = cfg.hidden_size
        self.attn_norm = RMSNorm(H, cfg.rms_norm_eps)
        self.attn = MLA(cfg)
        self.ffn_norm = RMSNorm(H, cfg.rms_norm_eps)
        if use_moe:
            self.ffn = MoELayer(cfg)
        else:
            self.ffn = SwiGLU(H, cfg.intermediate_size)

    def forward(self, x, cos, sin, positions, kv_cache=None):
        x = x + self.attn(self.attn_norm(x), cos, sin, positions, kv_cache)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class MiniMoELM(nn.Module):
    def __init__(self, cfg: MoEConfig, use_moe: bool = True, use_mtp: bool = False):
        super().__init__()
        self.cfg = cfg
        self.use_moe = use_moe
        self.use_mtp = use_mtp
        H = cfg.hidden_size
        self.embed = nn.Embedding(cfg.vocab_size, H)
        self.blocks = nn.ModuleList([Block(cfg, use_moe) for _ in range(cfg.num_layers)])
        self.norm = RMSNorm(H, cfg.rms_norm_eps)
        self.head = nn.Linear(H, cfg.vocab_size, bias=False)
        self.head.weight = self.embed.weight  # tied

        cos, sin = precompute_rope(cfg.qk_rope_head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        if use_mtp:
            self.mtp = MTPHead(cfg, self.embed, self.head)

    def make_caches(self, batch_size, max_seq_len, device=None, dtype=torch.float32):
        device = device or self.embed.weight.device
        return [
            MLACache(self.cfg, batch_size, max_seq_len, device, dtype)
            for _ in range(self.cfg.num_layers)
        ]

    def forward(self, input_ids, kv_caches=None):
        B, T = input_ids.shape
        x = self.embed(input_ids)
        if kv_caches is not None:
            start = kv_caches[0].seq_len
            positions = torch.arange(start, start + T, device=input_ids.device)
        else:
            positions = torch.arange(T, device=input_ids.device)

        for i, block in enumerate(self.blocks):
            cache = kv_caches[i] if kv_caches is not None else None
            x = block(x, self.rope_cos, self.rope_sin, positions, cache)

        h = self.norm(x)
        logits = self.head(h)

        if self.use_mtp:
            mtp_logits = self.mtp(h, input_ids)
            return logits, mtp_logits
        return logits
