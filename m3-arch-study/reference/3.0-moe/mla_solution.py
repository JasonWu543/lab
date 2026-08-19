"""参考实现：Multi-head Latent Attention（V2/V3 训练形态）+ 压缩 KV cache。

隐藏红线：decoupled RoPE 分支如何拼进 attention score。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .config_solution import MoEConfig
except ImportError:
    from config_solution import MoEConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps) * self.weight


# ---------------- RoPE ----------------
def precompute_rope(head_dim: int, max_seq_len: int, theta: float = 10000.0):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len).float()
    freqs = torch.outer(t, inv_freq)
    cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1)
    sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1)
    return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, positions: torch.Tensor):
    cos_p = cos[positions]
    sin_p = sin[positions]
    return x * cos_p + rotate_half(x) * sin_p


class MLACache:
    """只缓存 (latent, k_rope)，因此显存远小于标准 MHA。"""

    def __init__(self, cfg: MoEConfig, batch_size, max_seq_len, device, dtype):
        self.cfg = cfg
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.dtype = dtype
        self._len = 0
        self.latent_cache = torch.zeros(
            batch_size, max_seq_len, cfg.kv_lora_rank, device=device, dtype=dtype
        )
        self.k_rope_cache = torch.zeros(
            batch_size, max_seq_len, cfg.qk_rope_head_dim, device=device, dtype=dtype
        )

    def update(self, latent: torch.Tensor, k_rope: torch.Tensor):
        # latent: (B, t, kv_lora_rank); k_rope: (B, t, qk_rope_head_dim)
        t = latent.shape[1]
        self.latent_cache[:, self._len : self._len + t] = latent
        self.k_rope_cache[:, self._len : self._len + t] = k_rope
        self._len += t
        return self.latent_cache[:, : self._len], self.k_rope_cache[:, : self._len]

    @property
    def seq_len(self) -> int:
        return self._len

    def memory_bytes(self) -> int:
        bpe = torch.finfo(self.dtype).bits // 8
        return (
            self.batch_size
            * self._len
            * (self.cfg.kv_lora_rank + self.cfg.qk_rope_head_dim)
            * bpe
        )


def mha_cache_bytes(cfg: MoEConfig, batch_size: int, seq_len: int, dtype) -> int:
    """等价标准 MHA cache 字节数（K + V 各存 num_heads 份 full head）。"""
    bpe = torch.finfo(dtype).bits // 8
    per_token = cfg.num_heads * (cfg.qk_nope_head_dim + cfg.qk_rope_head_dim + cfg.v_head_dim)
    return batch_size * seq_len * per_token * 2 * bpe


class MLA(nn.Module):
    def __init__(self, cfg: MoEConfig):
        super().__init__()
        self.cfg = cfg
        H = cfg.hidden_size
        self.num_heads = cfg.num_heads
        self.qk_nope = cfg.qk_nope_head_dim
        self.qk_rope = cfg.qk_rope_head_dim
        self.v_head = cfg.v_head_dim
        self.qk_head = self.qk_nope + self.qk_rope

        self.q_a_proj = nn.Linear(H, cfg.q_lora_rank, bias=False)
        self.q_a_layernorm = RMSNorm(cfg.q_lora_rank, cfg.rms_norm_eps)
        self.q_b_proj = nn.Linear(cfg.q_lora_rank, self.num_heads * self.qk_head, bias=False)

        self.kv_a_proj = nn.Linear(H, cfg.kv_lora_rank, bias=False)
        self.kv_a_layernorm = RMSNorm(cfg.kv_lora_rank, cfg.rms_norm_eps)
        self.kv_b_proj = nn.Linear(
            cfg.kv_lora_rank, self.num_heads * (self.qk_nope + self.v_head), bias=False
        )
        self.k_rope_proj = nn.Linear(H, self.qk_rope, bias=False)  # 每 token 一份，跨 head 共享

        self.o_proj = nn.Linear(self.num_heads * self.v_head, H, bias=False)

    def _compute_kv(self, kv_normed: torch.Tensor):
        B, S, _ = kv_normed.shape
        kv = self.kv_b_proj(kv_normed)  # (B, S, heads*(nope+v))
        kv = kv.view(B, S, self.num_heads, self.qk_nope + self.v_head)
        k_nope, v = kv.split([self.qk_nope, self.v_head], dim=-1)
        return k_nope, v  # (B, S, heads, nope), (B, S, heads, v)

    def forward(self, x, cos, sin, positions, kv_cache: MLACache | None = None):
        B, T, _ = x.shape

        latent = self.kv_a_proj(x)                    # (B, T, kv_lora_rank)
        k_rope = self.k_rope_proj(x)                  # (B, T, qk_rope)
        # RoPE 施加在 rope key 上（decoupled 分支，每 token 一份）
        k_rope = apply_rope(k_rope, cos, sin, positions)

        if kv_cache is not None:
            q_positions = positions
            latent_all, k_rope_all = kv_cache.update(latent, k_rope)
            # k 的位置是整段缓存 [0, seq_len)
            k_positions = torch.arange(kv_cache.seq_len, device=x.device)
        else:
            latent_all, k_rope_all = latent, k_rope
            q_positions = positions
            k_positions = positions

        S = latent_all.shape[1]
        kv_normed = self.kv_a_layernorm(latent_all)
        k_nope, v = self._compute_kv(kv_normed)       # (B, S, heads, nope), (B, S, heads, v)

        # query
        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))  # (B, T, heads*qk_head)
        q = q.view(B, T, self.num_heads, self.qk_head)
        q_nope, q_rope = q.split([self.qk_nope, self.qk_rope], dim=-1)
        # 在 (B, heads, T, rope) 布局下施加 RoPE，使 cos[positions] (T, rope) 广播到最后两维
        q_rope = q_rope.transpose(1, 2)               # (B, heads, T, rope)
        q_rope = apply_rope(q_rope, cos, sin, q_positions)

        # attention score = nope 部分 + rope 部分（rope key 跨 head 广播）
        # nope: (B,heads,T,nope) x (B,heads,S,nope) -> (B,heads,T,S)
        q_nope_h = q_nope.transpose(1, 2)             # (B, heads, T, nope)
        k_nope_h = k_nope.transpose(1, 2)             # (B, heads, S, nope)
        score_nope = torch.matmul(q_nope_h, k_nope_h.transpose(-1, -2))

        # rope: q_rope (B,heads,T,rope) x k_rope_all (B,1,S,rope) 广播
        q_rope_h = q_rope                             # (B, heads, T, rope)
        k_rope_h = k_rope_all.unsqueeze(1)            # (B, 1, S, rope)
        score_rope = torch.matmul(q_rope_h, k_rope_h.transpose(-1, -2))

        scale = 1.0 / math.sqrt(self.qk_nope + self.qk_rope)
        scores = (score_nope + score_rope) * scale    # (B, heads, T, S)

        # causal mask：q 的绝对位置 >= k 的绝对位置才可见
        qpos = q_positions.view(T, 1)
        kpos = k_positions.view(1, S)
        causal = qpos >= kpos                          # (T, S)
        scores = scores.masked_fill(~causal, float("-inf"))

        attn = F.softmax(scores, dim=-1)               # (B, heads, T, S)
        v_h = v.transpose(1, 2)                        # (B, heads, S, v)
        ctx = torch.matmul(attn, v_h)                  # (B, heads, T, v)
        ctx = ctx.transpose(1, 2).reshape(B, T, self.num_heads * self.v_head)
        return self.o_proj(ctx)
