"""学生任务：Multi-head Latent Attention（训练形态）+ 压缩 KV cache。

RMSNorm / RoPE 工具函数已给（非本关重点）。MLACache 的 cache 读写框架给出，
memory_bytes 与 MLA.forward 的核心由学生写。

思考题：MLA 的低秩压缩为什么配合 RoPE 时需要 decoupled 的 rope 分支？
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from minimoe.config import MoEConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps) * self.weight


# ---------------- RoPE 工具（已给，直接用）----------------
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


def mha_cache_bytes(cfg: MoEConfig, batch_size: int, seq_len: int, dtype) -> int:
    """等价标准 MHA cache 字节数（K + V 各存 num_heads 份 full head）。已给，用于对比。"""
    bpe = torch.finfo(dtype).bits // 8
    per_token = cfg.num_heads * (cfg.qk_nope_head_dim + cfg.qk_rope_head_dim + cfg.v_head_dim)
    return batch_size * seq_len * per_token * 2 * bpe


class MLACache:
    """只缓存 (latent, k_rope)。update / seq_len 框架已给；memory_bytes 由学生写。"""

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
        # 写入新 token 的 latent / k_rope，返回截至当前的完整缓存
        t = latent.shape[1]
        self.latent_cache[:, self._len : self._len + t] = latent
        self.k_rope_cache[:, self._len : self._len + t] = k_rope
        self._len += t
        return self.latent_cache[:, : self._len], self.k_rope_cache[:, : self._len]

    @property
    def seq_len(self) -> int:
        return self._len

    def memory_bytes(self) -> int:
        """当前缓存实际占用字节数。
        提示：只存了 latent (kv_lora_rank) 和 k_rope (qk_rope_head_dim) 两块，
        每元素字节数可用 torch.finfo(self.dtype).bits // 8。
        """
        # TODO(学生): 返回闭式字节数（T5 会与手算对比）
        raise NotImplementedError


class MLA(nn.Module):
    """MLA 训练形态（不做矩阵吸收）。

    子模块（全部 bias=False，除 layernorm）：
      q_a_proj      : Linear(H, q_lora_rank)
      q_a_layernorm : RMSNorm(q_lora_rank)
      q_b_proj      : Linear(q_lora_rank, num_heads*(qk_nope+qk_rope))
      kv_a_proj     : Linear(H, kv_lora_rank)
      kv_a_layernorm: RMSNorm(kv_lora_rank)
      kv_b_proj     : Linear(kv_lora_rank, num_heads*(qk_nope+v_head))
      k_rope_proj   : Linear(H, qk_rope)          # 每 token 一份，跨 head 共享
      o_proj        : Linear(num_heads*v_head, H)
    """

    def __init__(self, cfg: MoEConfig):
        super().__init__()
        self.cfg = cfg
        self.num_heads = cfg.num_heads
        self.qk_nope = cfg.qk_nope_head_dim
        self.qk_rope = cfg.qk_rope_head_dim
        self.v_head = cfg.v_head_dim
        self.qk_head = self.qk_nope + self.qk_rope
        # TODO(学生): 定义上面列出的所有子模块
        raise NotImplementedError

    def forward(self, x, cos, sin, positions, kv_cache: "MLACache | None" = None):
        # x: (B, T, H)
        # 无 cache 流程提示：
        #   latent = kv_a_proj(x)                       # (B, T, kv_lora_rank)
        #   k_rope = apply_rope(k_rope_proj(x), ...)    # (B, T, qk_rope)  decoupled 分支
        #   kv_up  = kv_b_proj(kv_a_layernorm(latent)) 拆成 k_nope (…heads,nope), v (…heads,v)
        #   q      = q_b_proj(q_a_layernorm(q_a_proj(x))) 拆成 q_nope, q_rope；对 q_rope 施 RoPE
        #   score 的合成自己推——q 有 nope/rope 两部分，k 也有：它们怎么配对？
        #     等价视角：把 (nope, rope) 拼成一个长为 qk_nope+qk_rope 的向量做点积，
        #     由此 scale 该除以什么？k_rope 每 token 只有一份，跨 head 怎么用？
        #   causal mask（用 q/k 的绝对位置比较）-> softmax -> 加权 v -> o_proj
        #
        # 带 cache 流程提示：
        #   prefill：把整段 (latent, k_rope) 写入 cache
        #   decode ：每步只喂 1 个 token，cache.update 后用**整段缓存**的 latent 还原 k_nope/v
        #   注意 q 的位置是 positions，k 的位置是 arange(cache.seq_len)
        # TODO(学生): 实现 MLA 前向（cache / 非 cache 两条路径）
        raise NotImplementedError
