"""
Phase 1.2 参考答案：MiniLM model
学生文件：minilm/model/model.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from .config_solution import ModelConfig


def precompute_rope(head_dim: int, max_seq_len: int, theta: float = 10000.0, device=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """预计算 RoPE 的 cos/sin 表，shape: (max_seq_len, head_dim)"""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)  # (max_seq_len, head_dim//2)
    cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1)  # (max_seq_len, head_dim)
    sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1)  # (max_seq_len, head_dim)
    return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """前后分割旋转：x → [-x2, x1]"""
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply RoPE to q and k.
    q, k: (B, n_heads, T, head_dim)
    cos, sin: (max_seq_len, head_dim)
    positions: (T,) or (B, T)
    """
    cos_pos = cos[positions]  # (T, head_dim) or (B, T, head_dim)
    sin_pos = sin[positions]
    if positions.dim() == 1:
        # (T, head_dim) → (1, 1, T, head_dim)
        cos_pos = cos_pos.unsqueeze(0).unsqueeze(0)
        sin_pos = sin_pos.unsqueeze(0).unsqueeze(0)
    else:
        # (B, T, head_dim) → (B, 1, T, head_dim)
        cos_pos = cos_pos.unsqueeze(1)
        sin_pos = sin_pos.unsqueeze(1)
    q_rot = q * cos_pos + rotate_half(q) * sin_pos
    k_rot = k * cos_pos + rotate_half(k) * sin_pos
    return q_rot, k_rot


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """将 k/v 从 num_kv_heads 扩展到 num_heads"""
    if n_rep == 1:
        return x
    B, n_kv_heads, T, head_dim = x.shape
    return x.unsqueeze(2).expand(B, n_kv_heads, n_rep, T, head_dim).reshape(B, n_kv_heads * n_rep, T, head_dim)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        rms = (x.pow(2).mean(-1, keepdim=True) + self.eps).rsqrt()
        x = x * rms
        return (self.weight * x).to(orig_dtype)


class KVCache:
    def __init__(self, cfg: ModelConfig, batch_size: int, max_seq_len: int, device, dtype):
        head_dim = cfg.head_dim
        self._cache_k = [
            torch.zeros(batch_size, cfg.num_kv_heads, max_seq_len, head_dim, device=device, dtype=dtype)
            for _ in range(cfg.num_layers)
        ]
        self._cache_v = [
            torch.zeros(batch_size, cfg.num_kv_heads, max_seq_len, head_dim, device=device, dtype=dtype)
            for _ in range(cfg.num_layers)
        ]
        self._seq_len = 0

    @property
    def seq_len(self) -> int:
        return self._seq_len

    def update(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """写入新 k/v，返回完整历史 k/v（含本次）"""
        T_new = k.shape[2]
        start = self._seq_len
        self._cache_k[layer_idx][:, :, start:start + T_new] = k
        self._cache_v[layer_idx][:, :, start:start + T_new] = v
        return (
            self._cache_k[layer_idx][:, :, :start + T_new],
            self._cache_v[layer_idx][:, :, :start + T_new],
        )


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.num_heads = cfg.num_heads
        self.num_kv_heads = cfg.num_kv_heads
        self.head_dim = cfg.head_dim
        self.n_rep = cfg.num_heads // cfg.num_kv_heads
        bias = cfg.attention_bias

        self.q_proj = nn.Linear(cfg.hidden_size, cfg.num_heads * cfg.head_dim, bias=bias)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.num_kv_heads * cfg.head_dim, bias=bias)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.num_kv_heads * cfg.head_dim, bias=bias)
        self.o_proj = nn.Linear(cfg.num_heads * cfg.head_dim, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, positions: torch.Tensor, kv_cache: Optional[KVCache] = None, layer_idx: int = 0) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)   # (B, nh, T, hd)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        q, k = apply_rope(q, k, cos, sin, positions)

        if kv_cache is not None:
            k, v = kv_cache.update(layer_idx, k, v)
            T_total = k.shape[2]  # cached + new
        else:
            T_total = T

        # GQA: repeat k/v
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        # Scaled dot-product attention (manual, with causal mask)
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, nh, T, T_total)

        # Causal mask
        cache_len = T_total - T
        mask = torch.full((T, T_total), float('-inf'), device=x.device, dtype=x.dtype)
        for i in range(T):
            mask[i, :cache_len + i + 1] = 0.0
        attn = attn + mask.unsqueeze(0).unsqueeze(0)

        attn = torch.softmax(attn.float(), dim=-1).to(x.dtype)
        out = torch.matmul(attn, v)  # (B, nh, T, hd)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out)


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.norm1 = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, positions: torch.Tensor, kv_cache: Optional[KVCache] = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin, positions, kv_cache, self.layer_idx)
        x = x + self.mlp(self.norm2(x))
        return x


class MiniLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([DecoderLayer(cfg, i) for i in range(cfg.num_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

        # Weight tying
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.embedding.weight

        # Precompute RoPE buffers
        cos, sin = precompute_rope(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer('cos', cos, persistent=False)
        self.register_buffer('sin', sin, persistent=False)

        # Init weights
        self._init_weights()

    def _init_weights(self):
        std = 0.02
        scaled_std = 0.02 / (2 * self.cfg.num_layers) ** 0.5
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if 'o_proj' in name or 'down_proj' in name:
                    nn.init.normal_(module.weight, mean=0.0, std=scaled_std)
                else:
                    nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(self, input_ids: torch.Tensor, kv_cache: Optional[KVCache] = None, positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T = input_ids.shape
        x = self.embedding(input_ids)

        if positions is None:
            start = kv_cache.seq_len if kv_cache is not None else 0
            positions = torch.arange(start, start + T, device=input_ids.device)

        cos = self.cos
        sin = self.sin

        for layer in self.layers:
            x = layer(x, cos, sin, positions, kv_cache)

        if kv_cache is not None:
            kv_cache._seq_len += T

        x = self.norm(x)
        logits = self.lm_head(x)
        return logits
