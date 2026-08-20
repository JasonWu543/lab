"""
Phase 1.2：MiniLM Qwen-like Transformer

实现顺序建议：
  1. precompute_rope → rotate_half → apply_rope
  2. RMSNorm.forward
  3. repeat_kv
  4. KVCache.update
  5. Attention.forward
  6. MLP.forward
  7. DecoderLayer.forward
  8. MiniLM._init_weights → MiniLM.forward
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from .config import ModelConfig


# ─────────────────────────── RoPE 工具函数 ───────────────────────────────

def precompute_rope(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    device=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    预计算 RoPE 的 cos/sin 查找表。

    提示：
      - 每对维度需要一个频率；从 RoFormer §3.2.2 推导频率随维度的变化
      - 位置和频率怎样组合，才能得到 (max_seq_len, head_dim//2) 的相位表？
      - 最终 cos/sin 的 shape 是 (max_seq_len, head_dim)：
        从 head_dim//2 到 head_dim 怎么排布，取决于 rotate_half 布局
        （前后对半，与 HF Qwen2 一致）——先想清楚 rotate_half 再回来写这里

    返回: (cos, sin)，shape 均为 (max_seq_len, head_dim)
    """
    raise NotImplementedError


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    RoPE 的「旋转 90°」辅助操作，x: (..., head_dim)。

    自己推：二维旋转 (x1, x2) → (x1 cosθ − x2 sinθ, x1 sinθ + x2 cosθ)
    可以写成 x*cos + rot(x)*sin 的形式——rot(x) 应该是什么？
    注意本实现用「前后对半」配对（维度 i 与 i + head_dim//2 是一对），
    不是奇偶交错。写完用 test_rope_vs_hf 验证。
    """
    raise NotImplementedError


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    对 q, k 施加 RoPE 旋转位置编码。

    q, k:      (B, n_heads, T, head_dim)
    cos, sin:  (max_seq_len, head_dim)
    positions: (T,) 或 (B, T)

    提示：先索引对应位置的 cos/sin，再检查 (T,) 与 (B,T) 两种 positions
    怎样广播到 q/k；旋转表达式请由 rotate_half 的二维几何意义推导。

    返回: (q_rot, k_rot)
    """
    raise NotImplementedError


# ─────────────────────────── GQA 工具函数 ────────────────────────────────

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    将 k/v 的 kv_heads 维度重复 n_rep 次，扩展到 num_heads。
    x: (B, num_kv_heads, T, head_dim)
    返回: (B, num_kv_heads * n_rep, T, head_dim)

    提示：n_rep == 1 时直接返回 x（无 GQA）。
    """
    raise NotImplementedError


# ─────────────────────────── 模块实现 ────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        RMS LayerNorm：x / rms(x) * weight
        提示：
          - 先转 float32 计算，避免数值溢出
          - 思考平方均值、eps、开方/倒数的次序，以及归约维度
          - 返回前转回原始 dtype
        """
        raise NotImplementedError


class KVCache:
    """KV 缓存，用于自回归解码加速。"""

    def __init__(self, cfg: ModelConfig, batch_size: int, max_seq_len: int, device, dtype):
        head_dim = cfg.head_dim
        # 每层各分配一块 zero 缓冲区，shape: (B, num_kv_heads, max_seq_len, head_dim)
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
        """
        将新的 k/v 写入缓存，返回完整的历史 k/v（包含本次新增）。

        k, v: (B, num_kv_heads, T_new, head_dim)
        返回: (k_full, v_full)，shape (B, num_kv_heads, seq_len + T_new, head_dim)

        提示（引导，索引自己写）：
          - 想清楚三个量：本次写入的起始位置、写入区间、返回区间
          - 为什么返回的是「历史 + 本次」而不是只有本次？Attention 里谁需要它？
          - 注意：_seq_len 由 MiniLM.forward 统一在所有层处理完后更新，这里不要改
            ——想想如果每层各自更新会发生什么
        """
        raise NotImplementedError


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

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        """
        GQA Attention（兼容 MHA 和有缓存的解码）。

        x: (B, T, hidden_size)
        返回: (B, T, hidden_size)

        实现步骤：
          1. q/k/v 线性投影，reshape 为 (B, n_heads/n_kv_heads, T, head_dim)，transpose(1,2)
          2. apply_rope(q, k, cos, sin, positions)
          3. 若有 kv_cache：kv_cache.update → 得到完整历史 k, v
          4. repeat_kv 把 k/v 扩展到 num_heads
          5. 手动计算 scaled dot-product attention（禁止调 F.scaled_dot_product_attention，
             它是测试里的对照物），scale = head_dim^(-0.5)
             causal mask 自己推：无 cache 时是标准形状；有 cache 时 q 只有 T 个
             新 token 而 k 有 T_total 个——「第 i 个新 token 能看到哪些位置？」
             想清楚再写，这是 U2.4 cached/non-cached 对齐能不能过的关键
          6. softmax（在 float32 下），matmul with v，reshape，o_proj
        """
        raise NotImplementedError


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        SwiGLU MLP：down(silu(gate(x)) * up(x))
        """
        raise NotImplementedError


class DecoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.norm1 = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = MLP(cfg)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
    ) -> torch.Tensor:
        """
        Pre-norm Transformer 层：
          x = x + attn(norm1(x), ...)
          x = x + mlp(norm2(x))
        """
        raise NotImplementedError


class MiniLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([DecoderLayer(cfg, i) for i in range(cfg.num_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

        # Weight tying：tie_word_embeddings=True 时让 lm_head 与 embedding 共享权重
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.embedding.weight

        # 预计算 RoPE 查找表，注册为 buffer（跟随设备移动，不计入参数）
        cos, sin = precompute_rope(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer('cos', cos, persistent=False)
        self.register_buffer('sin', sin, persistent=False)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """
        GPT-2 风格权重初始化。
        - 普通 Linear：N(0, 0.02)
        - 残差投影（o_proj / down_proj）：N(0, 0.02 / sqrt(2 * num_layers))（深层缩放）
        - Embedding：N(0, 0.02)
        - bias：zeros
        """
        raise NotImplementedError

    def forward(
        self,
        input_ids: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向传播。

        input_ids: (B, T)
        返回: logits (B, T, vocab_size)

        步骤：
          1. embedding lookup
          2. 若 positions 为 None，从 kv_cache.seq_len（或 0）开始构造 arange
          3. 依次过每一层 DecoderLayer
          4. 若有 kv_cache，更新 kv_cache._seq_len += T
          5. 最终 norm → lm_head
        """
        raise NotImplementedError
