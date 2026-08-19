"""
Phase 1.2：ModelConfig
接口已冻结，学生无需修改此文件中的字段定义。
只需完成 __post_init__ 的 TODO 部分。
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    # 词表与维度
    vocab_size: int = 8192
    hidden_size: int = 256
    intermediate_size: int = 1024
    num_layers: int = 4
    num_heads: int = 8
    num_kv_heads: int = 4          # GQA：kv head 数，< num_heads 时启用 GQA
    head_dim: Optional[int] = None  # None → 自动推断为 hidden_size // num_heads
    max_seq_len: int = 2048
    # RoPE
    rope_theta: float = 10000.0
    # 归一化
    rms_norm_eps: float = 1e-6
    # Attention
    attention_bias: bool = True
    # Embedding
    tie_word_embeddings: bool = True

    def __post_init__(self):
        # TODO: 如果 head_dim 为 None，将其设为 hidden_size // num_heads
        if self.head_dim is None:
            raise NotImplementedError  # 替换这行：实现 head_dim 的自动推断
