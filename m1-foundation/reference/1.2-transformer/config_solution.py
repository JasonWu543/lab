"""
Phase 1.2 参考答案：ModelConfig
学生文件：minilm/model/config.py
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    vocab_size: int = 8192
    hidden_size: int = 256
    intermediate_size: int = 1024
    num_layers: int = 4
    num_heads: int = 8
    num_kv_heads: int = 4
    head_dim: Optional[int] = None  # None → hidden_size // num_heads
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    attention_bias: bool = True
    tie_word_embeddings: bool = True

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_heads
