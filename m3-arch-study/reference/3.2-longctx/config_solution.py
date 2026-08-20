"""参考答案沿用公开配置；复制覆盖时保持相同接口。"""

from dataclasses import dataclass

import torch


@dataclass
class KDAConfig:
    hidden_size: int = 64
    num_heads: int = 4
    head_dim: int = 16
    chunk_size: int = 16
    dtype: torch.dtype = torch.float32

    def __post_init__(self):
        if self.hidden_size != self.num_heads * self.head_dim:
            raise ValueError("hidden_size must equal num_heads * head_dim")
