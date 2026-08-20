"""Configuration for the Phase 3.2 teaching layer."""

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
