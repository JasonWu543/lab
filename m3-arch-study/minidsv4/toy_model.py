"""给定脚手架：可插拔 residual / Hyper-Connection 的 MLP toy 模型。"""

import torch
from torch import nn

from .hyper_conn import HyperConnection, collapse_stream, expand_stream


class ToyBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.up = nn.Linear(dim, dim * 2)
        self.down = nn.Linear(dim * 2, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(torch.nn.functional.gelu(self.up(self.norm(x))))


class ResidualToyModel(nn.Module):
    def __init__(self, dim: int, depth: int):
        super().__init__()
        self.blocks = nn.ModuleList(ToyBlock(dim) for _ in range(depth))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = x + block(x)
        return x


class HCToyModel(nn.Module):
    def __init__(self, dim: int, depth: int, expand_n: int):
        super().__init__()
        self.expand_n = expand_n
        self.blocks = nn.ModuleList(ToyBlock(dim) for _ in range(depth))
        self.connections = nn.ModuleList(
            HyperConnection(dim, expand_n, layer_id) for layer_id in range(depth)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = expand_stream(x, self.expand_n)
        for block, connection in zip(self.blocks, self.connections):
            h = connection.width_mix(h)
            h = connection.write(h, block(connection.read(h)))
        return collapse_stream(h)


def copy_block_weights(source: nn.Module, target: nn.Module) -> None:
    """给定工具：只复制两种 toy 模型共有的 MLP block 参数。"""
    target.blocks.load_state_dict(source.blocks.state_dict())
