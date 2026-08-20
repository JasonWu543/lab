"""Reference implementation of phase 8.1 tensor parallel layers."""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F


def _check_partition(size: int, world_size: int, rank: int, name: str) -> None:
    if world_size < 1:
        raise ValueError("world_size must be positive")
    if not 0 <= rank < world_size:
        raise ValueError("rank must be in [0, world_size)")
    if size % world_size:
        raise ValueError(f"{name} must be divisible by world_size")


def _full_weight(out_features: int, in_features: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    # A conventional small MLP initialization also keeps the frozen 1e-6 TP
    # summation-order tolerance meaningful on CPU gloo.
    return torch.randn(out_features, in_features, generator=generator) * 0.02


class _CopyToTP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_input = grad_output.contiguous()
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(grad_input, op=dist.ReduceOp.SUM)
        return grad_input


class _ReduceFromTP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        output = x.contiguous()
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(output, op=dist.ReduceOp.SUM)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output


class ColumnParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, world_size, rank, seed):
        super().__init__()
        _check_partition(out_features, world_size, rank, "out_features")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.world_size = int(world_size)
        self.rank = int(rank)
        local_out = out_features // world_size
        full = _full_weight(out_features, in_features, seed)
        self.weight = nn.Parameter(full.narrow(0, rank * local_out, local_out).clone())

    def forward(self, x):
        return F.linear(_CopyToTP.apply(x), self.weight)


class RowParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, world_size, rank, seed):
        super().__init__()
        _check_partition(in_features, world_size, rank, "in_features")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.world_size = int(world_size)
        self.rank = int(rank)
        local_in = in_features // world_size
        full = _full_weight(out_features, in_features, seed)
        self.weight = nn.Parameter(full.narrow(1, rank * local_in, local_in).clone())

    def forward(self, x):
        return _ReduceFromTP.apply(F.linear(x, self.weight))


class TPMLP(nn.Module):
    def __init__(self, dim: int, world_size: int, rank: int, seed: int):
        super().__init__()
        self.column = ColumnParallelLinear(dim, 4 * dim, world_size, rank, seed)
        self.row = RowParallelLinear(4 * dim, dim, world_size, rank, seed + 1)

    def forward(self, x):
        return self.row(F.gelu(self.column(x)))
