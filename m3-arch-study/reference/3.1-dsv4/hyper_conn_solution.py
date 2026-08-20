"""静态 Hyper-Connection 参考答案。"""

import torch
from torch import nn


class HyperConnection(nn.Module):
    def __init__(self, dim: int, expand_n: int, layer_id: int):
        super().__init__()
        if dim <= 0 or expand_n <= 0:
            raise ValueError("dim and expand_n must be positive")
        self.dim, self.expand_n, self.layer_id = dim, expand_n, layer_id
        # I_n 不改变 stream；读取第一条；每条都写入完整 residual。
        # 因为初始 streams 是复制品，这使每层逐位退化为 Pre-Norm residual。
        self.alpha = nn.Parameter(torch.eye(expand_n))
        read = torch.zeros(expand_n)
        read[0] = 1.0
        self.read_weight = nn.Parameter(read)
        self.beta = nn.Parameter(torch.ones(expand_n))

    def width_mix(self, h):
        return torch.einsum("ij,btjc->btic", self.alpha, h)

    def read(self, h):
        return torch.einsum("i,btic->btc", self.read_weight, h)

    def write(self, h, out):
        return h + self.beta.view(1, 1, -1, 1) * out.unsqueeze(2)


def expand_stream(x, n):
    if n <= 0:
        raise ValueError("n must be positive")
    return x.unsqueeze(2).expand(*x.shape[:2], n, x.shape[-1]).clone()


def collapse_stream(h):
    return h.mean(dim=2)
