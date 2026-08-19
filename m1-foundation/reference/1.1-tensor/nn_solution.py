"""Phase 1.1 — nn 模块参考答案（教师版）

Linear 初始化：Kaiming uniform，scale = 1/sqrt(in_features)。
RMSNorm：公式 x / sqrt(mean(x^2, dim=-1, keepdim=True) + eps) * weight。
SwiGLU：gate = silu(W1 x)，value = W3 x，output = W2(gate * value)，无 bias。
"""
from __future__ import annotations

import numpy as np

# 注意：在验证脚本中这里会被 monkeypatch 替换为参考答案版 Tensor
from minilm.tensor.tensor import Tensor


class Module:
    """神经网络模块基类。"""

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def parameters(self) -> list[Tensor]:
        """递归收集所有参数。"""
        params = []
        for v in self.__dict__.values():
            if isinstance(v, Tensor) and v.requires_grad:
                params.append(v)
            elif isinstance(v, Module):
                params.extend(v.parameters())
            elif isinstance(v, (list, tuple)):
                for item in v:
                    if isinstance(item, Tensor) and item.requires_grad:
                        params.append(item)
                    elif isinstance(item, Module):
                        params.extend(item.parameters())
        return params


class Linear(Module):
    """全连接层：y = x @ W.T + b。"""

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 seed: int | None = None):
        rng = np.random.default_rng(seed)
        scale = 1.0 / np.sqrt(in_features)
        W = rng.uniform(-scale, scale, (out_features, in_features))
        self.weight = Tensor(W, requires_grad=True)
        self._use_bias = bias
        if bias:
            # 偏置初始化为 0（常见做法）
            b = np.zeros(out_features, dtype=np.float64)
            self.bias = Tensor(b, requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        # x: (..., in_features)  →  (..., out_features)
        # x @ W.T 等价于 matmul(x, weight.T)
        result = x @ self.weight.transpose(0, 1)
        if self._use_bias:
            result = result + self.bias
        return result


class RMSNorm(Module):
    """RMS 归一化：output = x / sqrt(mean(x^2, -1, keepdim=True) + eps) * weight。"""

    def __init__(self, dim: int, eps: float = 1e-6):
        self.eps = eps
        self.weight = Tensor(np.ones(dim, dtype=np.float64), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        # x^2 的均值（沿最后一维，保留维度）
        rms = (x ** 2).mean(dim=-1, keepdim=True)
        # x / sqrt(rms + eps)
        normed = x / (rms + self.eps) ** 0.5
        # 乘以可学习缩放参数（broadcast）
        return normed * self.weight


class SwiGLU(Module):
    """SwiGLU：output = W2(silu(W1 x) * (W3 x))，无 bias。"""

    def __init__(self, dim: int, hidden_dim: int, seed: int | None = None):
        self.w1 = Linear(dim, hidden_dim, bias=False, seed=seed)
        self.w2 = Linear(hidden_dim, dim, bias=False,
                         seed=None if seed is None else seed + 1)
        self.w3 = Linear(dim, hidden_dim, bias=False,
                         seed=None if seed is None else seed + 2)

    def forward(self, x: Tensor) -> Tensor:
        # gate = silu(W1 x) = W1x * sigmoid(W1x)
        h1 = self.w1(x)
        gate = h1 * h1.sigmoid()   # SiLU = x * σ(x)
        # value = W3 x
        value = self.w3(x)
        # output = W2(gate * value)
        return self.w2(gate * value)
