"""Phase 1.1 — nn 模块骨架（学生实现文件）

在自己实现的 Tensor 之上构建 Module/Linear/RMSNorm/SwiGLU。

实现顺序建议：
  1. Module 基类（parameters() 递归收集子模块的参数）
  2. Linear（权重初始化 + 前向 xW^T + b）
  3. RMSNorm（公式：x / sqrt(mean(x^2, dim=-1, keepdim=True) + eps) * weight）
  4. SwiGLU（gate = silu(W1 x)，value = W3 x，output = W2(gate * value)，无 bias）

运行测试：
  python3 -m pytest tests/test_tensor.py -k T8 -q
"""
from __future__ import annotations

import numpy as np

from minilm.tensor.tensor import Tensor


class Module:
    """神经网络模块基类。

    __call__ 转发到 forward，parameters() 递归收集所有 Tensor 叶节点。
    """

    def __call__(self, *args, **kwargs):
        """转发到 forward。"""
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def parameters(self) -> list[Tensor]:
        """递归收集所有子模块（Module 属性）及直接 Tensor 属性的参数列表。

        提示：
          遍历 self.__dict__.values()：
          - 若是 Tensor 且 requires_grad=True → 收集
          - 若是 Module → 递归调用 .parameters()
          - 若是 list/tuple → 对每个元素判断以上两种情况
        """
        raise NotImplementedError


class Linear(Module):
    """全连接层：y = x @ W.T + b。

    参数：
        in_features  : 输入维度
        out_features : 输出维度
        bias         : 是否使用偏置（默认 True）

    提示（权重初始化）：
        rng = np.random.default_rng(seed)      # 或用全局 rng
        scale = 1.0 / np.sqrt(in_features)     # Kaiming uniform 近似
        W = rng.uniform(-scale, scale, (out_features, in_features))
        self.weight = Tensor(W, requires_grad=True)
        # bias 类似，shape=(out_features,)，初始化为 0 或小随机数
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        """前向：x @ weight.T + bias（若有）。

        提示：x.shape = (..., in_features)，结果 shape = (..., out_features)
        """
        raise NotImplementedError


class RMSNorm(Module):
    """RMS 归一化：output = x / sqrt(mean(x^2, dim=-1, keepdim=True) + eps) * weight。

    参数：
        dim : 特征维度
        eps : 数值稳定项（默认 1e-6）

    提示：
        self.weight = Tensor(np.ones(dim), requires_grad=True)
        注意 mean 对 dim=-1，keepdim=True；结果再乘以 weight（broadcast）
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


class SwiGLU(Module):
    """SwiGLU 激活块（Shazeer 2020）。

    结构：output = W2( silu(W1 x) * (W3 x) )，三个 Linear 均无 bias。

    参数：
        dim        : 输入/输出维度
        hidden_dim : 中间维度（W1/W3 输出维度，W2 输入维度）

    提示：
        silu(x) = x * sigmoid(x)，可用你实现的 Tensor.sigmoid() 来写
        self.w1 = Linear(dim, hidden_dim, bias=False)
        self.w2 = Linear(hidden_dim, dim, bias=False)
        self.w3 = Linear(dim, hidden_dim, bias=False)
    """

    def __init__(self, dim: int, hidden_dim: int):
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError
