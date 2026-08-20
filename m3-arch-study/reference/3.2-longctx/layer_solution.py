"""参考答案：带固定状态增量解码的 DeltaAttention。"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from minikda.config import KDAConfig
from minikda.delta import delta_rule_chunked


class DeltaAttention(nn.Module):
    def __init__(self, cfg: KDAConfig):
        super().__init__()
        self.cfg = cfg
        self.q_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.beta_proj = nn.Linear(cfg.hidden_size, cfg.num_heads, bias=True)
        self.alpha_proj = nn.Linear(cfg.hidden_size, cfg.num_heads, bias=True)
        self.o_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)

    def _project(self, x):
        b, t, _ = x.shape
        shape = (b, t, self.cfg.num_heads, self.cfg.head_dim)
        q = self.q_proj(x).view(shape).transpose(1, 2)
        k = self.k_proj(x).view(shape).transpose(1, 2)
        v = self.v_proj(x).view(shape).transpose(1, 2)
        beta = self.beta_proj(x).sigmoid().transpose(1, 2)
        alpha = self.alpha_proj(x).sigmoid().transpose(1, 2)
        return q, k, v, beta, alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v, beta, alpha = self._project(x)
        y = delta_rule_chunked(q, k, v, beta, alpha, self.cfg.chunk_size)
        return self.o_proj(y.transpose(1, 2).reshape_as(x))

    def step(self, x_t, state):
        """Rank-1 单步与递推公式相同；返回更新后读取，和 forward 对齐。"""
        if x_t.ndim == 2:
            x_t = x_t.unsqueeze(1)
        if x_t.ndim != 3 or x_t.shape[1] != 1:
            raise ValueError("x_t must have shape (B,1,H) or (B,H)")
        q, k, v, beta, alpha = self._project(x_t)
        q, k, v = q[:, :, 0].float(), k[:, :, 0].float(), v[:, :, 0].float()
        k = F.normalize(k, dim=-1)
        beta, alpha = beta[:, :, 0].float(), alpha[:, :, 0].float()
        b = x_t.shape[0]
        if state is None:
            state = torch.zeros(
                b, self.cfg.num_heads, self.cfg.head_dim, self.cfg.head_dim,
                device=x_t.device, dtype=torch.float32,
            )
        expected = (b, self.cfg.num_heads, self.cfg.head_dim, self.cfg.head_dim)
        if state.shape != expected:
            raise ValueError(f"state must have shape {expected}")
        old = torch.einsum("bhvd,bhd->bhv", state.float(), k)
        correction = beta[..., None] * (v - alpha[..., None] * old)
        state = alpha[..., None, None] * state.float()
        state = state + correction.unsqueeze(-1) * k.unsqueeze(-2)
        y = torch.einsum("bhvd,bhd->bhv", state, q)
        y = y.reshape(b, 1, self.cfg.hidden_size).to(x_t.dtype)
        return self.o_proj(y), state


def state_bytes(cfg) -> int:
    """每个 head 一个 D×D fp32 状态；单请求没有 batch/T 因子。"""
    # 核心定义要求 fp32 累加，因此 decode 状态也固定为 fp32。
    return cfg.num_heads * cfg.head_dim * cfg.head_dim * torch.tensor([], dtype=torch.float32).element_size()
