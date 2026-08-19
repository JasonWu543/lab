"""参考实现：等 activated FLOPs 的 config 生成器。

FFN 每 token 的 activated 参数（近似 FLOPs）：
- MoE ：(n_shared + top_k) 个 SwiGLU，每个 3 * H * moe_intermediate_size
- Dense：1 个 SwiGLU，3 * H * intermediate_size
"""
from __future__ import annotations

from dataclasses import replace

try:
    from .config_solution import MoEConfig
except ImportError:
    from config_solution import MoEConfig


def activated_ffn_params(cfg: MoEConfig, moe: bool) -> int:
    H = cfg.hidden_size
    if moe:
        n_active = cfg.n_shared_experts + cfg.top_k
        return n_active * 3 * H * cfg.moe_intermediate_size
    return 3 * H * cfg.intermediate_size


def dense_config_matching_flops(cfg: MoEConfig) -> MoEConfig:
    H = cfg.hidden_size
    M = activated_ffn_params(cfg, moe=True)
    d = round(M / (3 * H))
    return replace(cfg, intermediate_size=d)
