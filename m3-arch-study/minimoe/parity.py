"""学生任务：等 activated FLOPs 的 config 生成器。

闭式手算，几行代码。重点是想清楚「每 token 实际参与计算的 FFN 参数」怎么数。

思考题：MoE 一个 token 只激活 (n_shared + top_k) 个小 expert，
        怎样的 dense intermediate_size 才「等算力」？
"""
from __future__ import annotations

from dataclasses import replace

from minimoe.config import MoEConfig


def activated_ffn_params(cfg: MoEConfig, moe: bool) -> int:
    """每 token 实际参与计算的 FFN 参数量（≈ FLOPs 的代理量）。

    提示：一个 SwiGLU 有 gate/up/down 三个矩阵，参数量约 3 * H * inter。
      moe=True ：一个 token 激活 n_shared_experts + top_k 个宽度 moe_intermediate_size 的 expert
      moe=False：一个宽度 intermediate_size 的 dense SwiGLU
    """
    # TODO(学生): 返回激活 FFN 参数量
    raise NotImplementedError


def dense_config_matching_flops(cfg: MoEConfig) -> MoEConfig:
    """给定 MoE config，返回 activated FLOPs 相等的 Dense config（只调 intermediate_size）。

    提示：令 M = activated_ffn_params(cfg, moe=True)，解 3*H*d = M 得 d，round 成 int，
    用 dataclasses.replace 生成新 cfg。
    """
    # TODO(学生): 反解 intermediate_size 并返回新 cfg
    raise NotImplementedError
