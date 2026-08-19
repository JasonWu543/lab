"""学生任务：Router（aux-loss-free 均衡）+ SwiGLU expert + MoELayer（dropless）。

核心算法归学生手写。脚手架（gather-scatter 框架、shape 注释）已给出。

思考题（做完对照 SPEC §1）：
- bias 为什么只影响「选择」不影响 gating 权重？如果 bias 也进了权重会怎样？
- 细粒度 expert（每个很小）+ shared expert 各自解决什么问题？
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from minimoe.config import MoEConfig


class SwiGLU(nn.Module):
    """SwiGLU FFN expert: output = down(silu(gate(x)) * up(x))。

    子模块（bias=False）：
      gate: Linear(hidden_size, intermediate_size)
      up  : Linear(hidden_size, intermediate_size)
      down: Linear(intermediate_size, hidden_size)
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        # TODO(学生): 定义 gate / up / down 三个 Linear（bias=False）
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO(学生): 实现 down(silu(gate(x)) * up(x))
        raise NotImplementedError


class Router(nn.Module):
    """aux-loss-free 路由器。

    结构提示：
      self.weight : nn.Parameter, shape (n_experts, hidden_size)，用 kaiming 初始化
      self.bias   : buffer（register_buffer，**不是 Parameter**），shape (n_experts,)，初值全 0
                    -> buffer 才能保证 bias 无梯度（T3 会检查 .grad is None / requires_grad False）

    forward 流程提示：
      logits = x @ weight.T            # (N, n_experts)
      score  = sigmoid(logits)         # (N, n_experts)
      选 top-k 得到 topk_idx (N, top_k)
      gating 权重 topk_weight (N, top_k)，每 token 归一化到和为 1

    RED LINE（自己想清楚，不在注释里给答案）：
      - 选 top-k 时用 score 还是 score+bias？
      - 算 gating 权重时在哪个量上归一化？
    """

    def __init__(self, cfg: MoEConfig):
        super().__init__()
        self.cfg = cfg
        self.n_experts = cfg.n_routed_experts
        self.top_k = cfg.top_k
        self.gamma = cfg.bias_update_speed
        # TODO(学生): 定义 self.weight (Parameter) 与 self.bias (buffer)
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (N, H) -> 返回 (topk_idx (N, top_k) long, topk_weight (N, top_k) float)
        # TODO(学生): 计算 score；选 top-k；算归一化 gating 权重
        raise NotImplementedError

    @torch.no_grad()
    def update_bias(self, topk_idx: torch.Tensor) -> None:
        """aux-loss-free 均衡：统计本 batch 各 expert 负载，
        与均值比较后调整 bias（in-place 更新 self.bias）。

        提示：负载 = 各 expert 被选中的 token 计数（bincount）；均值 = 总数 / n_experts。
        RED LINE：过载 / 欠载 分别把 bias 往哪个方向调，自己推。
        """
        # TODO(学生): 统计负载并更新 self.bias
        raise NotImplementedError


class MoELayer(nn.Module):
    """fine-grained routed experts + always-on shared experts，dropless。

    属性：
      self.router : Router
      self.experts: nn.ModuleList[SwiGLU]，共 n_routed_experts 个，宽度 moe_intermediate_size
      self.shared : SwiGLU，宽度 moe_intermediate_size * n_shared_experts（恒定参与）

    dropless gather-scatter 框架（已给结构，学生填计算）：
      1. flatten x -> x_flat (N, H)
      2. router -> topk_idx, topk_weight；训练时调 update_bias
      3. 逐 expert：找命中它的 token -> gather -> expert 前向 -> 按 gating 权重 scatter-add
      4. 加 shared expert 输出；reshape 回 (B, T, H)
    """

    def __init__(self, cfg: MoEConfig):
        super().__init__()
        self.cfg = cfg
        # TODO(学生): 构造 self.router / self.experts / self.shared
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, H) -> (B, T, H)
        # 框架参考：
        #   B, T, H = x.shape
        #   x_flat = x.reshape(-1, H)                 # (N, H)
        #   topk_idx, topk_weight = self.router(x_flat)
        #   if self.training: self.router.update_bias(topk_idx)
        #   out = torch.zeros_like(x_flat)
        #   for e in range(n_routed_experts):
        #       找出 top_k 里命中 expert e 的 token（注意 top_k 有多个 slot）
        #       gather 这些 token -> 过 self.experts[e] -> 乘对应 gating 权重 -> index_add_ 回 out
        #       空 expert（无 token 命中）必须跳过，不能产生 NaN
        #   out = out + self.shared(x_flat)
        #   return out.reshape(B, T, H)
        # TODO(学生): 实现 dropless MoE 前向
        raise NotImplementedError
