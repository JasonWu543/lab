"""minisft/lora.py — LoRA (Low-Rank Adaptation) for any nn.Linear.

学生任务：实现 LoRALinear、inject_lora、merge_lora。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """
    在冻结的 nn.Linear 上叠加低秩适配器：
        output = base(x) + (alpha / r) * B(A(x))

    其中：
        A: (r, in_features)   — 可训练
        B: (out_features, r)  — 可训练
        base.weight / bias    — 冻结

    提示：
    ① 为什么 B 必须零初始化？想想注入瞬间前向必须不变。
    ② A 的初始化用什么分布？标准差应为多少？（论文里有说明）
    ③ 冻结 base 的参数：base.weight.requires_grad_(False)
    """

    def __init__(self, base: nn.Linear, r: int, alpha: float):
        """
        Args:
            base:  被包裹的原始 nn.Linear（将被冻结）
            r:     低秩维度
            alpha: 缩放系数，forward 里乘以 alpha/r
        """
        super().__init__()
        self.base = base
        self.r = r
        self.alpha = alpha
        # TODO: 初始化 self.lora_A 和 self.lora_B（nn.Parameter）
        # TODO: 冻结 base 的 weight 和 bias（如果有）
        raise NotImplementedError("LoRALinear.__init__: 你来实现")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: base(x) + (alpha/r) * B(A(x))
        # 提示：用 F.linear(x, weight) 而不是 nn.Linear 对象来分别做 A 和 B 的线性变换
        raise NotImplementedError("LoRALinear.forward: 你来实现")


def inject_lora(
    model: nn.Module,
    target_names: list[str],
    r: int,
    alpha: float,
) -> int:
    """
    遍历 model 的所有子模块，把名字以 target_names 中某个后缀结尾的 nn.Linear
    替换为 LoRALinear；冻结其他所有参数；返回替换数量。

    遍历模式（可参考）：
        for name, module in list(model.named_modules()):
            # name 是点分路径，如 "model.layers.0.self_attn.q_proj"
            # 用 name.endswith(tname) 判断是否是目标层
            # 用 name.split('.') 定位父模块和属性名

    Args:
        model:        待注入的模型
        target_names: 目标层名后缀列表，如 ["q_proj", "v_proj"]
        r:            LoRA 秩
        alpha:        LoRA 缩放系数

    Returns:
        实际替换的 nn.Linear 数量
    """
    # TODO: 实现
    raise NotImplementedError("inject_lora: 你来实现")


def merge_lora(model: nn.Module) -> None:
    """
    把所有 LoRALinear 的低秩更新合并回 base.weight，
    再把 LoRALinear 替换回普通 nn.Linear。

    合并公式：
        W_merged = W_base + (alpha / r) * B @ A

    提示：用 module.lora_B.data @ module.lora_A.data 计算 delta；
    然后创建新的 nn.Linear，赋值合并后的权重，再 setattr 回父模块。

    Args:
        model: 注入了 LoRA 的模型（原地修改）
    """
    # TODO: 实现
    raise NotImplementedError("merge_lora: 你来实现")
