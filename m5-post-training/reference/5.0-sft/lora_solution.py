"""Reference solution for minisft/lora.py — do not read until stuck for 30+ min."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear and adds a low-rank update (alpha/r) * B @ A."""

    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.base = base
        self.r = r
        self.alpha = alpha
        in_features = base.in_features
        out_features = base.out_features

        # A: (r, in) initialised N(0, 1/r)  →  std = 1/sqrt(r)
        self.lora_A = nn.Parameter(torch.randn(r, in_features) / (r ** 0.5))
        # B: (out, r) zero-initialised so that the adapter starts as identity
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        # Freeze base weights
        base.weight.requires_grad_(False)
        if base.bias is not None:
            base.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # base output + scaled low-rank update
        return self.base(x) + (self.alpha / self.r) * F.linear(F.linear(x, self.lora_A), self.lora_B)


def inject_lora(model: nn.Module, target_names: list[str], r: int, alpha: float) -> int:
    """
    Replace every nn.Linear whose attribute name ends with one of target_names
    with a LoRALinear. Freeze all non-LoRA parameters. Return replacement count.
    """
    count = 0
    # Collect (name, module) first to avoid mutation during iteration
    named = list(model.named_modules())
    for name, module in named:
        for tname in target_names:
            if name.endswith(tname) and isinstance(module, nn.Linear):
                parts = name.split(".")
                parent = model
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                attr = parts[-1]
                setattr(parent, attr, LoRALinear(module, r, alpha))
                count += 1
                break  # avoid double-replacing if name matches multiple target_names

    # Freeze everything except LoRA parameters
    for name, param in model.named_parameters():
        if "lora_A" not in name and "lora_B" not in name:
            param.requires_grad_(False)

    return count


def merge_lora(model: nn.Module) -> None:
    """
    Fold all LoRALinear adapters back into their base weights,
    replacing each LoRALinear with a plain nn.Linear.
    """
    named = list(model.named_modules())
    for name, module in named:
        if isinstance(module, LoRALinear):
            # delta W = (alpha / r) * B @ A
            delta = (module.alpha / module.r) * module.lora_B.data @ module.lora_A.data
            has_bias = module.base.bias is not None
            new_linear = nn.Linear(
                module.base.in_features, module.base.out_features, bias=has_bias
            )
            new_linear.weight.data = module.base.weight.data + delta
            if has_bias:
                new_linear.bias.data = module.base.bias.data

            parts = name.split(".")
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], new_linear)
