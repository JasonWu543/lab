"""Student exercise: gradient-equivalent micro-batch pipeline runner."""

from __future__ import annotations

import torch
from torch import nn


class PipelineRunner:
    def __init__(self, stages: list[nn.Module]):
        raise NotImplementedError("请校验 pipeline partition 并保存 stages")

    def train_step(self, x, y, num_microbatches: int, loss_fn) -> torch.Tensor:
        raise NotImplementedError("micro-batch mean loss 应如何缩放才能保持整批梯度？")
