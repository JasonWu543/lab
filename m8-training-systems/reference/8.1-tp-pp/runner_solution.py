"""Reference implementation of the single-process pipeline runner."""

from __future__ import annotations

import torch
from torch import nn


class PipelineRunner:
    def __init__(self, stages: list[nn.Module]):
        if not stages:
            raise ValueError("at least one stage is required")
        if not all(isinstance(stage, nn.Module) for stage in stages):
            raise TypeError("every stage must be an nn.Module")
        if any(isinstance(stage, nn.Sequential) and len(stage) == 0 for stage in stages):
            raise ValueError("number of stages cannot exceed number of layers")
        self.stages = list(stages)

    def train_step(self, x, y, num_microbatches: int, loss_fn) -> torch.Tensor:
        if isinstance(num_microbatches, bool) or not isinstance(num_microbatches, int):
            raise ValueError("num_microbatches must be a positive integer")
        if num_microbatches < 1:
            raise ValueError("num_microbatches must be a positive integer")
        if len(x) != len(y):
            raise ValueError("x and y must have equal batch size")
        if len(x) == 0 or len(x) % num_microbatches:
            raise ValueError("batch size must be nonzero and divisible by num_microbatches")

        x_chunks = x.chunk(num_microbatches, dim=0)
        y_chunks = y.chunk(num_microbatches, dim=0)
        losses = []
        for x_micro, y_micro in zip(x_chunks, y_chunks):
            activation = x_micro
            for stage in self.stages:
                activation = stage(activation)
            micro_loss = loss_fn(activation, y_micro)
            if micro_loss.ndim:
                micro_loss = micro_loss.mean()
            (micro_loss / num_microbatches).backward()
            losses.append(micro_loss.detach())
        return torch.stack(losses).mean()
