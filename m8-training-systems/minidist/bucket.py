"""Gradient-bucket exercise skeleton."""

from __future__ import annotations

import torch


def partition_buckets(
    params: list[torch.nn.Parameter], bucket_cap_bytes: int
) -> list[list[int]]:
    """Partition parameters into reverse-order greedy buckets."""
    raise NotImplementedError("反向遍历时，何时应封桶并开始下一个 bucket？")


def allreduce_gradients(model: torch.nn.Module, bucket_cap_bytes: int) -> None:
    """Flatten, average, and scatter back one all-reduce per bucket."""
    raise NotImplementedError("如何把每桶梯度展平、通信并安全写回各参数？")
