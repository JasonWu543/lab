"""Reference solution for reverse gradient buckets."""

from __future__ import annotations

import torch
import torch.distributed as dist


def partition_buckets(
    params: list[torch.nn.Parameter], bucket_cap_bytes: int
) -> list[list[int]]:
    if bucket_cap_bytes <= 0:
        raise ValueError("bucket_cap_bytes must be positive")
    buckets: list[list[int]] = []
    current: list[int] = []
    current_bytes = 0
    for index in range(len(params) - 1, -1, -1):
        size = params[index].numel() * params[index].element_size()
        if current and current_bytes + size > bucket_cap_bytes:
            buckets.append(current)
            current = []
            current_bytes = 0
        current.append(index)
        current_bytes += size
        if size > bucket_cap_bytes:
            buckets.append(current)
            current = []
            current_bytes = 0
    if current:
        buckets.append(current)
    return buckets


def allreduce_gradients(model: torch.nn.Module, bucket_cap_bytes: int) -> None:
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    world_size = dist.get_world_size()
    for bucket in partition_buckets(params, bucket_cap_bytes):
        flat = torch.cat(
            [
                parameter.grad.detach().reshape(-1)
                if parameter.grad is not None
                else torch.zeros_like(parameter).reshape(-1)
                for parameter in (params[index] for index in bucket)
            ]
        )
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        flat.div_(world_size)
        offset = 0
        for index in bucket:
            parameter = params[index]
            count = parameter.numel()
            value = flat[offset : offset + count].view_as(parameter)
            if parameter.grad is None:
                parameter.grad = value.clone()
            else:
                parameter.grad.copy_(value)
            offset += count
