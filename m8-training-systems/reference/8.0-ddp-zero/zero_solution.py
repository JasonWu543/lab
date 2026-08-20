"""Reference solution for parameter-sharded AdamW and ZeRO-2 semantics."""

from __future__ import annotations

import torch
import torch.distributed as dist

from minidist.bucket import allreduce_gradients


def shard_params(
    params: list[torch.nn.Parameter], world_size: int
) -> list[list[int]]:
    if world_size < 1:
        raise ValueError("world_size must be positive")
    shards = [[] for _ in range(world_size)]
    loads = [0] * world_size
    for index in sorted(range(len(params)), key=lambda i: (-params[i].numel(), i)):
        owner = min(range(world_size), key=lambda rank: (loads[rank], rank))
        shards[owner].append(index)
        loads[owner] += params[index].numel()
    return shards


class Zero1Optimizer:
    def __init__(
        self,
        params: list[torch.nn.Parameter],
        lr: float,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ) -> None:
        self.params = list(params)
        self.rank = dist.get_rank()
        self.shards = shard_params(self.params, dist.get_world_size())
        self.owned = self.shards[self.rank]
        owned_params = [self.params[index] for index in self.owned]
        self.optimizer = (
            torch.optim.AdamW(
                owned_params, lr=lr, betas=betas, eps=eps, weight_decay=0.0
            )
            if owned_params
            else None
        )

    def step(self) -> None:
        if self.optimizer is not None:
            self.optimizer.step()
        owners = {
            index: rank
            for rank, indices in enumerate(self.shards)
            for index in indices
        }
        for index, parameter in enumerate(self.params):
            dist.broadcast(parameter.data, src=owners[index])

    def state_bytes(self) -> int:
        if self.optimizer is None:
            return 0
        return sum(
            value.numel() * value.element_size()
            for state in self.optimizer.state.values()
            for value in state.values()
            if isinstance(value, torch.Tensor)
        )


def zero2_reduce_gradients(model, shards: list[list[int]]) -> None:
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    allreduce_gradients(model, max(1, sum(p.numel() * p.element_size() for p in params)))
    owned = set(shards[dist.get_rank()])
    for index, parameter in enumerate(params):
        if index not in owned:
            parameter.grad = None
