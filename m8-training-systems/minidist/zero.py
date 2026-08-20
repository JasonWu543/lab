"""ZeRO-1/2 semantic exercise skeleton."""

from __future__ import annotations

import torch


def shard_params(
    params: list[torch.nn.Parameter], world_size: int
) -> list[list[int]]:
    """Assign whole parameters by deterministic greedy load balancing."""
    raise NotImplementedError("怎样按参数大小做确定且负载均衡的 ownership 分配？")


class Zero1Optimizer:
    """AdamW with optimizer state sharded by whole parameters."""

    def __init__(
        self,
        params: list[torch.nn.Parameter],
        lr: float,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ) -> None:
        raise NotImplementedError("本 rank 应为哪些参数建立 AdamW 状态？")

    def step(self) -> None:
        raise NotImplementedError("owner 更新后，怎样让所有 rank 得到一致参数？")

    def state_bytes(self) -> int:
        raise NotImplementedError("哪些本地 optimizer tensor 应计入状态字节数？")


def zero2_reduce_gradients(model, shards: list[list[int]]) -> None:
    """Average all gradients, retaining only this rank's parameter shard."""
    raise NotImplementedError("平均梯度后，非 owner 的梯度应怎样释放？")
