"""给定脚手架：可复现的 copy 与 associative-recall 合成数据。"""

from __future__ import annotations

import torch


def generate_copy_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    *,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """返回随机 token 及其逐位置 copy 标签。"""
    tokens = torch.randint(
        vocab_size, (batch_size, seq_len), device=device, generator=generator
    )
    return tokens, tokens.clone()


def generate_associative_recall_batch(
    batch_size: int,
    num_pairs: int,
    vocab_size: int,
    *,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """生成 ``k1,v1,...,kn,vn,q``，标签为 q 对应的 value。

    key 使用 ``[0, vocab_size)``，value 使用不相交的
    ``[vocab_size, 2*vocab_size)``，从而显式保留 token 角色。每个样本内 key
    不重复；query 从已出现的 key 中均匀选择。
    """
    if not (1 <= num_pairs <= vocab_size):
        raise ValueError("num_pairs must be in [1, vocab_size]")
    scores = torch.rand(batch_size, vocab_size, device=device, generator=generator)
    keys = scores.argsort(dim=-1)[:, :num_pairs]
    values = torch.randint(
        vocab_size, (batch_size, num_pairs), device=device, generator=generator
    )
    query_slot = torch.randint(
        num_pairs, (batch_size,), device=device, generator=generator
    )
    rows = torch.arange(batch_size, device=device)
    query = keys[rows, query_slot]
    target = values[rows, query_slot]
    sequence = torch.empty(
        batch_size, 2 * num_pairs + 1, dtype=torch.long, device=device
    )
    sequence[:, 0 : 2 * num_pairs : 2] = keys
    sequence[:, 1 : 2 * num_pairs : 2] = values + vocab_size
    sequence[:, -1] = query
    return sequence, target
