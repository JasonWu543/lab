"""sampler_solution.py — 温度 / top-p 采样（参考实现）"""
from __future__ import annotations

import torch
from torch import Tensor

from .request_solution import Sequence


def sample(
    logits: Tensor,
    seqs: list[Sequence],
    generator: torch.Generator | None = None,
) -> list[int]:
    """
    按每个 seq 自己的 temperature / top_p 参数采样。
    logits: shape [len(seqs), vocab_size]（已取出各 seq 对应的最后一步 logit）。
    返回长度为 len(seqs) 的 token id 列表。
    """
    assert logits.shape[0] == len(seqs), "logits batch size must match seqs length"
    results: list[int] = []

    for i, seq in enumerate(seqs):
        logit = logits[i]          # [vocab_size]
        temperature = seq.request.temperature
        top_p = seq.request.top_p

        # ── greedy ───────────────────────────────────────────────────────
        if temperature == 0.0:
            token_id = int(logit.argmax().item())
            results.append(token_id)
            continue

        # ── 温度缩放 ──────────────────────────────────────────────────────
        logit = logit / temperature
        probs = torch.softmax(logit, dim=-1)

        # ── top-p 截断（nucleus sampling）─────────────────────────────────
        if top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            # 找到累积概率超过 top_p 的位置，之后的都截掉
            # 保留至少 1 个 token（shift by 1 再 mask）
            remove_mask = cumsum - sorted_probs > top_p
            sorted_probs[remove_mask] = 0.0
            # 归一化
            sorted_probs = sorted_probs / sorted_probs.sum()
            # 重新映射到原始词汇表顺序
            probs = torch.zeros_like(probs)
            probs.scatter_(0, sorted_indices, sorted_probs)

        # ── 多项式采样 ────────────────────────────────────────────────────
        token_id = int(
            torch.multinomial(probs, num_samples=1, generator=generator).item()
        )
        results.append(token_id)

    return results
