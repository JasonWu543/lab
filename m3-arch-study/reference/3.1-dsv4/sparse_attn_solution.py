"""Block sparse attention 参考答案。"""

from dataclasses import dataclass
import math

import torch


@dataclass
class SparseAttnConfig:
    block_size: int
    top_k: int
    local_blocks: int
    sink_blocks: int = 1


def build_block_mask(scores_qk_block, cfg, q_pos):
    """top-k 只在严格过去的块上做，当前块无条件保留。

    关键因果细节：当前块的 pooled 分数含未来 K（mean-pool 越过了 t），
    若让它参与 top-k 竞争，被污染的分数可以把某个过去块挤出选择集，
    使过去块的取舍依赖未来 K——输出泄漏。因此当前块必须排除出 gating、
    直接强制保留（SPEC 定义 local 窗口本就含当前块）。"""
    if scores_qk_block.ndim != 4:
        raise ValueError("scores_qk_block must have shape (B,H,Tq,N_blocks)")
    n_blocks = scores_qk_block.shape[-1]
    current = torch.div(q_pos.to(scores_qk_block.device), cfg.block_size,
                        rounding_mode="floor")
    if current.ndim != 1 or current.numel() != scores_qk_block.shape[2]:
        raise ValueError("q_pos must have shape (Tq,)")
    block_ids = torch.arange(n_blocks, device=scores_qk_block.device)
    causal = block_ids.view(1, -1) <= current.view(-1, 1)
    forced = (block_ids.view(1, -1) < cfg.sink_blocks)
    local_start = (current - cfg.local_blocks + 1).clamp_min(0)
    forced = forced | (
        (block_ids.view(1, -1) >= local_start.view(-1, 1))
        & causal
    )
    forced = forced | (block_ids.view(1, -1) == current.view(-1, 1))
    forced &= causal

    strictly_past = block_ids.view(1, -1) < current.view(-1, 1)
    masked_scores = scores_qk_block.masked_fill(
        ~strictly_past.view(1, 1, *strictly_past.shape), float("-inf")
    )
    k = min(max(cfg.top_k, 0), n_blocks)
    selected = torch.zeros_like(scores_qk_block, dtype=torch.bool)
    if k:
        indices = masked_scores.topk(k, dim=-1).indices
        selected.scatter_(-1, indices, True)
        # topk may have returned -inf entries if fewer than k past blocks exist.
        selected &= strictly_past.view(1, 1, *strictly_past.shape)
    return selected | forced.view(1, 1, *forced.shape)


def sparse_attention(q, k, v, cfg):
    """Dense 地算分数但在 softmax 前施加独立构造的 token mask。"""
    if q.shape != k.shape or q.shape != v.shape or q.ndim != 4:
        raise ValueError("q, k and v must share shape (B,H,T,D)")
    B, H, T, D = q.shape
    n_blocks = math.ceil(T / cfg.block_size)
    pad = n_blocks * cfg.block_size - T
    k_for_pool = torch.nn.functional.pad(k, (0, 0, 0, pad))
    pooled = k_for_pool.reshape(B, H, n_blocks, cfg.block_size, D).mean(dim=3)
    block_scores = torch.einsum("bhtd,bhnd->bhtn", q, pooled)
    q_pos = torch.arange(T, device=q.device)
    block_mask = build_block_mask(block_scores, cfg, q_pos)
    token_blocks = torch.arange(T, device=q.device) // cfg.block_size
    token_mask = block_mask[..., token_blocks]
    causal = torch.arange(T, device=q.device).view(1, T) <= q_pos.view(T, 1)
    allowed = token_mask & causal.view(1, 1, T, T)
    logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(D)
    logits = logits.masked_fill(~allowed, float("-inf"))
    return torch.matmul(torch.softmax(logits, dim=-1), v)


def sparse_attn_flops(T, D, cfg):
    """QK 和 AV 各需一次乘加，故每个所选 token 共 4D FLOPs。"""
    if T <= 0 or D <= 0:
        raise ValueError("T and D must be positive")
    n_blocks = math.ceil(T / cfg.block_size)
    selected_blocks = min(n_blocks, cfg.top_k + cfg.local_blocks + cfg.sink_blocks)
    selected_tokens = min(T, selected_blocks * cfg.block_size)
    return 4 * selected_tokens * D
