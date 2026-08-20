"""cache_adapter_solution.py — 分页 KV 与 transformers DynamicCache 的适配层（参考实现）

物理存储 shape: (num_layers, 2, num_blocks, block_size, n_kv_heads, head_dim)

gather 方向：按 seq.block_table 拼出连续 KV，组装成 DynamicCache。
             物理拷贝——真 vLLM 靠 paged attention kernel 原地读（差距写进 POSTMORTEM）。

scatter 方向：把新一步产出的 KV（DynamicCache 最新条目）写回分页存储。

DynamicCache API（transformers==4.52.4）：
  cache = DynamicCache()
  cache.update(key_states, value_states, layer_idx)
  # key_states: [batch, n_kv_heads, seq_len, head_dim]
  # cache.key_cache[l]: [batch, n_kv_heads, total_len, head_dim]
"""
from __future__ import annotations

import torch
from transformers import DynamicCache

from .request_solution import Sequence, SeqStatus


def gather_past_kv(
    paged_kv: torch.Tensor,   # [num_layers, 2, num_blocks, block_size, n_kv_heads, head_dim]
    seqs: list[Sequence],
) -> DynamicCache:
    """逐 seq 调用（Engine 每次传 [seq]），按 block_table gather 出历史 KV。"""
    assert len(seqs) == 1, "本实现每次只处理单条 seq"
    seq = seqs[0]
    clen = _cached_kv_len(seq)

    num_layers = paged_kv.shape[0]
    cache = DynamicCache()
    if clen == 0:
        return cache

    for layer_idx in range(num_layers):
        k_seq, v_seq = _gather_seq_kv(paged_kv, layer_idx, seq, clen)
        # [n_kv_heads, clen, head_dim] → [1, n_kv_heads, clen, head_dim]
        cache.update(k_seq.unsqueeze(0), v_seq.unsqueeze(0), layer_idx)

    return cache


def scatter_new_kv(
    paged_kv: torch.Tensor,
    new_cache: DynamicCache,
    seqs: list[Sequence],
    new_token_positions: list[int],
) -> None:
    """把 new_cache 中本步新增的 KV 写回 paged_kv。
    new_token_positions[0] = 本步前已有的 KV token 数（gather 时的 clen）。
    """
    assert len(seqs) == 1
    seq = seqs[0]
    start_pos = new_token_positions[0]
    num_layers = paged_kv.shape[0]

    for layer_idx in range(num_layers):
        k_all = new_cache.key_cache[layer_idx]    # [1, n_kv_heads, total_len, head_dim]
        v_all = new_cache.value_cache[layer_idx]
        num_new = k_all.shape[2] - start_pos
        if num_new <= 0:
            continue
        k_new = k_all[0, :, start_pos:, :]  # [n_kv_heads, num_new, head_dim]
        v_new = v_all[0, :, start_pos:, :]
        _scatter_seq_kv(paged_kv, layer_idx, seq, start_pos, k_new, v_new)


# ── 内部辅助 ──────────────────────────────────────────────────────────────────

def _cached_kv_len(seq: Sequence) -> int:
    """seq 进入本步前已有的 KV token 数。"""
    if seq.status == SeqStatus.RUNNING and seq.output_ids:
        return seq.num_tokens() - 1
    # A block-granular hit can cover an entire aligned prompt, but a CausalLM
    # still needs one input token to produce next-token logits.
    return min(seq.num_cached_tokens, max(seq.num_tokens() - 1, 0))


def _gather_seq_kv(
    paged_kv: torch.Tensor,
    layer_idx: int,
    seq: Sequence,
    clen: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """gather seq 的前 clen token 的 KV。返回 [n_kv_heads, clen, head_dim] x2。"""
    block_size = paged_kv.shape[3]
    k_parts: list[torch.Tensor] = []
    v_parts: list[torch.Tensor] = []
    remaining = clen

    for bid in seq.block_table:
        if remaining <= 0:
            break
        take = min(block_size, remaining)
        k_blk = paged_kv[layer_idx, 0, bid, :take]  # [take, n_kv_heads, head_dim]
        v_blk = paged_kv[layer_idx, 1, bid, :take]
        k_parts.append(k_blk.permute(1, 0, 2))      # [n_kv_heads, take, head_dim]
        v_parts.append(v_blk.permute(1, 0, 2))
        remaining -= take

    return torch.cat(k_parts, dim=1), torch.cat(v_parts, dim=1)


def _scatter_seq_kv(
    paged_kv: torch.Tensor,
    layer_idx: int,
    seq: Sequence,
    start_pos: int,
    k_new: torch.Tensor,  # [n_kv_heads, num_new, head_dim]
    v_new: torch.Tensor,
) -> None:
    """逐 token scatter 写回 paged_kv slots。"""
    block_size = paged_kv.shape[3]
    num_new = k_new.shape[1]

    for t in range(num_new):
        token_pos = start_pos + t
        blk_idx = token_pos // block_size
        slot = token_pos % block_size
        if blk_idx >= len(seq.block_table):
            break
        bid = seq.block_table[blk_idx]
        paged_kv[layer_idx, 0, bid, slot] = k_new[:, t, :]  # [n_kv_heads, head_dim]
        paged_kv[layer_idx, 1, bid, slot] = v_new[:, t, :]
