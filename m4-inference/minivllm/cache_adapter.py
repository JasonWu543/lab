"""cache_adapter.py — 分页 KV 与 transformers DynamicCache 的适配层
（Phase 4.0，Agent 已给出完整提示，属于脚手架代码）

本文件是唯一和模型输入输出直接交互的地方。
物理 KV 存储 shape：
  paged_kv: [num_layers, 2, num_blocks, block_size, n_kv_heads, head_dim]
  paged_kv[:, 0]  →  key 存储
  paged_kv[:, 1]  →  value 存储

gather_past_kv：
  从分页存储按 seq.block_table 拼出连续 KV → 组装 DynamicCache。
  每次物理拷贝一次（真 vLLM 靠 paged attention kernel 原地读，差距写 POSTMORTEM）。

scatter_new_kv：
  把模型前向后 past_key_values 中本步新增的 KV token 写回 paged_kv 对应 slots。

DynamicCache 在 transformers==4.52.4 的用法（已验证）：
  cache = DynamicCache()
  cache.update(key_states, value_states, layer_idx)
  # key_states shape: [batch, n_kv_heads, seq_len, head_dim]
  # 访问：cache.key_cache[layer_idx]  # [batch, n_kv_heads, total_len, head_dim]

────────────────────────────────────────────────────────────────────────────────
本文件的核心逻辑 Agent 已给出（gather/scatter 索引代码），
属于"脚手架"，不在泄题红线内。
你可以直接使用，也可以自己重写——关键是 gather 必须产出正确的 past_key_values
让 model(input_ids, past_key_values=...) 前向数值正确。

注意：gather_past_kv 为简化实现对每个 seq 逐个前向，不做 batching padding。
────────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import torch
from transformers import DynamicCache

from .request import Sequence, SeqStatus


def gather_past_kv(
    paged_kv: torch.Tensor,   # [num_layers, 2, num_blocks, block_size, n_kv_heads, head_dim]
    seqs: list[Sequence],
) -> DynamicCache:
    """
    按 block_table 从分页存储 gather 出各 seq 的历史 KV，
    组装为 DynamicCache（batch 维度 = len(seqs) = 1，本实现逐 seq 调用）。

    返回的 cache 可直接传入 model(input_ids, past_key_values=cache, use_cache=True)。
    """
    num_layers, _, num_blocks, block_size, n_kv_heads, head_dim = paged_kv.shape
    assert len(seqs) == 1, "本实现每次只处理单条 seq（Engine 逐条前向）"
    seq = seqs[0]
    clen = _cached_kv_len(seq)

    cache = DynamicCache()
    if clen == 0:
        return cache  # 空 cache，模型从头计算

    for layer_idx in range(num_layers):
        k_seq, v_seq = _gather_seq_kv(paged_kv, layer_idx, seq, clen)
        # k_seq: [n_kv_heads, clen, head_dim] → unsqueeze batch dim → [1, n_kv_heads, clen, head_dim]
        cache.update(k_seq.unsqueeze(0), v_seq.unsqueeze(0), layer_idx)

    return cache


def scatter_new_kv(
    paged_kv: torch.Tensor,
    new_cache: DynamicCache,
    seqs: list[Sequence],
    new_token_positions: list[int],
) -> None:
    """
    把 new_cache 中本步新产生的 KV token 写回 paged_kv。
    new_token_positions[i] = seqs[i] 本步前已有的 KV token 数（即 gather 时的 clen）。
    """
    num_layers = paged_kv.shape[0]
    assert len(seqs) == 1

    seq = seqs[0]
    start_pos = new_token_positions[0]

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
    return seq.num_cached_tokens


def _gather_seq_kv(
    paged_kv: torch.Tensor,
    layer_idx: int,
    seq: Sequence,
    clen: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """从 paged_kv gather seq 的前 clen 个 token 的 KV。返回 ([n_kv_heads, clen, head_dim],...)。"""
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

    k = torch.cat(k_parts, dim=1)  # [n_kv_heads, clen, head_dim]
    v = torch.cat(v_parts, dim=1)
    return k, v


def _scatter_seq_kv(
    paged_kv: torch.Tensor,
    layer_idx: int,
    seq: Sequence,
    start_pos: int,
    k_new: torch.Tensor,  # [n_kv_heads, num_new, head_dim]
    v_new: torch.Tensor,
) -> None:
    """把新 KV token 写回 paged_kv 的对应 slots（逐 token scatter）。"""
    block_size = paged_kv.shape[3]
    num_new = k_new.shape[1]

    for t in range(num_new):
        token_pos = start_pos + t
        blk_idx = token_pos // block_size
        slot = token_pos % block_size
        if blk_idx >= len(seq.block_table):
            break
        bid = seq.block_table[blk_idx]
        # k_new[:, t, :]: [n_kv_heads, head_dim]
        paged_kv[layer_idx, 0, bid, slot] = k_new[:, t, :]
        paged_kv[layer_idx, 1, bid, slot] = v_new[:, t, :]
