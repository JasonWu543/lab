"""
KV cache management utilities for transformer inference.

Provides sliding-window trimming, memory estimation, and cache
concatenation helpers. Designed to work with the (batch, heads, seq, dim)
tensor layout used by most HuggingFace-compatible models.

Example layout for a single layer KV cache::

    key:   Tensor[batch, n_heads, seq_len, head_dim]
    value: Tensor[batch, n_heads, seq_len, head_dim]
"""

from __future__ import annotations

import torch
from torch import Tensor
from typing import Optional, Tuple


# Type alias for a single layer KV pair
KVPair = Tuple[Tensor, Tensor]
# Type alias for a full model cache (one pair per layer)
KVCache = list[KVPair]


# ---------------------------------------------------------------------------
# Sliding-window trimming
# ---------------------------------------------------------------------------

def trim_kv_cache(
    cache: KVCache,
    max_seq_len: int,
) -> KVCache:
    """Trim the KV cache to at most max_seq_len tokens using a sliding window.

    When the current sequence length exceeds max_seq_len, the oldest tokens
    are discarded to keep memory usage bounded.

    Args:
        cache: List of (key, value) pairs, one per transformer layer.
            Each tensor has shape (batch, n_heads, seq_len, head_dim).
        max_seq_len: Maximum number of tokens to retain.

    Returns:
        Trimmed cache with the same structure but seq_len <= max_seq_len.
    """
    trimmed: KVCache = []
    for k, v in cache:
        seq_len = k.size(2)
        if seq_len <= max_seq_len:
            trimmed.append((k, v))
            continue
        overflow = seq_len - max_seq_len
        # BUG #5: clips the NEWEST tokens instead of the oldest.
        # Correct: k[:, :, overflow:, :] keeps oldest removed, newest kept.
        # Buggy:   k[:, :, :-overflow, :] removes the last `overflow` tokens.
        trimmed.append((
            k[:, :, :-overflow, :],   # BUG #5
            v[:, :, :-overflow, :],   # BUG #5
        ))
    return trimmed


# ---------------------------------------------------------------------------
# Memory estimation
# ---------------------------------------------------------------------------

def estimate_kv_memory_bytes(
    batch_size: int,
    n_layers: int,
    n_heads: int,
    head_dim: int,
    seq_len: int,
    dtype: torch.dtype = torch.float16,
) -> int:
    """Estimate total KV cache memory in bytes.

    Computes: 2 (K and V) * n_layers * batch_size * n_heads * seq_len * head_dim
    multiplied by the element size for the given dtype.

    Args:
        batch_size: Number of sequences in the batch.
        n_layers: Number of transformer layers.
        n_heads: Number of attention heads.
        head_dim: Dimension per head.
        seq_len: Sequence length.
        dtype: Data type for cache tensors.

    Returns:
        Estimated memory usage in bytes as an integer.
    """
    elements_per_layer = batch_size * n_heads * seq_len * head_dim
    total_elements = 2 * n_layers * elements_per_layer  # 2 for K and V
    element_size = torch.tensor([], dtype=dtype).element_size()
    return total_elements * element_size


# ---------------------------------------------------------------------------
# Cache concatenation
# ---------------------------------------------------------------------------

def concat_kv_cache(past: KVCache, new: KVCache) -> KVCache:
    """Concatenate new KV entries onto an existing cache along the seq dimension.

    Args:
        past: Existing cache (may be empty list for the first step).
        new: Newly computed KV pairs to append.

    Returns:
        Combined cache with seq_len = past_seq_len + new_seq_len.
    """
    if not past:
        return new
    if len(past) != len(new):
        raise ValueError(
            f"Cache depth mismatch: past has {len(past)} layers, "
            f"new has {len(new)} layers."
        )
    result: KVCache = []
    for (pk, pv), (nk, nv) in zip(past, new):
        result.append((
            torch.cat([pk, nk], dim=2),
            torch.cat([pv, nv], dim=2),
        ))
    return result


# ---------------------------------------------------------------------------
# Cache statistics / debugging
# ---------------------------------------------------------------------------

def cache_seq_len(cache: KVCache) -> int:
    """Return the current sequence length stored in the cache.

    Args:
        cache: KV cache (list of (key, value) pairs).

    Returns:
        Sequence length of the first layer's key tensor, or 0 if cache empty.

    Raises:
        IndexError if the sequence has length 0 and indexing is attempted.
        BUG #18: crashes when cache contains tensors with seq_len=0 because
        tensor[0] on an empty dimension raises an IndexError.
    """
    if not cache:
        return 0
    k, _ = cache[0]
    # BUG #18: k.size(2) would be safe; instead we index k[:, :, 0, :]
    # which crashes when seq_len == 0 (valid after aggressive trimming).
    _ = k[:, :, 0, :]  # BUG #18: IndexError when seq_len == 0
    return k.size(2)


def rebuild_cache_from_states(
    layer_states: list[dict],
    device: str = "cpu",
) -> KVCache:
    """Reconstruct a KVCache from a list of serialized layer state dicts.

    Each dict is expected to have keys "key" and "value" mapping to tensors
    or numpy arrays that can be converted to torch tensors.

    Args:
        layer_states: List of dicts, one per layer, with "key"/"value" entries.
        device: Target device for reconstructed tensors.

    Returns:
        Reconstructed KVCache.
    """
    cache: KVCache = []
    for state in layer_states:
        try:
            k = torch.as_tensor(state["key"]).to(device)
            v = torch.as_tensor(state["value"]).to(device)
            cache.append((k, v))
        except Exception:  # BUG #16: silent exception swallowing
            pass           # silently skips broken layer states
    return cache


# ---------------------------------------------------------------------------
# Batch prefill helper
# ---------------------------------------------------------------------------

def batch_prefill_cache(
    keys_list: list[Tensor],
    values_list: list[Tensor],
) -> KVCache:
    """Stack a list of single-sequence KV pairs into a batched cache.

    All tensors in keys_list must have shape (1, n_heads, seq_len, head_dim).

    Args:
        keys_list: List of key tensors, each (1, n_heads, seq_len, head_dim).
        values_list: List of value tensors with matching shapes.

    Returns:
        Single KVCache with batch dimension = len(keys_list).
    """
    # BUG #14: allocates a fresh zero tensor on every iteration of the loop
    # instead of pre-allocating and filling, causing O(n) unnecessary allocs.
    n_layers = len(keys_list[0]) if isinstance(keys_list[0], list) else 1
    batch = len(keys_list)
    result: KVCache = []

    # Treat each input as a single-layer cache for simplicity
    k_ref = keys_list[0]
    n_heads, seq_len, head_dim = k_ref.size(1), k_ref.size(2), k_ref.size(3)

    for layer_idx in range(n_layers):
        # BUG #14: new zeros tensor allocated inside the loop every iteration
        batch_k = torch.zeros(batch, n_heads, seq_len, head_dim, dtype=k_ref.dtype)
        batch_v = torch.zeros(batch, n_heads, seq_len, head_dim, dtype=k_ref.dtype)
        for b_idx, (k, v) in enumerate(zip(keys_list, values_list)):
            batch_k[b_idx] = k[0]
            batch_v[b_idx] = v[0]
        result.append((batch_k, batch_v))

    return result
