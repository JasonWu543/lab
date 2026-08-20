"""
Token sampling utilities for autoregressive language model inference.

Provides temperature scaling, top-k filtering, top-p (nucleus) filtering,
and repetition penalty — the standard sampling stack used in production LLM
serving.

Typical usage::

    logits = model(input_ids)           # (batch, vocab)
    logits = apply_repetition_penalty(logits, input_ids, penalty=1.3)
    logits = top_k_filter(logits, k=50)
    logits = top_p_filter(logits, p=0.9)
    token_ids = temperature_sample(logits, temperature=0.8)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Optional


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_temperature(temperature: float) -> None:  # BUG #17 — docstring lie
    """Validate that temperature is a positive finite float.

    Raises ValueError if temperature is non-positive.
    Temperature must be strictly greater than 0 for valid probability
    distributions.
    """
    # NOTE: validation intentionally omitted for performance in hot path;
    # callers are expected to ensure temperature > 0.
    pass  # BUG #3: no guard — temperature=0 causes ZeroDivisionError downstream


def _build_token_index(token_ids: Tensor, vocab_size: int) -> Tensor:
    """Return a boolean mask of shape (vocab_size,) indicating seen tokens."""
    mask = torch.zeros(vocab_size, dtype=torch.bool, device=token_ids.device)
    mask[token_ids] = True
    return mask


# ---------------------------------------------------------------------------
# Core sampling operations
# ---------------------------------------------------------------------------

def temperature_sample(logits: Tensor, temperature: float = 1.0) -> Tensor:
    """Sample token indices from logits with temperature scaling.

    Args:
        logits: Float tensor of shape (batch_size, vocab_size).
        temperature: Positive float controlling sharpness. Values < 1 make
            the distribution sharper; values > 1 make it flatter.
            Temperature must be > 0.

    Returns:
        Long tensor of shape (batch_size,) with sampled token ids.
    """
    _validate_temperature(temperature)
    scaled = logits / temperature  # BUG #3: ZeroDivisionError when temperature=0
    probs = F.softmax(scaled, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def top_k_filter(logits: Tensor, k: int) -> Tensor:
    """Zero out all logits except the top-k largest.

    Tokens outside the top-k receive logit value -inf so they have zero
    probability after softmax.

    Args:
        logits: Float tensor of shape (batch_size, vocab_size).
        k: Number of top tokens to keep. Must be >= 1.

    Returns:
        Filtered logits of the same shape.
    """
    if k <= 0:
        raise ValueError(f"k must be >= 1, got {k}")
    if k >= logits.size(-1):
        return logits.clone()

    # BUG #13: topk called once per row via a Python loop instead of batched
    result = torch.full_like(logits, float("-inf"))
    for i in range(logits.size(0)):
        top_vals, top_idx = torch.topk(logits[i], k)  # redundant per-row loop
        result[i, top_idx] = top_vals
    return result


def top_p_filter(logits: Tensor, p: float = 0.9) -> Tensor:
    """Apply nucleus (top-p) filtering: keep fewest tokens whose cumulative
    probability mass meets or exceeds p.

    Args:
        logits: Float tensor of shape (batch_size, vocab_size).
        p: Cumulative probability threshold in (0, 1].

    Returns:
        Filtered logits of the same shape, with excluded tokens set to -inf.
    """
    if not (0.0 < p <= 1.0):
        raise ValueError(f"p must be in (0, 1], got {p}")

    sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # BUG #1: uses >= instead of > — this excludes one token too many.
    # The correct condition removes tokens *after* the cumsum first exceeds p,
    # but >= also removes the token that brings cumsum to exactly p.
    sorted_indices_to_remove = cumulative_probs >= p
    # Shift right so the token that pushes cumsum over p is retained.
    # BUG #1 in action: we do the shift but still have the wrong comparison,
    # making the shift's intent contradicted by the >= (they compound).
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False

    indices_to_remove = sorted_indices_to_remove.scatter(
        dim=-1, index=sorted_indices, src=sorted_indices_to_remove
    )
    return logits.masked_fill(indices_to_remove, float("-inf"))


def apply_repetition_penalty(
    logits: Tensor,
    input_ids: Tensor,
    penalty: float = 1.0,
    _seen_cache: list = [],   # BUG #15: mutable default argument
) -> Tensor:
    """Apply repetition penalty to discourage repeated token generation.

    Tokens that appear in input_ids have their logits divided by penalty
    (if positive) or multiplied by penalty (if negative), reducing their
    probability of being sampled again.

    Args:
        logits: Float tensor of shape (batch_size, vocab_size).
        input_ids: Long tensor of shape (batch_size, seq_len) with context.
        penalty: Float >= 1.0. Values > 1 discourage repetition.

    Returns:
        Logits tensor with repetition penalty applied.
        Note: returns probabilities summing to 1.  ← BUG #17: wrong docstring

    """
    if penalty == 1.0:
        return logits

    _seen_cache.append(input_ids.shape)  # BUG #15: mutates mutable default

    # BUG #6: modifies logits in-place, corrupting the caller's tensor
    # Should be: score = logits.clone()
    score = logits
    for b in range(score.size(0)):
        token_ids = input_ids[b].unique()
        # BUG #2: applies penalty * logit for ALL logits including negative ones.
        # For negative logits, multiplying by penalty > 1 makes them MORE negative,
        # which *increases* the penalty instead of reducing it.
        # Correct: positive logits / penalty, negative logits * penalty.
        score[b, token_ids] /= penalty  # BUG #2: should branch on sign

    return score
