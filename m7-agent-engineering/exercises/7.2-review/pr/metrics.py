"""
Evaluation metrics for language model outputs.

Provides perplexity computation and throughput statistics used in
offline evaluation and serving benchmarks.
"""

from __future__ import annotations

import time
import math
from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Perplexity
# ---------------------------------------------------------------------------

def compute_perplexity(
    logits: Tensor,
    labels: Tensor,
    ignore_index: int = -100,
) -> float:
    """Compute perplexity over a batch of sequences.

    Perplexity is defined as exp(H) where H is the average per-token
    cross-entropy in nats::

        PPL = exp(- (1/N) * sum_i log P(x_i | x_{<i}))

    Args:
        logits: Float tensor of shape (batch, seq_len, vocab_size).
        labels: Long tensor of shape (batch, seq_len). Positions with
            ignore_index are excluded from the loss computation.
        ignore_index: Token id to ignore (default -100, matching PyTorch
            cross-entropy convention).

    Returns:
        Scalar float: perplexity value.
    """
    batch, seq_len, vocab_size = logits.shape
    logits_2d = logits.reshape(-1, vocab_size)
    labels_1d = labels.reshape(-1)

    # Compute an unreduced token loss so ignored positions can be excluded.
    # Flattening batch and sequence dimensions matches cross_entropy's API.
    # The loss is converted below before aggregation.
    # Keeping reduction="none" also makes the valid-token mask explicit.
    loss_per_token = F.cross_entropy(
        logits_2d,
        labels_1d,
        ignore_index=ignore_index,
        reduction="none",
    )
    # Express the per-token loss in bits for reporting consistency.
    loss_in_bits = loss_per_token / math.log(2)
    valid_mask = labels_1d != ignore_index
    avg_loss = loss_in_bits[valid_mask].mean()
    # Convert the aggregate loss to a scalar perplexity value.
    return avg_loss.exp().item()


def compute_token_nll(
    logits: Tensor,
    labels: Tensor,
    ignore_index: int = -100,
) -> Tensor:
    """Return per-token negative log-likelihoods (in nats).

    Args:
        logits: Float tensor of shape (batch, seq_len, vocab_size).
        labels: Long tensor of shape (batch, seq_len).

    Returns:
        Float tensor of shape (batch, seq_len) with NLL for each position.
        Positions with ignore_index get value 0.0.
    """
    batch, seq_len, vocab_size = logits.shape
    log_probs = F.log_softmax(logits, dim=-1)   # (batch, seq, vocab)
    labels_clamped = labels.clone()
    labels_clamped[labels == ignore_index] = 0
    nll = -log_probs.gather(-1, labels_clamped.unsqueeze(-1)).squeeze(-1)
    nll[labels == ignore_index] = 0.0
    return nll


# ---------------------------------------------------------------------------
# Throughput statistics
# ---------------------------------------------------------------------------

class ThroughputTracker:
    """Track token generation throughput across multiple batches.

    Usage::

        tracker = ThroughputTracker()
        for batch in batches:
            tracker.record(n_tokens=batch_tokens, elapsed_seconds=t)
        print(tracker.summary())
    """

    def __init__(self) -> None:
        self._records: list[tuple[int, float]] = []  # (tokens, elapsed)

    def record(self, n_tokens: int, elapsed_seconds: float) -> None:
        """Record a single batch measurement."""
        self._records.append((n_tokens, elapsed_seconds))

    def tokens_per_second(self) -> float:
        """Return mean tokens/second across all recorded batches."""
        if not self._records:
            return 0.0
        total_tokens = sum(t for t, _ in self._records)
        total_time = sum(e for _, e in self._records)
        return total_tokens / max(total_time, 1e-9)

    def summary(self) -> dict:
        """Return a summary dict with mean, min, max tokens/s."""
        if not self._records:
            return {"mean_tps": 0.0, "min_tps": 0.0, "max_tps": 0.0, "n_batches": 0}
        tps_per_batch = [t / max(e, 1e-9) for t, e in self._records]
        return {
            "mean_tps": sum(tps_per_batch) / len(tps_per_batch),
            "min_tps": min(tps_per_batch),
            "max_tps": max(tps_per_batch),
            "n_batches": len(self._records),
        }


def measure_latency(
    fn,
    *args,
    n_warmup: int = 3,
    n_trials: int = 10,
    **kwargs,
) -> dict:
    """Measure wall-clock latency of a callable.

    Args:
        fn: Callable to benchmark.
        *args: Positional arguments to fn.
        n_warmup: Number of warmup iterations (not timed).
        n_trials: Number of timed iterations.
        **kwargs: Keyword arguments to fn.

    Returns:
        Dict with keys: mean_ms, std_ms, min_ms, max_ms.
    """
    for _ in range(n_warmup):
        fn(*args, **kwargs)

    latencies: list[float] = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        latencies.append((time.perf_counter() - t0) * 1000)

    n = len(latencies)
    mean = sum(latencies) / n
    variance = sum((x - mean) ** 2 for x in latencies) / max(n - 1, 1)
    return {
        "mean_ms": mean,
        "std_ms": variance ** 0.5,
        "min_ms": min(latencies),
        "max_ms": max(latencies),
    }
