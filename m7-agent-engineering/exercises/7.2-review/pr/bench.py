"""
Benchmark: eager top-p sampling vs. compiled (torch.compile) top-p sampling.

Compares throughput of the nucleus sampling implementation under two
execution modes and prints a side-by-side throughput table.

Run with::

    python bench.py
"""

from __future__ import annotations

import time
import torch
from sampling import top_p_filter, temperature_sample


# ---------------------------------------------------------------------------
# Two "implementations" to compare
# ---------------------------------------------------------------------------

def _eager_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Eager (no compilation) top-p filter + multinomial sample."""
    filtered = top_p_filter(logits, p)
    return temperature_sample(filtered, temperature=1.0)


# Compile once at module level for fair reuse in the timed loop
_compiled_top_p_filter = torch.compile(top_p_filter, fullgraph=False)


def _compiled_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    """torch.compile top-p filter + multinomial sample."""
    filtered = _compiled_top_p_filter(logits, p)
    return temperature_sample(filtered, temperature=1.0)


# ---------------------------------------------------------------------------
# Benchmark harness
# ---------------------------------------------------------------------------

def _time_fn(fn, *args, n_trials: int = 50, **kwargs) -> float:
    """Return mean wall-clock time in milliseconds over n_trials calls."""
    elapsed = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        elapsed.append((time.perf_counter() - t0) * 1e3)
    return sum(elapsed) / len(elapsed)


def run_benchmark(
    vocab_size: int = 32_000,
    p: float = 0.9,
    n_trials: int = 50,
) -> None:
    """Run the eager vs. compiled benchmark and print results."""

    print(f"Benchmark: top-p sampling  (vocab={vocab_size}, p={p})")
    print("-" * 60)

    # Run a short warmup before collecting timing samples.
    # This also ensures compilation is complete before the timed region.
    # A small warmup batch keeps startup time manageable.
    # Timed inputs are constructed separately below.

    # Warmup
    warmup_logits = torch.randn(1, vocab_size)
    for _ in range(5):
        _compiled_top_p(warmup_logits, p)

    # Choose representative serving batch sizes for each execution mode.
    # Throughput is normalized to tokens per second below.
    # Latency and batch size are printed alongside the normalized result.
    eager_batch  = 64
    compiled_batch = 1

    eager_logits    = torch.randn(eager_batch, vocab_size)
    compiled_logits = torch.randn(compiled_batch, vocab_size)

    eager_ms    = _time_fn(_eager_top_p,    eager_logits,    p, n_trials=n_trials)
    compiled_ms = _time_fn(_compiled_top_p, compiled_logits, p, n_trials=n_trials)

    eager_tps    = eager_batch    / (eager_ms    / 1e3)
    compiled_tps = compiled_batch / (compiled_ms / 1e3)

    print(f"  {'Method':<18} {'Batch':>6} {'Latency (ms)':>14} {'Tokens/s':>12}")
    print(f"  {'-'*18} {'-'*6} {'-'*14} {'-'*12}")
    print(f"  {'eager':<18} {eager_batch:>6} {eager_ms:>14.3f} {eager_tps:>12,.0f}")
    print(f"  {'compiled':<18} {compiled_batch:>6} {compiled_ms:>14.3f} {compiled_tps:>12,.0f}")
    print()

    speedup = eager_tps / max(compiled_tps, 1e-9)
    print(f"  Reported speedup (compiled/eager): {1/speedup:.2f}x")


if __name__ == "__main__":
    run_benchmark()
