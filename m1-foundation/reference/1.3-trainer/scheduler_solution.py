"""
reference/1.3-trainer/scheduler_solution.py
Reference answer for minilm/training/scheduler.py (Phase 1.3, Unit U3.2)

DO NOT read before spending 30+ minutes on your own.
"""

from __future__ import annotations

import math


def lr_at(
    step: int,
    *,
    max_lr: float,
    min_lr: float,
    warmup_steps: int,
    total_steps: int,
) -> float:
    """Cosine learning-rate schedule with linear warmup.

    Phases:
        [0, warmup_steps)   : linear ramp from 0 → max_lr
        [warmup_steps, total_steps): cosine decay from max_lr → min_lr
        [total_steps, ∞)    : constant min_lr
    """
    if step < warmup_steps:
        # Linear warmup: fraction of the way through warmup
        return max_lr * step / warmup_steps
    elif step >= total_steps:
        return min_lr
    else:
        # Cosine decay
        # progress goes from 0.0 (at warmup_steps) to 1.0 (at total_steps)
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr + (max_lr - min_lr) * cosine_factor
