"""
minilm/training/scheduler.py — Phase 1.3, Unit U3.2: LR Scheduler

Unlocking order:  U3.1 data  →  U3.2 loop  →  U3.3 checkpoint  →  U3.4 failure
Test command:     cd m1-foundation && python3 -m pytest tests/test_trainer.py -x -q
Reference path:   reference/1.3-trainer/scheduler_solution.py   (look only after 30+ min stuck)

YOUR TASK: implement lr_at.

The schedule has three phases:
  [0, warmup_steps)          linear ramp:  lr = max_lr * step / warmup_steps
  [warmup_steps, total_steps) cosine decay: progress = (step - warmup) / (total - warmup)
                                            cosine_factor = 0.5 * (1 + cos(π * progress))
                                            lr = min_lr + (max_lr - min_lr) * cosine_factor
  [total_steps, ∞)           constant:     lr = min_lr

Think about: why linear warmup? what goes wrong without it?
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
    """Return the learning rate for the given step.

    Phases:
        [0, warmup_steps)          : linear ramp from 0 → max_lr
        [warmup_steps, total_steps): cosine decay from max_lr → min_lr
        [total_steps, ∞)           : constant min_lr
    """
    raise NotImplementedError("U3.2: implement lr_at")
