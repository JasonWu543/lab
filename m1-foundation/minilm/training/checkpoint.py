"""
minilm/training/checkpoint.py — Phase 1.3, Unit U3.3: Checkpoint & Resume

Unlocking order:  U3.1 data  →  U3.2 loop  →  U3.3 checkpoint  →  U3.4 failure
Test command:     cd m1-foundation && python3 -m pytest tests/test_trainer.py -x -q
Reference path:   reference/1.3-trainer/checkpoint_solution.py   (look only after 30+ min stuck)

YOUR TASK: implement save_checkpoint, load_checkpoint, and CheckpointCorruptError.

Key design question: "What, besides model and optimizer, generates randomness during training?
Missing even one of them will cause the resumed trajectory to diverge."

Atomic write pattern (required):
    1. Write payload to <path>.tmp with torch.save()
    2. f.flush() + os.fsync(f.fileno())  — ensure bytes reach disk
    3. os.replace(tmp_path, path)        — atomic rename (POSIX guarantee)
    If step 3 never runs (crash/exception), the OLD checkpoint at <path> is unaffected.

Corruption detection:
    Wrap torch.load() in a try/except and raise CheckpointCorruptError on any failure.
    Also validate that all required keys are present before calling load_state_dict().

Forbidden: silent partial loads. If "optimizer_state" is missing, raise — don't skip.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch


class CheckpointCorruptError(RuntimeError):
    """Raised when a checkpoint file is truncated or otherwise unreadable."""


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    extra: dict | None = None,
) -> None:
    """Atomically write a checkpoint.

    Writes to <path>.tmp first, fsyncs, then os.replace to <path>.
    Must save: model state_dict, optimizer state_dict, step, extra dict,
    and ALL RNG states needed to reproduce exact training trajectories.
    """
    raise NotImplementedError("U3.3: implement save_checkpoint")


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> dict:
    """Load a checkpoint and restore all state (including RNG).

    Returns {"step": int, "extra": dict}.
    Raises CheckpointCorruptError if the file is truncated, corrupt,
    or missing required keys (including "optimizer_state").
    """
    raise NotImplementedError("U3.3: implement load_checkpoint")
