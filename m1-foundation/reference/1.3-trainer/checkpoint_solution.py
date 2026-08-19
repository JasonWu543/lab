"""
reference/1.3-trainer/checkpoint_solution.py
Reference answer for minilm/training/checkpoint.py (Phase 1.3, Unit U3.3)

DO NOT read before spending 30+ minutes on your own.
"""

from __future__ import annotations

import io
import os
import random
from pathlib import Path

import numpy as np
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
    Saves:
        - model state_dict
        - optimizer state_dict
        - training step
        - extra metadata dict
        - full RNG states: torch, numpy, python random
    """
    path = Path(path)
    tmp_path = Path(str(path) + ".tmp")

    payload = {
        "step": step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "extra": extra or {},
        "rng": {
            "torch": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
    }

    # Write to temp file, fsync, then atomically replace
    with open(tmp_path, "wb") as f:
        torch.save(payload, f)
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp_path, path)


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> dict:
    """Load a checkpoint and restore all state (including RNG).

    Returns {"step": int, "extra": dict}.
    Raises CheckpointCorruptError if the file is truncated or corrupt.
    """
    path = Path(path)

    try:
        payload = torch.load(str(path), map_location="cpu", weights_only=False)
    except Exception as e:
        raise CheckpointCorruptError(
            f"Failed to load checkpoint from {path}: {e}"
        ) from e

    # Validate required keys
    required = {"step", "model_state", "optimizer_state", "rng"}
    missing = required - set(payload.keys())
    if missing:
        raise CheckpointCorruptError(
            f"Checkpoint missing keys: {missing}"
        )

    # Validate optimizer_state specifically (T10: missing optimizer key)
    if "optimizer_state" not in payload:
        raise CheckpointCorruptError("Checkpoint is missing 'optimizer_state'.")

    model.load_state_dict(payload["model_state"])
    optimizer.load_state_dict(payload["optimizer_state"])

    # Restore all RNG states
    rng = payload["rng"]
    torch.set_rng_state(rng["torch"])
    np.random.set_state(rng["numpy"])
    random.setstate(rng["python"])

    return {"step": payload["step"], "extra": payload.get("extra", {})}
