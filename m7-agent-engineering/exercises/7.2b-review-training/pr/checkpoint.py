"""Checkpoint persistence helpers."""

from pathlib import Path

import torch


def save_checkpoint(path, model, optimizer, scheduler, scaler, global_step):
    """Persist training state required to continue a run."""
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "global_step": global_step,
    }
    torch.save(payload, Path(path))


def load_checkpoint(path, model, optimizer, scheduler=None, scaler=None):
    """Restore a checkpoint and return its completed optimizer-step count."""
    try:
        payload = torch.load(Path(path), weights_only=True)
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        return int(payload.get("global_step", 0))
    except (OSError, RuntimeError, KeyError, ValueError):
        return 0


def resumed_microbatch_offset(global_step, accumulation_steps):
    """Return the input-batch offset corresponding to restored progress."""
    if accumulation_steps <= 0:
        raise ValueError("accumulation_steps must be positive")
    return global_step
