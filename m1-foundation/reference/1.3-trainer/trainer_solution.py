"""
reference/1.3-trainer/trainer_solution.py
Reference answer for minilm/training/trainer.py (Phase 1.3, Unit U3.2–U3.4)

DO NOT read before spending 30+ minutes on your own.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .checkpoint_solution import CheckpointCorruptError, load_checkpoint, save_checkpoint
from .scheduler_solution import lr_at


class NonFiniteLossError(RuntimeError):
    """Raised when a training step produces a non-finite loss (NaN or inf)."""


@dataclass
class TrainConfig:
    max_steps: int
    micro_batch_size: int
    grad_accum_steps: int = 1
    max_lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 10
    grad_clip: float = 1.0
    seed: int = 42
    ckpt_every: int = 0          # 0 = no auto-save
    ckpt_path: str | None = None
    log_path: str | None = None  # jsonl, one line per step


class Trainer:
    """Single-device trainer with grad accumulation, cosine LR, and checkpoint/resume.

    Ordering inside train_step:
        1. Loop grad_accum_steps micro batches:
               loss = cross_entropy(logits, targets) / grad_accum_steps
               loss.backward()
        2. Clip global gradient norm
        3. optimizer.step()
        4. Update LR via scheduler
        5. optimizer.zero_grad()
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: TrainConfig,
        dataloader: DataLoader,
    ) -> None:
        self.model = model
        self.cfg = cfg
        self.dataloader = dataloader

        # Reproducible optimizer init
        torch.manual_seed(cfg.seed)

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.max_lr,
            betas=(0.9, 0.95),
            weight_decay=0.1,
        )
        self.step = 0
        self._data_iter: Iterator | None = None
        self._log_file = None
        if cfg.log_path:
            self._log_file = open(cfg.log_path, "a")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Pull next batch from the infinite dataloader iterator."""
        if self._data_iter is None:
            self._data_iter = iter(self.dataloader)
        try:
            return next(self._data_iter)
        except StopIteration:
            self._data_iter = iter(self.dataloader)
            return next(self._data_iter)

    def _current_lr(self) -> float:
        cfg = self.cfg
        return lr_at(
            self.step,
            max_lr=cfg.max_lr,
            min_lr=cfg.min_lr,
            warmup_steps=cfg.warmup_steps,
            total_steps=cfg.max_steps,
        )

    def _apply_lr(self) -> None:
        lr = self._current_lr()
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train_step(self) -> dict:
        """One optimizer step (internally loops grad_accum_steps micro batches).

        Returns {"step", "loss", "lr", "grad_norm"}.
        Raises NonFiniteLossError on non-finite loss without advancing step.
        """
        cfg = self.cfg
        self.model.train()

        # Set LR before we do anything (step scheduler before optimizer.step
        # so that the LR logged corresponds to the step we're about to execute)
        lr = self._current_lr()
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

        self.optimizer.zero_grad()

        accumulated_loss = 0.0

        for _ in range(cfg.grad_accum_steps):
            x, y = self._get_batch()
            logits = self.model(x)  # (B, T, V)
            V = logits.size(-1)
            loss = F.cross_entropy(logits.view(-1, V), y.view(-1))

            if not math.isfinite(loss.item()):
                raise NonFiniteLossError(
                    f"Non-finite loss={loss.item()} at step {self.step}"
                )

            # Divide BEFORE backward so accumulated gradients are already normalized
            (loss / cfg.grad_accum_steps).backward()
            accumulated_loss += loss.item() / cfg.grad_accum_steps

        # Clip global gradient norm
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), cfg.grad_clip
        ).item()

        self.optimizer.step()
        self.step += 1

        log_entry = {
            "step": self.step,
            "loss": accumulated_loss,
            "lr": lr,
            "grad_norm": grad_norm,
        }

        if self._log_file is not None:
            self._log_file.write(json.dumps(log_entry) + "\n")
            self._log_file.flush()

        return log_entry

    def train(self) -> list[dict]:
        """Run training until cfg.max_steps, returning all log entries."""
        logs = []
        while self.step < self.cfg.max_steps:
            entry = self.train_step()
            logs.append(entry)

            if (
                self.cfg.ckpt_every > 0
                and self.cfg.ckpt_path is not None
                and self.step % self.cfg.ckpt_every == 0
            ):
                save_checkpoint(
                    self.cfg.ckpt_path,
                    model=self.model,
                    optimizer=self.optimizer,
                    step=self.step,
                )
        return logs

    def save(self, path: str | Path) -> None:
        """Save checkpoint at current step."""
        save_checkpoint(
            path,
            model=self.model,
            optimizer=self.optimizer,
            step=self.step,
        )

    def resume(self, path: str | Path) -> None:
        """Restore from checkpoint and fast-forward dataloader to saved step.

        Fast-forward strategy: re-create the iterator and skip the batches
        already consumed. NOTE: one optimizer step consumes grad_accum_steps
        micro batches — skipping only `step` batches diverges when accum > 1.
        This is O(step) but simple; see POSTMORTEM for large-dataset alternatives.
        """
        info = load_checkpoint(path, model=self.model, optimizer=self.optimizer)
        self.step = info["step"]

        # Fast-forward the dataloader by consuming step * grad_accum_steps batches
        self._data_iter = iter(self.dataloader)
        for _ in range(self.step * self.cfg.grad_accum_steps):
            try:
                next(self._data_iter)
            except StopIteration:
                self._data_iter = iter(self.dataloader)
                next(self._data_iter)
