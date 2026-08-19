"""
minilm/training/trainer.py — Phase 1.3, Units U3.2–U3.4: Trainer

Unlocking order:  U3.1 data  →  U3.2 loop  →  U3.3 checkpoint  →  U3.4 failure
Test command:     cd m1-foundation && python3 -m pytest tests/test_trainer.py -x -q
Reference path:   reference/1.3-trainer/trainer_solution.py   (look only after 30+ min stuck)

YOUR TASK: implement train_step(), train(), save(), resume(), and the helpers.

Optimizer (frozen): torch.optim.AdamW(betas=(0.9, 0.95), weight_decay=0.1)

train_step() 要做的事（顺序自己排，这是 U3.2 的核心考点）：
    - 用 lr_at(self.step, ...) 设置 optimizer 各 param group 的 lr
    - 循环 grad_accum_steps 个 micro batch：_get_batch() → forward →
      F.cross_entropy → backward；上报的 loss 是各 micro batch 的平均
    - clip（torch.nn.utils.clip_grad_norm_）、optimizer.step、zero_grad、
      非有限 loss 检查（NonFiniteLossError，且 step 不推进、参数不被污染）
      ——这些各自放在循环内还是循环外、放在 backward 前还是后，自己推
    返回 {"step": self.step, "loss": ..., "lr": lr, "grad_norm": grad_norm}

先回答再动手（答案写进 POSTMORTEM）：
    - loss 除以 grad_accum_steps，应该在 backward 之前还是之后？为什么？
      （提示：梯度是对谁求的、accumulate 的语义是什么）
    - clip 应该对每个 micro batch 的梯度做，还是对累积后的整体梯度做？
      两种做法什么时候结果不同？测试里有一个用例专门抓这个。
    - NonFiniteLossError 抛早了或抛晚了，参数各会发生什么？

jsonl log format (one JSON object per line):
    {"step": 1, "loss": 4.23, "lr": 3e-4, "grad_norm": 0.85}

resume() fast-forward strategy:
    Re-create _data_iter = iter(dataloader), then skip the batches that were
    already consumed before the checkpoint. 想清楚：一个 optimizer step 消耗
    几个 batch？（测试里有 accum>1 的恢复用例专门抓这个 off-by-N）
    This is O(step) — acceptable for this lab. Write in POSTMORTEM why this is
    expensive at scale and name at least one alternative.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .checkpoint import CheckpointCorruptError, load_checkpoint, save_checkpoint
from .scheduler import lr_at


class NonFiniteLossError(RuntimeError):
    """Raised when a training step produces a non-finite loss (NaN or inf)."""


@dataclass
class TrainConfig:
    """Training hyperparameters and run settings.

    Fields:
        max_steps        : total optimizer steps to run
        micro_batch_size : batch size per micro-step (samples, not tokens)
        grad_accum_steps : number of micro-steps per optimizer step (default 1 = no accum)
        max_lr           : peak learning rate
        min_lr           : final (floor) learning rate
        warmup_steps     : steps for linear warmup phase
        grad_clip        : global gradient norm clip threshold
        seed             : master seed (model init, dataloader generator)
        ckpt_every       : save checkpoint every N steps (0 = disabled)
        ckpt_path        : path prefix for auto-saved checkpoints
        log_path         : path for jsonl training log (None = no log file)
    """
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
    """Single-device trainer with grad accumulation, cosine LR, and checkpoint/resume."""

    def __init__(
        self,
        model: nn.Module,
        cfg: TrainConfig,
        dataloader: DataLoader,
    ) -> None:
        self.model = model
        self.cfg = cfg
        self.dataloader = dataloader
        self.step = 0
        self._data_iter: Iterator | None = None
        self._log_file = None

        # TODO U3.2: seed global RNG with cfg.seed, then create the AdamW optimizer
        #   torch.manual_seed(cfg.seed)
        #   self.optimizer = torch.optim.AdamW(...)
        raise NotImplementedError("U3.2: implement Trainer.__init__")

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

    def train_step(self) -> dict:
        """One optimizer step (internally loops grad_accum_steps micro batches).

        Returns {"step", "loss", "lr", "grad_norm"}.
        Raises NonFiniteLossError on non-finite loss without advancing step.
        """
        # TODO U3.2: implement the training step following the docstring order above
        raise NotImplementedError("U3.2: implement Trainer.train_step")

    def train(self) -> list[dict]:
        """Run training until cfg.max_steps, returning all log entries."""
        # TODO U3.2: loop train_step() until self.step >= self.cfg.max_steps,
        # handle auto-checkpoint if ckpt_every > 0
        raise NotImplementedError("U3.2: implement Trainer.train")

    def save(self, path: str | Path) -> None:
        """Save a checkpoint at the current step."""
        # TODO U3.3: delegate to save_checkpoint
        raise NotImplementedError("U3.3: implement Trainer.save")

    def resume(self, path: str | Path) -> None:
        """Restore from checkpoint and fast-forward dataloader to saved step.

        Fast-forward: re-create iterator and skip `step` batches.
        """
        # TODO U3.3: delegate to load_checkpoint, update self.step,
        # then fast-forward self._data_iter past the already-consumed batches
        # (count them carefully; StopIteration means an epoch boundary —
        # handle it the same way _get_batch does)
        raise NotImplementedError("U3.3: implement Trainer.resume")
