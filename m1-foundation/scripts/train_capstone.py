#!/usr/bin/env python3
"""
scripts/train_capstone.py — Phase 1.3, Unit U3.5: Capstone Training Script

Copilot scaffold: argparse, config loading, logging loop, and checkpoint
resume are FULLY implemented here. The only TODOs are:
  - Wire up MiniLM model from minilm.model  (Phase 1.2)
  - Wire up BPETokenizer from minilm.tokenizer  (Phase 1.0)

Usage:
    python3 scripts/train_capstone.py \\
        --data_path  data/tinystories/train.bin \\
        --output_dir runs/capstone_001 \\
        --max_steps  10000 \\
        --micro_batch_size 8 \\
        --grad_accum_steps 4 \\
        --max_lr    3e-4 \\
        --min_lr    3e-5 \\
        --warmup_steps 200 \\
        --grad_clip  1.0 \\
        --seed       42 \\
        --ckpt_every 500 \\
        [--resume runs/capstone_001/ckpt.pt]

Requirement before going to cloud (LAB_DESIGN §0.2):
    python3 -m pytest tests/test_trainer.py -q  → all green locally first
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Trainer imports (fully implemented by you in Phase 1.3)
# ---------------------------------------------------------------------------
from minilm.training.data import PackedDataset, make_dataloader
from minilm.training.trainer import TrainConfig, Trainer

# ---------------------------------------------------------------------------
# TODO 1: Wire up MiniLM model (Phase 1.2)
# ---------------------------------------------------------------------------
# Uncomment and adjust once Phase 1.2 is complete:
#
#   from minilm.model import MiniLM, MiniLMConfig
#
# Then in build_model() below, replace the placeholder with:
#   model_cfg = MiniLMConfig(
#       vocab_size=tokenizer.vocab_size,
#       n_layers=args.n_layers,
#       d_model=args.d_model,
#       n_heads=args.n_heads,
#       ...
#   )
#   model = MiniLM(model_cfg)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# TODO 2: Wire up BPETokenizer (Phase 1.0)
# ---------------------------------------------------------------------------
# Uncomment and adjust once Phase 1.0 BPETokenizer is serializable:
#
#   from minilm.tokenizer import BPETokenizer
#
# Then in main() below:
#   tokenizer = BPETokenizer.load(args.tokenizer_path)
#   vocab_size = tokenizer.vocab_size
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model placeholder (remove once Phase 1.2 is wired in)
# ---------------------------------------------------------------------------

import torch.nn as nn
import torch.nn.functional as F


class _PlaceholderModel(nn.Module):
    """Tiny stand-in model.

    DELETE this class and replace build_model() with MiniLM once Phase 1.2 is ready.
    """

    def __init__(self, vocab_size: int = 512, hidden: int = 64) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.fc = nn.Linear(hidden, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T) → (B, T, V)
        return self.fc(self.embed(x))


def build_model(args: argparse.Namespace, vocab_size: int) -> nn.Module:
    """Construct and return the language model.

    TODO: Replace _PlaceholderModel with MiniLM from Phase 1.2.
    """
    # TODO (Phase 1.2): replace with real MiniLM
    log.warning(
        "Using placeholder model — wire up MiniLM from minilm.model before cloud training."
    )
    return _PlaceholderModel(vocab_size=vocab_size, hidden=128)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MiniLM capstone training script (Phase 1.3 / U3.5)"
    )

    # Data
    p.add_argument("--data_path", required=True, type=Path,
                   help="Path to the .bin memmap token file (created by write_memmap)")
    p.add_argument("--seq_len", type=int, default=256,
                   help="Context length (tokens per sample)")
    p.add_argument("--tokenizer_path", type=Path, default=None,
                   help="Path to saved BPETokenizer (Phase 1.0); determines vocab_size")
    p.add_argument("--vocab_size", type=int, default=512,
                   help="Fallback vocab size if tokenizer_path is not provided")

    # Model (for Phase 1.2 MiniLM)
    p.add_argument("--n_layers", type=int, default=6, help="Transformer depth")
    p.add_argument("--d_model", type=int, default=384, help="Embedding dimension")
    p.add_argument("--n_heads", type=int, default=6, help="Attention heads")
    p.add_argument("--d_ff", type=int, default=1536, help="FFN hidden size")

    # Training
    p.add_argument("--max_steps", type=int, required=True)
    p.add_argument("--micro_batch_size", type=int, default=8)
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--max_lr", type=float, default=3e-4)
    p.add_argument("--min_lr", type=float, default=3e-5)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)

    # Checkpointing
    p.add_argument("--output_dir", type=Path, required=True,
                   help="Directory for checkpoints and logs")
    p.add_argument("--ckpt_every", type=int, default=500,
                   help="Save checkpoint every N optimizer steps (0 = disabled)")
    p.add_argument("--resume", type=Path, default=None,
                   help="Path to a .pt checkpoint to resume from")

    # Misc
    p.add_argument("--log_interval", type=int, default=10,
                   help="Print log line every N steps")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.output_dir / "ckpt.pt"
    log_path = args.output_dir / "train_log.jsonl"

    # Save config snapshot for reproducibility
    config_snap = vars(args).copy()
    config_snap = {k: str(v) if isinstance(v, Path) else v for k, v in config_snap.items()}
    (args.output_dir / "config.json").write_text(json.dumps(config_snap, indent=2))
    log.info("Config saved to %s/config.json", args.output_dir)

    # ---------------------------------------------------------------------------
    # TODO: Load tokenizer and determine vocab_size
    # ---------------------------------------------------------------------------
    # if args.tokenizer_path is not None:
    #     tokenizer = BPETokenizer.load(args.tokenizer_path)
    #     vocab_size = tokenizer.vocab_size
    #     log.info("Tokenizer loaded: vocab_size=%d", vocab_size)
    # else:
    vocab_size = args.vocab_size
    log.warning("No tokenizer_path — using vocab_size=%d", vocab_size)

    # ---------------------------------------------------------------------------
    # Dataset & dataloader
    # ---------------------------------------------------------------------------
    # Strip .bin suffix if user passes full path
    bin_prefix = args.data_path
    if str(bin_prefix).endswith(".bin"):
        bin_prefix = Path(str(bin_prefix)[:-4])

    dataset = PackedDataset(bin_prefix, seq_len=args.seq_len)
    log.info("Dataset: %d samples, seq_len=%d", len(dataset), args.seq_len)

    dataloader = make_dataloader(
        dataset,
        batch_size=args.micro_batch_size,
        shuffle=True,
        seed=args.seed,
    )

    # ---------------------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------------------
    torch.manual_seed(args.seed)
    model = build_model(args, vocab_size=vocab_size)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("Model params: %s M", f"{n_params / 1e6:.1f}")

    # ---------------------------------------------------------------------------
    # Trainer config
    # ---------------------------------------------------------------------------
    cfg = TrainConfig(
        max_steps=args.max_steps,
        micro_batch_size=args.micro_batch_size,
        grad_accum_steps=args.grad_accum_steps,
        max_lr=args.max_lr,
        min_lr=args.min_lr,
        warmup_steps=args.warmup_steps,
        grad_clip=args.grad_clip,
        seed=args.seed,
        ckpt_every=args.ckpt_every,
        ckpt_path=str(ckpt_path),
        log_path=str(log_path),
    )

    trainer = Trainer(model, cfg, dataloader)

    # ---------------------------------------------------------------------------
    # Resume if requested
    # ---------------------------------------------------------------------------
    if args.resume is not None:
        log.info("Resuming from checkpoint: %s", args.resume)
        trainer.resume(args.resume)
        log.info("Resumed at step %d / %d", trainer.step, args.max_steps)

    # ---------------------------------------------------------------------------
    # Training loop with timing and console logging
    # ---------------------------------------------------------------------------
    log.info(
        "Starting training — steps %d → %d  (effective_batch=%d tokens)",
        trainer.step,
        args.max_steps,
        args.micro_batch_size * args.grad_accum_steps * args.seq_len,
    )

    t0 = time.perf_counter()
    step_times: list[float] = []

    while trainer.step < args.max_steps:
        ts = time.perf_counter()
        entry = trainer.train_step()
        step_times.append(time.perf_counter() - ts)

        if trainer.step % args.log_interval == 0 or trainer.step == 1:
            avg_ms = 1000 * sum(step_times[-args.log_interval:]) / len(step_times[-args.log_interval:])
            tokens_per_sec = (
                args.micro_batch_size * args.grad_accum_steps * args.seq_len / (avg_ms / 1000)
            )
            log.info(
                "step %d/%d | loss=%.4f | lr=%.2e | gnorm=%.3f | %.0f tok/s",
                trainer.step,
                args.max_steps,
                entry["loss"],
                entry["lr"],
                entry["grad_norm"],
                tokens_per_sec,
            )

    total_time = time.perf_counter() - t0
    log.info(
        "Training complete in %.1f s  (%.0f steps/s)",
        total_time,
        args.max_steps / total_time,
    )

    # Final checkpoint
    final_ckpt = args.output_dir / "final_ckpt.pt"
    trainer.save(final_ckpt)
    log.info("Final checkpoint saved to %s", final_ckpt)


if __name__ == "__main__":
    main()
