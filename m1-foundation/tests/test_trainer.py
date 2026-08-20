"""
tests/test_trainer.py — Phase 1.3 Trainer test suite (T1–T10)

Run:
    cd m1-foundation && python3 -m pytest tests/test_trainer.py -x -q

All tests run on CPU with deterministic algorithms where needed.
Total target: ≤ 90 seconds.
"""

from __future__ import annotations

import copy
import math
import os
import random
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from minilm.training.data import PackedDataset, make_dataloader, write_memmap
from minilm.training.checkpoint import (
    CheckpointCorruptError,
    load_checkpoint,
    save_checkpoint,
)
from minilm.training.scheduler import lr_at
from minilm.training.trainer import NonFiniteLossError, TrainConfig, Trainer


# ---------------------------------------------------------------------------
# Tiny model definition (self-contained — does NOT import minilm.model)
# ---------------------------------------------------------------------------

class TinyLM(nn.Module):
    """Minimal language model for testing the Trainer independently of Phase 1.2."""

    def __init__(self, vocab_size: int = 64, hidden: int = 32) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:  # (B, T) → (B, T, V)
        x = self.embed(input_ids)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


VOCAB = 64
SEQ_LEN = 8


def _write_tokens(tmp_path: Path, n: int = 512, seed: int = 0) -> Path:
    """Write random token ids to a tmp memmap file, return prefix path."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    ids = rng.integers(0, VOCAB, size=n, dtype=np.uint16)
    prefix = tmp_path / "tokens"
    write_memmap(ids.tolist(), prefix)
    return prefix


def _make_dataset(tmp_path: Path, n: int = 512) -> PackedDataset:
    prefix = _write_tokens(tmp_path, n=n)
    return PackedDataset(prefix, seq_len=SEQ_LEN)


def _make_model(seed: int = 42) -> TinyLM:
    torch.manual_seed(seed)
    return TinyLM(vocab_size=VOCAB, hidden=32)


# ---------------------------------------------------------------------------
# T1: memmap write/read roundtrip; PackedDataset correctness
# ---------------------------------------------------------------------------

class TestT1Memmap:
    def test_roundtrip(self, tmp_path):
        """write_memmap then read back gives identical token sequence."""
        ids = list(range(100))
        prefix = tmp_path / "test"
        write_memmap(ids, prefix)

        bin_path = prefix.with_suffix(".bin")
        loaded = np.fromfile(str(bin_path), dtype=np.uint16).tolist()
        assert loaded == ids, f"Roundtrip failed: {loaded[:10]} != {ids[:10]}"

    def test_meta_json_written(self, tmp_path):
        """meta.json is created alongside .bin."""
        import json
        ids = list(range(50))
        prefix = tmp_path / "tok"
        write_memmap(ids, prefix)
        meta_path = Path(str(prefix) + ".meta.json")
        assert meta_path.exists(), "meta.json not created"
        meta = json.loads(meta_path.read_text())
        assert meta["n_tokens"] == 50
        assert meta["dtype"] == "uint16"

    def test_packed_dataset_len(self, tmp_path):
        """__len__ == floor((N - 1) / seq_len)."""
        n_tokens = 100
        seq_len = 8
        expected_len = (n_tokens - 1) // seq_len  # = 12
        prefix = _write_tokens(tmp_path, n=n_tokens)
        ds = PackedDataset(prefix, seq_len=seq_len)
        assert len(ds) == expected_len, (
            f"Expected len={expected_len}, got {len(ds)}. "
            "Hint: formula is floor((N-1) / seq_len)."
        )

    def test_xy_shift(self, tmp_path):
        """x and y differ by a one-token right shift: y[i] == x[i+1] for all but last."""
        prefix = _write_tokens(tmp_path, n=200)
        ds = PackedDataset(prefix, seq_len=SEQ_LEN)

        for i in range(min(5, len(ds))):
            x, y = ds[i]
            assert x.shape == (SEQ_LEN,), f"x.shape={x.shape}"
            assert y.shape == (SEQ_LEN,), f"y.shape={y.shape}"
            assert x.dtype == torch.int64, f"x.dtype={x.dtype}"
            assert y.dtype == torch.int64, f"y.dtype={y.dtype}"
            # y[j] == x[j+1] for j in 0..SEQ_LEN-2
            assert torch.equal(y[:-1], x[1:]), (
                f"Sample {i}: y[:-1] != x[1:] — y is not a right-shift of x"
            )

    def test_consecutive_samples_overlap(self, tmp_path):
        """Consecutive samples share the boundary token (end of x[i] == start of y[i+1])."""
        prefix = _write_tokens(tmp_path, n=200)
        ds = PackedDataset(prefix, seq_len=SEQ_LEN)

        for i in range(min(4, len(ds) - 1)):
            _, y_i = ds[i]
            x_next, _ = ds[i + 1]
            assert y_i[-1].item() == x_next[0].item(), (
                f"Sample {i}/{i+1} boundary mismatch: y[-1]={y_i[-1]} x_next[0]={x_next[0]}"
            )


# ---------------------------------------------------------------------------
# T2: reproducible dataloader ordering
# ---------------------------------------------------------------------------

class TestT2DataloaderSeed:
    def _collect_indices(self, ds, seed, shuffle=True, bs=4) -> list[int]:
        """Collect first batch of token ids from dataloader to check order."""
        dl = make_dataloader(ds, batch_size=bs, shuffle=shuffle, seed=seed)
        xs = []
        for x, _ in dl:
            xs.extend(x[:, 0].tolist())  # first token of each seq as proxy for order
        return xs

    def test_same_seed_same_order(self, tmp_path):
        """Two dataloaders with the same seed produce identical batch ordering."""
        ds = _make_dataset(tmp_path)
        order_a = self._collect_indices(ds, seed=7)
        order_b = self._collect_indices(ds, seed=7)
        assert order_a == order_b, (
            "Same seed produced different ordering — generator not seeded correctly."
        )

    def test_different_seed_different_order(self, tmp_path):
        """Two dataloaders with different seeds (very likely) differ in ordering."""
        ds = _make_dataset(tmp_path, n=256)
        order_a = self._collect_indices(ds, seed=1)
        order_b = self._collect_indices(ds, seed=99)
        assert order_a != order_b, (
            "Different seeds produced identical ordering — seed not applied?"
        )

    def test_no_shuffle_deterministic(self, tmp_path):
        """shuffle=False always gives the same order regardless of seed."""
        ds = _make_dataset(tmp_path)
        order_a = self._collect_indices(ds, seed=1, shuffle=False)
        order_b = self._collect_indices(ds, seed=99, shuffle=False)
        assert order_a == order_b


# ---------------------------------------------------------------------------
# T3: lr_at curve snapshot
# ---------------------------------------------------------------------------

class TestT3Scheduler:
    MAX_LR = 1e-3
    MIN_LR = 1e-4
    WARMUP = 10
    TOTAL = 100

    def _lr(self, step):
        return lr_at(
            step,
            max_lr=self.MAX_LR,
            min_lr=self.MIN_LR,
            warmup_steps=self.WARMUP,
            total_steps=self.TOTAL,
        )

    def test_warmup_starts_at_zero(self):
        assert self._lr(0) == pytest.approx(0.0, abs=1e-12), (
            "lr_at(0) should be 0 (start of linear warmup)"
        )

    def test_warmup_endpoint(self):
        """At step == warmup_steps - 1, lr should be (warmup-1)/warmup * max_lr."""
        expected = self.MAX_LR * (self.WARMUP - 1) / self.WARMUP
        got = self._lr(self.WARMUP - 1)
        assert got == pytest.approx(expected, rel=1e-9), (
            f"Warmup endpoint wrong: expected {expected}, got {got}"
        )

    def test_at_warmup_is_max_lr(self):
        """Cosine phase starts: lr_at(warmup) should equal max_lr (progress=0 → cos(0)=1)."""
        got = self._lr(self.WARMUP)
        # progress=0 → cosine_factor=1 → lr = min_lr + (max_lr-min_lr)*1 = max_lr
        assert got == pytest.approx(self.MAX_LR, rel=1e-9), (
            f"lr_at(warmup_steps) should equal max_lr={self.MAX_LR}, got {got}"
        )

    def test_cosine_midpoint(self):
        """Midpoint of cosine phase (progress=0.5) → cosine_factor=0.5."""
        mid_step = self.WARMUP + (self.TOTAL - self.WARMUP) // 2  # step 55
        progress = (mid_step - self.WARMUP) / (self.TOTAL - self.WARMUP)
        cos_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        expected = self.MIN_LR + (self.MAX_LR - self.MIN_LR) * cos_factor
        got = self._lr(mid_step)
        assert got == pytest.approx(expected, rel=1e-9), (
            f"Cosine midpoint wrong: expected {expected}, got {got}"
        )

    def test_after_total_is_min_lr(self):
        """After total_steps, lr is constant at min_lr."""
        for step in [self.TOTAL, self.TOTAL + 1, self.TOTAL + 100]:
            got = self._lr(step)
            assert got == pytest.approx(self.MIN_LR, rel=1e-9), (
                f"lr_at({step}) should be min_lr={self.MIN_LR}, got {got}"
            )

    def test_lr_monotone_decreasing_in_cosine_phase(self):
        """LR is non-increasing during the cosine phase."""
        lrs = [self._lr(s) for s in range(self.WARMUP, self.TOTAL + 1)]
        for i in range(1, len(lrs)):
            assert lrs[i] <= lrs[i - 1] + 1e-12, (
                f"LR increased at step {self.WARMUP + i}: {lrs[i-1]} → {lrs[i]}"
            )


# ---------------------------------------------------------------------------
# T4: grad accumulation equivalence (HARD)
# ---------------------------------------------------------------------------

class TestT4GradAccum:
    """
    accum=4 × micro_bs=2  should equal  accum=1 × batch=8
    Same seed → same data → parameter updates within 1e-6.
    """

    def _run_one_step(self, accum: int, micro_bs: int, tmp_path: Path, tag: str,
                      inflate: float = 1.0, grad_clip: float = 1.0):
        """Returns (parameter values, log dict) after one optimizer step."""
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(42)

        n_tokens = (accum * micro_bs + 2) * SEQ_LEN + 10
        prefix = _write_tokens(tmp_path / tag, n=n_tokens, seed=0)
        ds = PackedDataset(prefix, seq_len=SEQ_LEN)
        dl = make_dataloader(ds, batch_size=micro_bs, shuffle=False, seed=42)

        model = _make_model(seed=42)
        if inflate != 1.0:
            with torch.no_grad():
                for p in model.parameters():
                    p.mul_(inflate)
        cfg = TrainConfig(
            max_steps=1,
            micro_batch_size=micro_bs,
            grad_accum_steps=accum,
            max_lr=1e-3,
            min_lr=1e-4,
            warmup_steps=0,  # constant lr for comparability
            grad_clip=grad_clip,
            seed=42,
        )
        trainer = Trainer(model, cfg, dl)
        log = trainer.train_step()

        return [p.detach().clone() for p in model.parameters()], log

    def test_accum_equivalence(self, tmp_path):
        torch.use_deterministic_algorithms(True)
        params_accum, _ = self._run_one_step(accum=4, micro_bs=2, tmp_path=tmp_path, tag="accum")
        params_batch, _ = self._run_one_step(accum=1, micro_bs=8, tmp_path=tmp_path, tag="batch")

        for i, (pa, pb) in enumerate(zip(params_accum, params_batch)):
            max_diff = (pa - pb).abs().max().item()
            assert max_diff < 1e-6, (
                f"Param {i}: grad-accum vs large-batch differ by {max_diff:.2e} > 1e-6. "
                "Hint: think about WHEN you normalize the loss by grad_accum_steps — "
                "what does backward() accumulate?"
            )

    def test_accum_equivalence_with_active_clip(self, tmp_path):
        """Same equivalence, but with gradients large enough that clipping fires.
        Clipping per micro-batch instead of once on the accumulated gradient
        passes the plain equivalence test (clip is a no-op there) but fails here."""
        torch.use_deterministic_algorithms(True)
        params_accum, log_a = self._run_one_step(
            accum=4, micro_bs=2, tmp_path=tmp_path, tag="accum_clip",
            inflate=30.0, grad_clip=0.5)
        params_batch, log_b = self._run_one_step(
            accum=1, micro_bs=8, tmp_path=tmp_path, tag="batch_clip",
            inflate=30.0, grad_clip=0.5)

        # sanity: clipping must actually be active in both runs
        assert log_a["grad_norm"] > 0.5 and log_b["grad_norm"] > 0.5, (
            f"Setup failed: pre-clip grad norms ({log_a['grad_norm']:.3f}, "
            f"{log_b['grad_norm']:.3f}) should exceed grad_clip=0.5"
        )
        for i, (pa, pb) in enumerate(zip(params_accum, params_batch)):
            max_diff = (pa - pb).abs().max().item()
            assert max_diff < 1e-6, (
                f"Param {i}: differ by {max_diff:.2e} with active clipping. "
                "Hint: should clip act on each micro-batch's gradient, or once on "
                "the accumulated gradient? When do the two differ?"
            )


# ---------------------------------------------------------------------------
# T5: gradient clipping
# ---------------------------------------------------------------------------

class TestT5Clipping:
    def _global_grad_norm(self, model: nn.Module) -> float:
        total = sum(p.grad.norm() ** 2 for p in model.parameters() if p.grad is not None)
        return float(total.sqrt())

    def test_large_gradient_clipped(self, tmp_path):
        """After clip, global grad norm == grad_clip."""
        grad_clip = 1.0
        prefix = _write_tokens(tmp_path, n=512)
        ds = PackedDataset(prefix, seq_len=SEQ_LEN)
        dl = make_dataloader(ds, batch_size=4, shuffle=False, seed=0)

        model = _make_model()
        cfg = TrainConfig(max_steps=1, micro_batch_size=4, grad_clip=grad_clip, warmup_steps=0)
        trainer = Trainer(model, cfg, dl)

        # Manually inflate gradients to make norm >> clip threshold
        model.train()
        x, y = next(iter(dl))
        logits = model(x)
        V = logits.size(-1)
        loss = F.cross_entropy(logits.view(-1, V), y.view(-1))
        loss.backward()
        for p in model.parameters():
            if p.grad is not None:
                p.grad.data *= 1000.0  # force large norm

        pre_norm = self._global_grad_norm(model)
        assert pre_norm > 100.0, f"Setup failed: pre-clip norm={pre_norm:.1f} not large"

        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        post_norm = self._global_grad_norm(model)

        assert post_norm == pytest.approx(grad_clip, rel=1e-5), (
            f"After clip, norm={post_norm:.6f}, expected {grad_clip}. "
            "Clipping did not normalize to exactly grad_clip."
        )

    def test_small_gradient_unclipped(self, tmp_path):
        """Small gradients (norm < grad_clip) are not modified."""
        grad_clip = 100.0  # large clip threshold
        prefix = _write_tokens(tmp_path, n=256)
        ds = PackedDataset(prefix, seq_len=SEQ_LEN)
        dl = make_dataloader(ds, batch_size=2, shuffle=False, seed=0)
        model = _make_model()

        model.train()
        x, y = next(iter(dl))
        logits = model(x)
        V = logits.size(-1)
        loss = F.cross_entropy(logits.view(-1, V), y.view(-1))
        loss.backward()

        # Capture gradients before clip
        grads_before = [p.grad.clone() for p in model.parameters() if p.grad is not None]
        pre_norm = self._global_grad_norm(model)

        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        grads_after = [p.grad.clone() for p in model.parameters() if p.grad is not None]
        for gb, ga in zip(grads_before, grads_after):
            assert torch.allclose(gb, ga), (
                f"Small gradient (norm={pre_norm:.4f}) was modified by clip={grad_clip}"
            )


# ---------------------------------------------------------------------------
# T6: checkpoint resume (HARD) — loss trajectory must match exactly
# ---------------------------------------------------------------------------

class TestT6Resume:

    def _make_setup(self, tmp_path: Path, tag: str, *, n_tokens: int,
                    total_steps: int, accum: int, micro_bs: int):
        prefix = _write_tokens(tmp_path / tag, n=n_tokens, seed=7)
        ds = PackedDataset(prefix, seq_len=SEQ_LEN)
        dl = make_dataloader(ds, batch_size=micro_bs, shuffle=True, seed=42)
        model = _make_model(seed=42)
        cfg = TrainConfig(
            max_steps=total_steps,
            micro_batch_size=micro_bs,
            grad_accum_steps=accum,
            warmup_steps=0,
            seed=42,
        )
        return model, cfg, dl

    def _check_resume(self, tmp_path: Path, *, n_tokens: int, total_steps: int,
                      split_at: int, accum: int = 1, micro_bs: int = 2,
                      hint: str = ""):
        """Baseline (uninterrupted) vs save-at-split_at-then-resume:
        the resumed loss trajectory must match the baseline step by step.
        NOTE: max_steps is shared by all runs so the LR schedule is identical."""
        torch.use_deterministic_algorithms(True)

        # --- Baseline: uninterrupted run ---
        model_b, cfg_b, dl_b = self._make_setup(
            tmp_path, "base", n_tokens=n_tokens, total_steps=total_steps,
            accum=accum, micro_bs=micro_bs)
        trainer_b = Trainer(model_b, cfg_b, dl_b)
        all_logs = trainer_b.train()
        baseline_logs = all_logs[split_at:]

        # --- Run 1: identical config, stop manually at split_at, save ---
        ckpt_path = tmp_path / "ckpt.pt"
        model_r1, cfg_r1, dl_r1 = self._make_setup(
            tmp_path, "run1", n_tokens=n_tokens, total_steps=total_steps,
            accum=accum, micro_bs=micro_bs)
        trainer_r1 = Trainer(model_r1, cfg_r1, dl_r1)
        for _ in range(split_at):
            trainer_r1.train_step()
        trainer_r1.save(ckpt_path)

        # --- Run 2: "new process semantics" — rebuild everything, resume ---
        prefix = tmp_path / "run1" / "tokens"
        ds_r2 = PackedDataset(prefix, seq_len=SEQ_LEN)
        dl_r2 = make_dataloader(ds_r2, batch_size=micro_bs, shuffle=True, seed=42)
        model_r2 = _make_model(seed=42)
        cfg_r2 = TrainConfig(
            max_steps=total_steps,
            micro_batch_size=micro_bs,
            grad_accum_steps=accum,
            warmup_steps=0,
            seed=42,
        )
        trainer_r2 = Trainer(model_r2, cfg_r2, dl_r2)
        trainer_r2.resume(ckpt_path)

        resumed_losses = [e["loss"] for e in trainer_r2.train()]

        assert len(resumed_losses) == len(baseline_logs), (
            f"Expected {len(baseline_logs)} resumed steps, got {len(resumed_losses)}"
        )
        for i, (baseline, rl) in enumerate(zip(baseline_logs, resumed_losses)):
            bl = baseline["loss"]
            abs_step = split_at + i + 1
            assert bl == rl, (
                f"Resume diverged at step {abs_step}: "
                f"baseline_loss={bl:.8f}, resumed_loss={rl:.8f}, diff={abs(bl-rl):.2e}. "
                + (hint or "Hint: check that ALL state (model, optimizer, RNG, "
                           "dataloader position) is restored.")
            )

        for name, baseline_param in model_b.state_dict().items():
            assert torch.equal(baseline_param, model_r2.state_dict()[name]), \
                f"final parameter {name} is not bit-identical after resume"

        def assert_nested_equal(a, b):
            if isinstance(a, torch.Tensor):
                assert torch.equal(a, b)
            elif isinstance(a, dict):
                assert a.keys() == b.keys()
                for key in a:
                    assert_nested_equal(a[key], b[key])
            elif isinstance(a, (list, tuple)):
                assert len(a) == len(b)
                for x, y in zip(a, b):
                    assert_nested_equal(x, y)
            else:
                assert a == b

        assert_nested_equal(
            trainer_b.optimizer.state_dict(), trainer_r2.optimizer.state_dict())

    def test_resume_matches_baseline(self, tmp_path):
        # 1024 tokens → 63 batches/epoch：30 步不跨 epoch，纯恢复正确性
        self._check_resume(tmp_path, n_tokens=1024, total_steps=30, split_at=20)

    def test_resume_across_epoch_boundary(self, tmp_path):
        # 129 tokens → 16 samples → 8 batches/epoch：30 步跨 3 个 epoch，
        # split 落在 epoch 中间。快进时不处理 StopIteration 的实现会在这里挂
        self._check_resume(
            tmp_path, n_tokens=129, total_steps=30, split_at=20,
            hint="Hint: fast-forward crosses an epoch boundary here — "
                 "does your resume() handle StopIteration like _get_batch does?")

    def test_resume_with_grad_accum(self, tmp_path):
        # accum=2：一个 optimizer step 消耗 2 个 batch。
        # 快进只跳 step 个 batch 的实现会在这里数据错位
        self._check_resume(
            tmp_path, n_tokens=2048, total_steps=15, split_at=10, accum=2,
            hint="Hint: how many micro batches does ONE optimizer step consume? "
                 "Count what your fast-forward actually skips.")


# ---------------------------------------------------------------------------
# T7: atomic write — crash before os.replace leaves old checkpoint intact
# ---------------------------------------------------------------------------

class TestT7Atomicity:
    def test_crash_before_replace_leaves_old_ckpt(self, tmp_path, monkeypatch):
        ckpt_path = tmp_path / "stable.pt"
        model = _make_model()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1)

        # Save a clean checkpoint at step 1
        save_checkpoint(ckpt_path, model=model, optimizer=opt, step=1)
        assert ckpt_path.exists(), "Initial checkpoint not created"

        # Simulate crash: monkeypatch os.replace to raise before it completes
        original_replace = os.replace

        def crash_replace(src, dst):
            raise RuntimeError("simulated crash before replace")

        monkeypatch.setattr(os, "replace", crash_replace)

        with pytest.raises(RuntimeError, match="simulated crash"):
            save_checkpoint(ckpt_path, model=model, optimizer=opt, step=5)

        # After monkeypatch is restored (happens automatically at end of test)
        monkeypatch.setattr(os, "replace", original_replace)

        # Old checkpoint at step=1 must still load cleanly
        model2 = _make_model()
        opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1)
        info = load_checkpoint(ckpt_path, model=model2, optimizer=opt2)
        assert info["step"] == 1, (
            f"Old checkpoint corrupted by failed write! step={info['step']} (expected 1). "
            "Ensure writes go to .tmp first, then os.replace atomically."
        )


# ---------------------------------------------------------------------------
# T8: corruption detection — truncated file → CheckpointCorruptError
# ---------------------------------------------------------------------------

class TestT8Corruption:
    def test_truncated_file_raises(self, tmp_path):
        ckpt_path = tmp_path / "corrupt.pt"
        model = _make_model()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1)
        save_checkpoint(ckpt_path, model=model, optimizer=opt, step=3)

        # Truncate to first 10 bytes
        ckpt_path.write_bytes(ckpt_path.read_bytes()[:10])

        model2 = _make_model()
        opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1)

        with pytest.raises(CheckpointCorruptError):
            load_checkpoint(ckpt_path, model=model2, optimizer=opt2)


# ---------------------------------------------------------------------------
# T9: non-finite loss → NonFiniteLossError, params unchanged
# ---------------------------------------------------------------------------

class TestT9NonFinite:
    def test_nan_loss_raises_and_no_param_update(self, tmp_path):
        """A model that produces NaN logits triggers NonFiniteLossError; params unchanged."""

        class NaNModel(nn.Module):
            """Returns NaN logits unconditionally."""
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(VOCAB, 8)
                self.fc = nn.Linear(8, VOCAB)

            def forward(self, x):
                out = self.fc(self.embed(x))
                # Inject NaN
                return out * float("nan")

        prefix = _write_tokens(tmp_path, n=256)
        ds = PackedDataset(prefix, seq_len=SEQ_LEN)
        dl = make_dataloader(ds, batch_size=2, shuffle=False, seed=0)

        model = NaNModel()
        cfg = TrainConfig(max_steps=2, micro_batch_size=2, warmup_steps=0)
        trainer = Trainer(model, cfg, dl)

        # Capture params before
        params_before = [p.detach().clone() for p in model.parameters()]

        with pytest.raises(NonFiniteLossError):
            trainer.train_step()

        # Params must be unchanged (step should not have advanced)
        params_after = [p.detach().clone() for p in model.parameters()]
        for i, (pb, pa) in enumerate(zip(params_before, params_after)):
            assert torch.equal(pb, pa), (
                f"Param {i} was modified by a NaN step — optimizer.step() should not run."
            )

        assert trainer.step == 0, (
            f"Step advanced to {trainer.step} despite NonFiniteLossError (should stay 0)."
        )

    def test_inf_loss_raises(self, tmp_path):
        """Inf loss also triggers NonFiniteLossError."""

        class InfModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(VOCAB, 8)
                self.fc = nn.Linear(8, VOCAB)

            def forward(self, x):
                out = self.fc(self.embed(x))
                return out * float("inf")

        prefix = _write_tokens(tmp_path, n=256)
        ds = PackedDataset(prefix, seq_len=SEQ_LEN)
        dl = make_dataloader(ds, batch_size=2, shuffle=False, seed=0)

        model = InfModel()
        cfg = TrainConfig(max_steps=1, micro_batch_size=2, warmup_steps=0)
        trainer = Trainer(model, cfg, dl)

        with pytest.raises(NonFiniteLossError):
            trainer.train_step()


# ---------------------------------------------------------------------------
# T10: missing optimizer key → clean error, not silent re-init
# ---------------------------------------------------------------------------

class TestT10MissingOptimizerKey:
    def test_missing_optimizer_raises_corrupt_error(self, tmp_path):
        """A checkpoint with optimizer_state deleted raises CheckpointCorruptError."""
        ckpt_path = tmp_path / "stripped.pt"
        model = _make_model()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1)
        save_checkpoint(ckpt_path, model=model, optimizer=opt, step=7)

        # Strip optimizer_state from the checkpoint
        payload = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        del payload["optimizer_state"]
        torch.save(payload, str(ckpt_path))

        model2 = _make_model()
        opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1)

        with pytest.raises(CheckpointCorruptError):
            load_checkpoint(ckpt_path, model=model2, optimizer=opt2)
