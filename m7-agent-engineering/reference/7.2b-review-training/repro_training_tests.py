"""High-value training-loop tests for use after the timed review."""

import importlib
import os
import sys
from pathlib import Path

import torch

PR = Path(os.environ.get(
    "TRAINING_REPRO_PR",
    Path(__file__).parents[2] / "exercises" / "7.2b-review-training" / "pr",
))
sys.path.insert(0, str(PR))
trainer = importlib.import_module("trainer")
checkpoint = importlib.import_module("checkpoint")
scheduler = importlib.import_module("scheduler")


def _model():
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    return model


def test_accumulation_matches_equivalent_large_batch():
    micro_model = _model()
    micro_optim = torch.optim.SGD(micro_model.parameters(), lr=0.1)
    one = (torch.ones(1, 1), torch.ones(1, 1))
    trainer.train_epoch(
        micro_model, [one, one], micro_optim,
        accumulation_steps=2, max_grad_norm=1e9, history=[]
    )

    large_model = _model()
    large_optim = torch.optim.SGD(large_model.parameters(), lr=0.1)
    large = (torch.ones(2, 1), torch.ones(2, 1))
    trainer.train_epoch(large_model, [large], large_optim, max_grad_norm=1e9, history=[])
    assert torch.allclose(micro_model.weight, large_model.weight, atol=1e-7)


def test_checkpoint_restores_scheduler_progress(tmp_path):
    model = _model()
    optim = torch.optim.SGD(model.parameters(), lr=0.1)
    schedule = scheduler.CosineSchedule(optim, 1, 4)
    schedule.step()
    path = tmp_path / "state.pt"
    checkpoint.save_checkpoint(path, model, optim, schedule, None, 1)

    restored_model = _model()
    restored_optim = torch.optim.SGD(restored_model.parameters(), lr=0.1)
    restored_schedule = scheduler.CosineSchedule(restored_optim, 1, 4)
    checkpoint.load_checkpoint(path, restored_model, restored_optim, restored_schedule)
    assert restored_schedule.step_count == 1
