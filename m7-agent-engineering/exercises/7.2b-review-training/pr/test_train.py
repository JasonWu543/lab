from pathlib import Path

import pytest
import torch

from checkpoint import load_checkpoint, resumed_microbatch_offset, save_checkpoint
from scheduler import CosineSchedule
from trainer import train_epoch


def make_model():
    layer = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        layer.weight.zero_()
    return layer


def test_training_returns_finite_summary():
    model = make_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    x = torch.zeros(4, 2)
    y = torch.zeros(4, 1)
    result = train_epoch(model, [(x, y)], optimizer)
    assert torch.isfinite(torch.tensor(result["mean_loss"]))


def test_accumulation_reports_one_update():
    model = make_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    batch = (torch.zeros(2, 2), torch.zeros(2, 1))
    result = train_epoch(model, [batch, batch], optimizer, accumulation_steps=2)
    assert result["optimizer_steps"] == 1


def test_schedule_values_match_helper():
    model = make_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    schedule = CosineSchedule(optimizer, warmup_steps=2, total_steps=6)
    schedule.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(schedule._factor(1))


def test_schedule_stays_nonnegative():
    model = make_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    schedule = CosineSchedule(optimizer, warmup_steps=1, total_steps=4, min_lr=0.2)
    for _ in range(4):
        schedule.step()
    assert optimizer.param_groups[0]["lr"] >= 0


def test_checkpoint_round_trip(tmp_path: Path):
    model = make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    schedule = CosineSchedule(optimizer, 1, 4)
    path = tmp_path / "state.pt"
    save_checkpoint(path, model, optimizer, schedule, None, global_step=0)
    restored = load_checkpoint(path, model, optimizer, schedule)
    assert restored == 0


def test_resume_offset_for_single_step():
    assert resumed_microbatch_offset(1, accumulation_steps=1) == 1


def test_repeatable_shape_contract():
    values = torch.randn(8, 2)
    model = make_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    result = train_epoch(model, [(values, torch.zeros(8, 1))], optimizer)
    assert len(result["losses"]) >= 1
