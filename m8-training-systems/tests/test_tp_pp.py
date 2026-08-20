"""Acceptance tests for frozen phase 8.1 (TP + GPipe)."""

from __future__ import annotations

import copy
import importlib.util
import os
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from torch.nn import functional as F

from minidist.comm import run_distributed


def _load_reference(filename: str, module_name: str):
    path = Path(__file__).parents[1] / "reference" / "8.1-tp-pp" / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if os.environ.get("M8_REFERENCE") == "1":
    _layers = _load_reference("layers_solution.py", "tp_layers_solution")
    _schedule = _load_reference("schedule_solution.py", "pp_schedule_solution")
    _runner = _load_reference("runner_solution.py", "pp_runner_solution")
    TPMLP = _layers.TPMLP
    PipelineRunner = _runner.PipelineRunner
    bubble_fraction = _schedule.bubble_fraction
    gpipe_schedule = _schedule.gpipe_schedule
else:
    from minipp.runner import PipelineRunner
    from minipp.schedule import bubble_fraction, gpipe_schedule
    from minitp.layers import TPMLP


def _full_weights(dim: int, seed: int):
    g1 = torch.Generator(device="cpu").manual_seed(seed)
    g2 = torch.Generator(device="cpu").manual_seed(seed + 1)
    return (
        torch.randn(4 * dim, dim, generator=g1) * 0.02,
        torch.randn(dim, 4 * dim, generator=g2) * 0.02,
    )


def _tp_equivalence_worker(rank: int, world_size: int, seed: int):
    dim = 6
    torch.manual_seed(seed + 99)
    x = torch.randn(5, dim, requires_grad=True)
    model = TPMLP(dim, world_size, rank, seed)

    collective_counts = {"all_reduce": 0, "all_gather": 0}
    original_all_reduce = dist.all_reduce
    original_all_gather = dist.all_gather

    def counted_all_reduce(*args, **kwargs):
        collective_counts["all_reduce"] += 1
        return original_all_reduce(*args, **kwargs)

    def counted_all_gather(*args, **kwargs):
        collective_counts["all_gather"] += 1
        return original_all_gather(*args, **kwargs)

    dist.all_reduce = counted_all_reduce
    dist.all_gather = counted_all_gather
    try:
        output = model(x)
        loss = output.square().mean()
        loss.backward()
    finally:
        dist.all_reduce = original_all_reduce
        dist.all_gather = original_all_gather

    x_full = x.detach().clone().requires_grad_(True)
    w1_data, w2_data = _full_weights(dim, seed)
    w1 = w1_data.requires_grad_(True)
    w2 = w2_data.requires_grad_(True)
    expected = F.linear(F.gelu(F.linear(x_full, w1)), w2)
    expected.square().mean().backward()
    col_rows = (4 * dim) // world_size
    row_cols = (4 * dim) // world_size
    return {
        "output": output.detach().tolist(),
        "expected": expected.detach().tolist(),
        "x_grad": x.grad.tolist(),
        "x_grad_expected": x_full.grad.tolist(),
        "column_grad": model.column.weight.grad.tolist(),
        "column_grad_expected": w1.grad[
            rank * col_rows : (rank + 1) * col_rows
        ].tolist(),
        "row_grad": model.row.weight.grad.tolist(),
        "row_grad_expected": w2.grad[
            :, rank * row_cols : (rank + 1) * row_cols
        ].tolist(),
        "collective_counts": collective_counts,
    }


def _tp_determinism_worker(rank: int, world_size: int, seed: int):
    torch.manual_seed(seed + 7)
    x = torch.randn(3, 4)
    return TPMLP(4, world_size, rank, seed)(x).detach().tolist()


@pytest.fixture(scope="module")
def tp_equivalence_results():
    return run_distributed(_tp_equivalence_worker, 2, 113)


def test_t1_tp_forward_matches_full_mlp(tp_equivalence_results):
    for result in tp_equivalence_results:
        torch.testing.assert_close(
            torch.tensor(result["output"]),
            torch.tensor(result["expected"]),
            rtol=0,
            atol=1e-6,
        )
    assert tp_equivalence_results[0]["output"] == tp_equivalence_results[1]["output"]


def test_t2_tp_backward_matches_full_mlp(tp_equivalence_results):
    for result in tp_equivalence_results:
        for actual, expected in (
            ("x_grad", "x_grad_expected"),
            ("column_grad", "column_grad_expected"),
            ("row_grad", "row_grad_expected"),
        ):
            torch.testing.assert_close(
                torch.tensor(result[actual]),
                torch.tensor(result[expected]),
                rtol=1e-6,
                atol=1e-8,
            )
        assert result["collective_counts"] == {"all_reduce": 2, "all_gather": 0}


@pytest.mark.parametrize("num_stages,num_microbatches", [(1, 1), (2, 1), (2, 4), (4, 3)])
def test_t3_gpipe_schedule_and_bubble(num_stages, num_microbatches):
    schedule = gpipe_schedule(num_stages, num_microbatches)
    expected_ticks = 2 * (num_microbatches + num_stages - 1)
    assert len(schedule) == expected_ticks

    where = {}
    for tick, operations in enumerate(schedule):
        stages_this_tick = [stage for stage, _, _ in operations]
        assert len(stages_this_tick) == len(set(stages_this_tick))
        for stage, microbatch, direction in operations:
            assert 0 <= stage < num_stages
            assert 0 <= microbatch < num_microbatches
            assert direction in {"F", "B"}
            key = (stage, microbatch, direction)
            assert key not in where
            where[key] = tick

    assert len(where) == 2 * num_stages * num_microbatches
    for stage in range(num_stages):
        for microbatch in range(num_microbatches):
            forward_tick = where[(stage, microbatch, "F")]
            backward_tick = where[(stage, microbatch, "B")]
            assert forward_tick < backward_tick
            if stage:
                assert where[(stage - 1, microbatch, "F")] < forward_tick
            if stage + 1 < num_stages:
                assert where[(stage + 1, microbatch, "B")] < backward_tick
            # Flush GPipe starts no backward until every micro-batch reached
            # the final stage.
            assert backward_tick > max(
                where[(num_stages - 1, mb, "F")] for mb in range(num_microbatches)
            )

    occupied = sum(len(operations) for operations in schedule)
    actual_idle = 1.0 - occupied / (len(schedule) * num_stages)
    assert bubble_fraction(num_stages, num_microbatches) == pytest.approx(actual_idle)
    assert bubble_fraction(num_stages, num_microbatches) == pytest.approx(
        (num_stages - 1) / (num_microbatches + num_stages - 1)
    )


def _make_pipeline(seed: int):
    torch.manual_seed(seed)
    return [
        torch.nn.Sequential(torch.nn.Linear(6, 8), torch.nn.GELU()),
        torch.nn.Sequential(torch.nn.Linear(8, 3)),
    ]


def test_t4_pipeline_gradients_match_full_batch():
    torch.manual_seed(991)
    x = torch.randn(8, 6)
    y = torch.randn(8, 3)
    loss_fn = torch.nn.MSELoss(reduction="mean")

    for microbatches in (1, 2, 4):
        stages = _make_pipeline(45)
        baseline_stages = copy.deepcopy(stages)
        baseline = torch.nn.Sequential(*baseline_stages)
        expected_loss = loss_fn(baseline(x), y)
        expected_loss.backward()

        observed_batch_sizes = []

        def recording_loss(prediction, target):
            observed_batch_sizes.append(len(prediction))
            return F.mse_loss(prediction, target)

        loss = PipelineRunner(stages).train_step(x, y, microbatches, recording_loss)
        assert loss.ndim == 0
        assert observed_batch_sizes == [len(x) // microbatches] * microbatches
        torch.testing.assert_close(loss, expected_loss.detach(), rtol=1e-6, atol=1e-7)
        for stage, expected_stage in zip(stages, baseline_stages):
            for parameter, expected_parameter in zip(
                stage.parameters(), expected_stage.parameters()
            ):
                if microbatches == 1:
                    assert torch.equal(parameter.grad, expected_parameter.grad)
                else:
                    torch.testing.assert_close(
                        parameter.grad,
                        expected_parameter.grad,
                        rtol=1e-6,
                        atol=1e-7,
                    )


def test_t4_mean_loss_scaling_semantics():
    """The semantic gate remains useful despite the frozen bitwise blocker."""
    torch.manual_seed(404)
    x = torch.randn(8, 6)
    y = torch.randn(8, 3)
    stages = _make_pipeline(46)
    baseline_stages = copy.deepcopy(stages)
    F.mse_loss(torch.nn.Sequential(*baseline_stages)(x), y).backward()
    PipelineRunner(stages).train_step(x, y, 4, F.mse_loss)
    for stage, expected_stage in zip(stages, baseline_stages):
        for parameter, expected_parameter in zip(stage.parameters(), expected_stage.parameters()):
            torch.testing.assert_close(
                parameter.grad, expected_parameter.grad, rtol=2e-6, atol=2e-7
            )


def test_t5_pipeline_boundaries():
    schedule = gpipe_schedule(3, 1)
    assert len(schedule) == 6
    assert bubble_fraction(3, 1) == pytest.approx(2 / 3)
    with pytest.raises(ValueError):
        PipelineRunner([torch.nn.Sequential(torch.nn.Linear(2, 2)), torch.nn.Sequential()])
    runner = PipelineRunner([torch.nn.Linear(2, 2)])
    with pytest.raises(ValueError):
        runner.train_step(torch.randn(5, 2), torch.randn(5, 2), 2, F.mse_loss)


def test_t6_tp_and_pp_are_deterministic():
    first = run_distributed(_tp_determinism_worker, 2, 717)
    second = run_distributed(_tp_determinism_worker, 2, 717)
    assert first == second

    torch.manual_seed(12)
    x = torch.randn(4, 6)
    y = torch.randn(4, 3)
    runs = []
    for _ in range(2):
        stages = _make_pipeline(88)
        loss = PipelineRunner(stages).train_step(x, y, 2, F.mse_loss)
        runs.append((loss, [parameter.grad.clone() for stage in stages for parameter in stage.parameters()]))
    assert torch.equal(runs[0][0], runs[1][0])
    assert all(torch.equal(a, b) for a, b in zip(runs[0][1], runs[1][1]))
