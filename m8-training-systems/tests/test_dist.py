"""Frozen acceptance tests for Phase 8.0 (at most six mp.spawn calls)."""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn.functional as F

from minidist.bucket import allreduce_gradients, partition_buckets
from minidist.comm import run_distributed
from minidist.zero import Zero1Optimizer, shard_params, zero2_reduce_gradients


def _model() -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(5, 7), torch.nn.Tanh(), torch.nn.Linear(7, 3)
    )


def _flat_params(model: torch.nn.Module) -> torch.Tensor:
    return torch.cat([parameter.detach().reshape(-1) for parameter in model.parameters()])


def _ddp_equivalence_worker(rank: int, world_size: int) -> dict:
    torch.set_num_threads(1)
    torch.manual_seed(8080)
    initial = _model()
    x, target = torch.randn(8, 5), torch.randn(8, 3)

    baseline = copy.deepcopy(initial)
    F.mse_loss(baseline(x), target).backward()
    expected = [parameter.grad.clone() for parameter in baseline.parameters()]

    caps = [1, 1 << 30, 128]
    close, errors, cap_grads = [], [], []
    for cap in caps:
        candidate = copy.deepcopy(initial)
        lo, hi = rank * 4, (rank + 1) * 4
        F.mse_loss(candidate(x[lo:hi]), target[lo:hi]).backward()
        allreduce_gradients(candidate, cap)
        close.append(
            all(
                torch.allclose(p.grad, want, rtol=1e-6, atol=1e-7)
                for p, want in zip(candidate.parameters(), expected)
            )
        )
        errors.append(
            max(
                (p.grad - want).abs().max().item()
                for p, want in zip(candidate.parameters(), expected)
            )
        )
        cap_grads.append(
            torch.cat([parameter.grad.reshape(-1) for parameter in candidate.parameters()]).tolist()
        )
    return {"close": close, "max_abs_error": errors, "cap_grads": cap_grads}


def _zero1_worker(rank: int, world_size: int) -> dict:
    torch.set_num_threads(1)
    torch.manual_seed(8081)
    distributed = _model()
    baseline = copy.deepcopy(distributed)
    optimizer = Zero1Optimizer(list(distributed.parameters()), lr=3e-3)
    baseline_optimizer = torch.optim.AdamW(
        baseline.parameters(), lr=3e-3, weight_decay=0.0
    )
    generator = torch.Generator().manual_seed(991)
    for _ in range(20):
        x = torch.randn(8, 5, generator=generator)
        target = torch.randn(8, 3, generator=generator)
        distributed.zero_grad(set_to_none=True)
        lo, hi = rank * 4, (rank + 1) * 4
        F.mse_loss(distributed(x[lo:hi]), target[lo:hi]).backward()
        allreduce_gradients(distributed, 128)
        optimizer.step()

        baseline_optimizer.zero_grad(set_to_none=True)
        F.mse_loss(baseline(x), target).backward()
        baseline_optimizer.step()

    params = list(distributed.parameters())
    owned = shard_params(params, world_size)[rank]
    expected_bytes = sum(params[i].numel() * 8 + 4 for i in owned)
    flat = _flat_params(distributed)
    return {
        "exact": torch.equal(flat, _flat_params(baseline)),
        "close": torch.allclose(flat, _flat_params(baseline), rtol=1e-6, atol=1e-7),
        "max_abs_error": (flat - _flat_params(baseline)).abs().max().item(),
        "params": flat.tolist(),
        "state_bytes": optimizer.state_bytes(),
        "expected_bytes": expected_bytes,
    }


def _zero2_worker(rank: int, world_size: int) -> dict:
    torch.set_num_threads(1)
    torch.manual_seed(8082)
    model = _model()
    x = torch.randn(4, 5) + rank
    target = torch.randn(4, 3)
    F.mse_loss(model(x), target).backward()
    before = [parameter.grad.clone().tolist() for parameter in model.parameters()]
    # 故意不同于 shard_params 的默认结果，确保实现使用调用方传入的 ownership。
    shards = [[0, 2], [1, 3]]
    zero2_reduce_gradients(model, shards)
    after = [
        None if parameter.grad is None else parameter.grad.tolist()
        for parameter in model.parameters()
    ]
    return {"before": before, "after": after, "owned": shards[rank]}


def _world_one_worker(rank: int, world_size: int) -> dict:
    torch.manual_seed(8083)
    model = _model()
    baseline = copy.deepcopy(model)
    x, target = torch.randn(6, 5), torch.randn(6, 3)
    F.mse_loss(model(x), target).backward()
    F.mse_loss(baseline(x), target).backward()
    allreduce_gradients(model, 64)
    grads_equal = all(
        torch.equal(left.grad, right.grad)
        for left, right in zip(model.parameters(), baseline.parameters())
    )
    optimizer = Zero1Optimizer(list(model.parameters()), lr=1e-2)
    plain = torch.optim.AdamW(baseline.parameters(), lr=1e-2, weight_decay=0.0)
    optimizer.step()
    plain.step()
    return {
        "grads_equal": grads_equal,
        "params_equal": torch.equal(_flat_params(model), _flat_params(baseline)),
    }


def _determinism_worker(rank: int, world_size: int) -> list[float]:
    torch.set_num_threads(1)
    torch.manual_seed(8084)
    model = _model()
    x = torch.randn(4, 5) + rank * 0.25
    F.mse_loss(model(x), torch.zeros(4, 3)).backward()
    allreduce_gradients(model, 96)
    return torch.cat([p.grad.reshape(-1) for p in model.parameters()]).tolist()


@pytest.fixture(scope="module")
def zero1_results() -> list[dict]:
    """Share the expensive 20-step spawn between T3 and T5."""
    return run_distributed(_zero1_worker, 2)


def test_t1_partition_buckets_closed_form() -> None:
    params = [
        torch.nn.Parameter(torch.empty(2)),
        torch.nn.Parameter(torch.empty(3)),
        torch.nn.Parameter(torch.empty(5)),
    ]
    assert partition_buckets(params, 20) == [[2], [1, 0]]
    assert partition_buckets(params, 19) == [[2], [1], [0]]
    assert partition_buckets(params, 21) == [[2], [1, 0]]
    with pytest.raises(ValueError):
        partition_buckets(params, 0)


def test_t2_ddp_matches_single_process_and_is_distributed_bitwise() -> None:
    results = run_distributed(_ddp_equivalence_worker, 2)
    assert all(all(result["close"]) for result in results), results
    for result in results:
        assert result["cap_grads"][0] == result["cap_grads"][1] == result["cap_grads"][2]
    assert results[0]["cap_grads"] == results[1]["cap_grads"]


def test_t3_zero1_matches_adamw_and_ranks_are_bitwise(zero1_results: list[dict]) -> None:
    assert all(result["close"] for result in zero1_results), zero1_results
    assert zero1_results[0]["params"] == zero1_results[1]["params"]


def test_t4_shard_params_is_deterministic_and_balanced() -> None:
    params = [torch.nn.Parameter(torch.empty(size)) for size in (8, 7, 6, 5)]
    assert shard_params(params, 2) == [[0, 3], [1, 2]]
    shards = shard_params(params, 3)
    loads = [sum(params[i].numel() for i in shard) for shard in shards]
    assert max(loads) - min(loads) <= 8
    assert shard_params(params, 1) == [[0, 1, 2, 3]]


def test_t5_zero1_state_bytes_closed_form(zero1_results: list[dict]) -> None:
    assert all(r["state_bytes"] == r["expected_bytes"] for r in zero1_results)
    total = sum(r["state_bytes"] for r in zero1_results)
    total_numel = sum(parameter.numel() for parameter in _model().parameters())
    assert total == total_numel * 8 + len(list(_model().parameters())) * 4


def test_t6_zero2_keeps_only_owned_average_gradients() -> None:
    results = run_distributed(_zero2_worker, 2)
    for rank, result in enumerate(results):
        for index, after in enumerate(result["after"]):
            if index not in result["owned"]:
                assert after is None
                continue
            left = torch.tensor(results[0]["before"][index])
            right = torch.tensor(results[1]["before"][index])
            assert torch.equal(torch.tensor(after), (left + right) / 2)


def test_t7_world_size_one_degenerates_exactly() -> None:
    assert run_distributed(_world_one_worker, 1) == [
        {"grads_equal": True, "params_equal": True}
    ]


def test_t8_run_distributed_is_deterministic() -> None:
    assert run_distributed(_determinism_worker, 2) == run_distributed(
        _determinism_worker, 2
    )
