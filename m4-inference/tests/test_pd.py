"""SPEC 4.2 验收测试：所有期望值均由闭式或逐 tick 手算独立得到。"""

from dataclasses import asdict

import pytest

from minipd.request import PDRequest
from minipd.simulate import make_workload, run_colocated_baseline, run_sim
from minipd.transfer import TransferModel, transfer_ticks
from minipd.workers import DecodeWorker, PrefillWorker


def test_t1_transfer_ticks_closed_form() -> None:
    """T1：整除、非整除及零 payload 的闭式结果。"""
    model = TransferModel(base_ticks=2, bytes_per_token=3, bandwidth_bytes_per_tick=6)
    assert transfer_ticks(model, 0) == 2
    assert transfer_ticks(model, 4) == 4
    assert transfer_ticks(model, 5) == 5


def test_t2_chunked_prefill_and_fifo_allocation() -> None:
    """T2：单请求三 tick；双请求在一个 tick 内按 FIFO 跨请求分配。"""
    worker = PrefillWorker(prefill_tokens_per_tick=99, chunk_size=4)
    req = PDRequest(0, 0, 10, 1)
    worker.add(req)
    assert worker.step(0) == []
    assert worker.step(1) == []
    assert worker.step(2) == [req]

    worker = PrefillWorker(prefill_tokens_per_tick=6, chunk_size=4)
    first, second = PDRequest(1, 0, 6, 1), PDRequest(2, 0, 4, 1)
    worker.add(first)
    worker.add(second)
    assert worker.step(0) == []
    assert (first.prefilled_tokens, second.prefilled_tokens) == (4, 2)
    assert worker.step(1) == [first, second]


def test_t3_single_request_exact_timeline() -> None:
    """T3：逐 tick 时间线精确对拍。

    t=0 admission；t=1/2 prefill 4/2；t=3 transfer 2→1；
    t=4 transfer 1→0 后 decode 首 token；t=5、6 再产 token 并完成。
    因此 TTFT=4，finish=6，TPOT=(6-4)/2=1。
    """
    result = run_sim(
        [PDRequest(7, 0, 6, 3)],
        prefill_tokens_per_tick=4,
        chunk_size=4,
        decode_slots=1,
        transfer=TransferModel(1, 1, 6),
    )
    assert result.ttft == {7: 4}
    assert result.finish == {7: 6}
    assert result.tpot == {7: 1.0}
    assert result.total_ticks == 7
    # 整数 tick 的定义下结果精确，不需要近似容差。
    assert result.throughput == 3 / 7


def test_t4_decode_slot_waiter_starts_next_tick() -> None:
    """T4：slot=1 的补位者严格在下一 tick 才产首 token。"""
    worker = DecodeWorker(1)
    first, second = PDRequest(0, 0, 1, 2), PDRequest(1, 0, 1, 1)
    worker.add(first)
    worker.add(second)
    assert worker.step(10) == []
    assert worker.step(11) == [first]
    assert second.first_token_time is None
    assert worker.step(12) == [second]
    assert second.first_token_time == 12 > first.finish_time


def test_t5_seed_and_simulation_are_deterministic() -> None:
    """T5：workload 与模拟的所有结果字段可复现。"""
    left = make_workload(12, 2026, 0.8, (2, 20), (1, 6))
    right = make_workload(12, 2026, 0.8, (2, 20), (1, 6))
    assert [asdict(r) for r in left] == [asdict(r) for r in right]
    model = TransferModel(1, 4, 20)
    assert asdict(run_sim(left, 8, 4, 2, model)) == asdict(run_sim(right, 8, 4, 2, model))


def test_t6_small_chunks_reduce_short_request_ttft() -> None:
    """T6：小 chunk 让排在长 prompt 后的短请求更早获得 prefill 预算。"""
    workload = [PDRequest(0, 0, 30, 2), PDRequest(1, 1, 4, 2)]
    model = TransferModel(0, 0, 1)
    small = run_sim(workload, 8, 4, 2, model)
    large = run_sim(workload, 8, 8, 2, model)
    assert small.ttft[1] < large.ttft[1]


def test_t7_pd_separation_reduces_worst_tpot() -> None:
    """T7：持续长 prefill 抢光共享预算时，PD 的 decode 不被阻塞。"""
    workload = [PDRequest(0, 0, 1, 8)] + [PDRequest(i, i, 12, 2) for i in range(1, 7)]
    pd = run_sim(workload, 8, 4, 2, TransferModel(0, 0, 1))
    colocated = run_colocated_baseline(workload, tokens_per_tick=4, decode_slots=2)
    assert max(pd.tpot.values()) < max(colocated.tpot.values())


def test_t8_slow_transfer_reverses_ttft_advantage() -> None:
    """T8：极低带宽的迁移成本使 PD 的 TTFT 严格差于无迁移 baseline。"""
    workload = [PDRequest(0, 0, 8, 2)]
    pd = run_sim(workload, 8, 8, 1, TransferModel(2, 100, 1))
    colocated = run_colocated_baseline(workload, tokens_per_tick=8, decode_slots=1)
    assert pd.ttft[0] > colocated.ttft[0]


def test_t9_boundaries_and_empty_workload() -> None:
    """T9：单 prompt、单输出 token 与空 workload。"""
    model = TransferModel(0, 0, 1)
    one = run_sim([PDRequest(0, 0, 1, 1)], 1, 1, 1, model)
    assert one.ttft == {0: 2}
    assert one.finish == {0: 2}
    assert one.tpot == {0: 0.0}
    empty = run_sim([], 1, 1, 1, model)
    assert asdict(empty) == {"ttft": {}, "tpot": {}, "finish": {}, "total_ticks": 0, "throughput": 0.0}
    assert asdict(run_colocated_baseline([], 1, 1)) == asdict(empty)
