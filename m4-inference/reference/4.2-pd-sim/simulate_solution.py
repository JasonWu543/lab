"""Phase 4.2 simulate 参考答案；给定脚手架与学生入口一并自包含。"""

from collections import deque
from dataclasses import dataclass

import numpy as np

from .request import PDRequest
from .scheduler import PDScheduler
from .transfer import TransferModel
from .workers import DecodeWorker, PrefillWorker


@dataclass
class SimResult:
    ttft: dict[int, int]
    tpot: dict[int, float]
    finish: dict[int, int]
    total_ticks: int
    throughput: float


def _result(requests: list[PDRequest], total_ticks: int) -> SimResult:
    ttft = {r.req_id: int(r.first_token_time - r.arrival_time) for r in requests}
    tpot = {r.req_id: (r.finish_time - r.first_token_time) / max(1, r.max_new_tokens - 1) for r in requests}
    finish = {r.req_id: int(r.finish_time) for r in requests}
    produced = sum(r.max_new_tokens for r in requests)
    return SimResult(ttft, tpot, finish, total_ticks, produced / total_ticks if total_ticks else 0.0)


def run_sim(requests, prefill_tokens_per_tick, chunk_size, decode_slots, transfer):
    """复制输入并驱动 tick；最后一个完成 tick 为 t，故总跨度按 t+1 计。"""
    work = [PDRequest(r.req_id, r.arrival_time, r.prompt_len, r.max_new_tokens) for r in requests]
    if not work:
        return _result([], 0)
    scheduler = PDScheduler(PrefillWorker(prefill_tokens_per_tick, chunk_size), DecodeWorker(decode_slots), transfer)
    for req in work:
        scheduler.submit(req)
    t = 0
    while not scheduler.all_done():
        scheduler.tick(t)
        t += 1
    return _result(work, t)


def run_colocated_baseline(requests, tokens_per_tick, decode_slots):
    if tokens_per_tick <= 0 or decode_slots <= 0:
        raise ValueError("capacities must be positive")
    work = [PDRequest(r.req_id, r.arrival_time, r.prompt_len, r.max_new_tokens) for r in requests]
    if not work:
        return _result([], 0)
    future = deque(sorted(work, key=lambda r: (r.arrival_time, r.req_id)))
    prefill, waiting, active = deque(), deque(), []
    done = t = 0
    while done < len(work):
        while future and future[0].arrival_time <= t:
            prefill.append(future.popleft())
        budget = tokens_per_tick
        while budget and prefill:
            req = prefill[0]
            used = min(budget, req.prompt_len - req.prefilled_tokens)
            req.prefilled_tokens += used
            budget -= used
            if req.prefilled_tokens == req.prompt_len:
                prefill.popleft()
                (active if len(active) < decode_slots else waiting).append(req)
            else:
                break
        for req in list(active)[:budget]:
            req.generated_tokens += 1
            if req.first_token_time is None:
                req.first_token_time = t
            if req.generated_tokens == req.max_new_tokens:
                req.finish_time = t
                active.remove(req)
                done += 1
        while len(active) < decode_slots and waiting:
            active.append(waiting.popleft())
        t += 1
    return _result(work, t)


def make_workload(n, seed, arrival_rate, prompt_len_range, gen_len_range):
    if n < 0 or arrival_rate <= 0:
        raise ValueError("invalid workload size/rate")
    rng = np.random.default_rng(seed)
    arrivals = np.rint(np.cumsum(rng.exponential(1.0 / arrival_rate, size=n))).astype(int)
    prompts = rng.integers(prompt_len_range[0], prompt_len_range[1] + 1, size=n)
    gens = rng.integers(gen_len_range[0], gen_len_range[1] + 1, size=n)
    return [PDRequest(i, int(arrivals[i]), int(prompts[i]), int(gens[i])) for i in range(n)]
