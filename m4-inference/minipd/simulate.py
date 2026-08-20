"""PD 与合并部署模拟入口。"""

from collections import deque
from dataclasses import dataclass

import numpy as np

from .request import PDRequest
from .transfer import TransferModel


@dataclass
class SimResult:
    """一次模拟的逐请求延迟与整体吞吐结果。"""

    ttft: dict[int, int]
    tpot: dict[int, float]
    finish: dict[int, int]
    total_ticks: int
    throughput: float


def run_sim(
    requests: list[PDRequest],
    prefill_tokens_per_tick: int,
    chunk_size: int,
    decode_slots: int,
    transfer: TransferModel,
) -> SimResult:
    """运行 PD 分离模拟直到所有请求完成。

    请复制请求，避免一次模拟污染调用方或下一次确定性对拍；逐 tick 提交
    arrival_time 已到的请求，并从运行时字段汇总冻结指标定义。
    """
    raise NotImplementedError("请实现 PD 离散时间模拟闭环")


def _result(requests: list[PDRequest], total_ticks: int) -> SimResult:
    """从已完成请求构造结果（给定脚手架内部辅助函数）。"""
    ttft = {
        req.req_id: int(req.first_token_time - req.arrival_time)  # type: ignore[operator]
        for req in requests
    }
    tpot = {
        req.req_id: (req.finish_time - req.first_token_time)  # type: ignore[operator]
        / max(1, req.max_new_tokens - 1)
        for req in requests
    }
    finish = {req.req_id: int(req.finish_time) for req in requests}  # type: ignore[arg-type]
    produced = sum(req.max_new_tokens for req in requests)
    return SimResult(ttft, tpot, finish, total_ticks, produced / total_ticks if total_ticks else 0.0)


def run_colocated_baseline(
    requests: list[PDRequest], tokens_per_tick: int, decode_slots: int
) -> SimResult:
    """模拟 prefill 优先、共享 token 预算的合并部署（给定脚手架）。"""
    if tokens_per_tick <= 0 or decode_slots <= 0:
        raise ValueError("tokens_per_tick and decode_slots must be positive")
    work = [PDRequest(r.req_id, r.arrival_time, r.prompt_len, r.max_new_tokens) for r in requests]
    if not work:
        return _result([], 0)
    future = deque(sorted(work, key=lambda r: (r.arrival_time, r.req_id)))
    prefill: deque[PDRequest] = deque()
    active: list[PDRequest] = []
    waiting: deque[PDRequest] = deque()
    done = 0
    t = 0
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
        # 只有 prefill 剩下的共享预算才能用于 decode，体现干扰。
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


def make_workload(
    n: int,
    seed: int,
    arrival_rate: float,
    prompt_len_range: tuple[int, int],
    gen_len_range: tuple[int, int],
) -> list[PDRequest]:
    """用固定 NumPy RNG 生成泊松间隔与闭区间长度（给定脚手架）。"""
    if n < 0 or arrival_rate <= 0:
        raise ValueError("n must be non-negative and arrival_rate must be positive")
    if prompt_len_range[0] > prompt_len_range[1] or gen_len_range[0] > gen_len_range[1]:
        raise ValueError("invalid length range")
    rng = np.random.default_rng(seed)
    gaps = rng.exponential(1.0 / arrival_rate, size=n)
    arrivals = np.rint(np.cumsum(gaps)).astype(int)
    prompts = rng.integers(prompt_len_range[0], prompt_len_range[1] + 1, size=n)
    gens = rng.integers(gen_len_range[0], gen_len_range[1] + 1, size=n)
    return [PDRequest(i, int(arrivals[i]), int(prompts[i]), int(gens[i])) for i in range(n)]
