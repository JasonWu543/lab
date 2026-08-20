"""Process-management scaffolding supplied to students."""

from __future__ import annotations

import os
import queue as queue_module
import tempfile
import time
from datetime import timedelta
from typing import Any, Callable

import torch.distributed as dist
import torch.multiprocessing as mp


def _distributed_entry(
    rank: int,
    fn: Callable[..., Any],
    world_size: int,
    rendezvous: str,
    queue: Any,
    args: tuple[Any, ...],
) -> None:
    """Initialize one rank, report its result, and always tear down gloo."""
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60),
    )
    try:
        queue.put((rank, fn(rank, world_size, *args)))
    finally:
        dist.destroy_process_group()


def run_distributed(fn: Callable[..., Any], world_size: int, *args: Any) -> list[Any]:
    """Run ``fn(rank, world_size, *args)`` under mp.spawn and collect results."""
    if world_size < 1 or world_size > 4:
        raise ValueError("world_size must be in [1, 4]")
    context = mp.get_context("spawn")
    # 必须在等待 worker 退出的同时排空结果管道；若先 join 后 get，大返回值可能
    # 填满 pipe，使 worker 卡在 Queue feeder flush，最终被误判成计算超时。
    result_queue = context.Queue()
    with tempfile.TemporaryDirectory(prefix="minidist-") as tmpdir:
        rendezvous = os.path.join(tmpdir, "rdv")
        processes = mp.spawn(
            _distributed_entry,
            args=(fn, world_size, rendezvous, result_queue, args),
            nprocs=world_size,
            join=False,
        )
        deadline = time.monotonic() + 60.0
        ranked = []
        while True:
            while len(ranked) < world_size:
                try:
                    ranked.append(result_queue.get_nowait())
                except queue_module.Empty:
                    break
            if processes.join(
                timeout=max(0.0, min(0.2, deadline - time.monotonic()))
            ):
                break
            if time.monotonic() < deadline:
                continue
            for process in processes.processes:
                if process.is_alive():
                    process.terminate()
            for process in processes.processes:
                process.join(timeout=2.0)
                if process.is_alive():
                    process.kill()
                    process.join()
            raise TimeoutError("distributed worker exceeded the 60s test timeout")
        while len(ranked) < world_size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("distributed result collection exceeded 60s")
            try:
                ranked.append(result_queue.get(timeout=min(0.2, remaining)))
            except queue_module.Empty:
                continue
    result_queue.close()
    result_queue.join_thread()
    ranked.sort(key=lambda pair: pair[0])
    return [value for _, value in ranked]
