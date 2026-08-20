"""bench_engine.py — mini-vLLM 推理引擎 benchmark

用法：
  python3 benchmarks/bench_engine.py
  python3 benchmarks/bench_engine.py --model-path ../m1-foundation/data/qwen2.5-0.5b
  python3 benchmarks/bench_engine.py --model-path /path/to/Qwen2.5-0.5B --num-requests 32

报告指标：
  - Throughput (tokens/s)：output tokens / 总时间
  - TTFT p50 / p95（ms）：首 token 延迟（step 数 × 每步平均时间）
  - TPOT p50 / p95（ms）：per-output-token 延迟
  - 并发退化曲线（不同 max_batch_tokens 下的 throughput）

权重路径不存在时报错退出并提示。
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch


def parse_args():
    p = argparse.ArgumentParser(description="mini-vLLM engine benchmark")
    p.add_argument(
        "--model-path",
        default=str(Path(__file__).parent.parent.parent / "m1-foundation" / "data" / "qwen2.5-0.5b"),
        help="HuggingFace 模型目录（默认：../m1-foundation/data/qwen2.5-0.5b）",
    )
    p.add_argument("--num-requests", type=int, default=20, help="总请求数")
    p.add_argument("--prompt-len", type=int, default=64, help="平均 prompt 长度")
    p.add_argument("--output-len", type=int, default=32, help="每个请求的 max_new_tokens")
    p.add_argument("--poisson-rate", type=float, default=2.0, help="泊松到达率（请求/step）")
    p.add_argument("--num-blocks", type=int, default=512)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--concurrency-sweep",
        action="store_true",
        help="是否扫描不同 max_batch_tokens 画退化曲线",
    )
    return p.parse_args()


def load_model(model_path: str):
    """加载真权重模型，路径不存在时报错退出。"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = Path(model_path)
    if not path.exists():
        print(f"[ERROR] 模型路径不存在：{path}", file=sys.stderr)
        print(
            "  请先下载 Qwen2.5-0.5B 权重：\n"
            "  huggingface-cli download Qwen/Qwen2.5-0.5B --local-dir "
            f"{path}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Loading model from {path} ...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(str(path), torch_dtype=torch.float32)
    model.eval()
    print(f"Model loaded in {time.time()-t0:.1f}s")
    return model


def make_requests(model, num_requests, prompt_len, output_len, poisson_rate, seed):
    """生成泊松到达的随机请求（用 vocab_size 随机 token 模拟 prompt）。"""
    import random as rnd
    rnd.seed(seed)
    np.random.seed(seed)

    vocab_size = model.config.vocab_size
    requests = []
    arrival_step = 0

    for i in range(num_requests):
        # 随机 prompt 长度（±50%）
        plen = max(4, int(rnd.gauss(prompt_len, prompt_len * 0.3)))
        prompt_ids = [rnd.randint(1, vocab_size - 1) for _ in range(plen)]

        from minivllm.request import Request
        req = Request(
            req_id=i,
            prompt_ids=prompt_ids,
            max_new_tokens=output_len,
            arrival_step=arrival_step,
        )
        requests.append(req)

        # 泊松间隔
        gap = np.random.poisson(1.0 / poisson_rate)
        arrival_step += max(1, gap)

    return requests


def run_benchmark(model, requests, num_blocks, block_size, max_batch_tokens):
    """运行引擎，返回 (elapsed_s, metrics_dict)。"""
    from minivllm.engine import Engine

    engine = Engine(
        model,
        num_blocks=num_blocks,
        block_size=block_size,
        max_batch_tokens=max_batch_tokens,
    )

    # 按 arrival_step 排序，模拟真实到达
    sorted_reqs = sorted(requests, key=lambda r: r.arrival_step)

    req_iter = iter(sorted_reqs)
    next_req = next(req_iter, None)

    t_start = time.time()

    while engine.scheduler.num_waiting > 0 or engine.scheduler.num_running > 0 or next_req is not None:
        # 本 step 前加入该到的请求
        while next_req is not None and next_req.arrival_step <= engine._step_count:
            engine.add_request(next_req)
            next_req = next(req_iter, None)

        if engine.scheduler.num_waiting == 0 and engine.scheduler.num_running == 0:
            # 等待下一个请求到达（推进 step 计数）
            engine._step_count += 1
            continue

        engine.step()

    elapsed = time.time() - t_start
    return elapsed, engine.metrics()


def percentile(values, p):
    if not values:
        return float("nan")
    arr = sorted(values)
    idx = int(len(arr) * p / 100)
    return arr[min(idx, len(arr) - 1)]


def print_report(elapsed, metrics, step_time_ms, label=""):
    if label:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")

    n_out_total = sum(m["n_out"] for m in metrics.values())
    throughput = n_out_total / elapsed if elapsed > 0 else 0

    ttft_ms = [m["ttft_steps"] * step_time_ms for m in metrics.values()]
    tpot_ms = [m["tpot_steps"] * step_time_ms for m in metrics.values() if m["n_out"] > 1]

    print(f"  Elapsed          : {elapsed:.2f}s")
    print(f"  Total output toks: {n_out_total}")
    print(f"  Throughput       : {throughput:.1f} tokens/s")
    print(f"  TTFT p50 / p95   : {percentile(ttft_ms,50):.1f} / {percentile(ttft_ms,95):.1f} ms")
    print(f"  TPOT p50 / p95   : {percentile(tpot_ms,50):.1f} / {percentile(tpot_ms,95):.1f} ms")


def main():
    args = parse_args()
    model = load_model(args.model_path)

    print(f"\nGenerating {args.num_requests} requests (prompt~{args.prompt_len}, out={args.output_len})...")
    requests = make_requests(
        model, args.num_requests, args.prompt_len, args.output_len,
        args.poisson_rate, args.seed
    )

    # 估算每步时间（热身 5 步）
    print("Warmup...")
    from minivllm.request import Request
    from minivllm.engine import Engine
    warmup_req = Request(req_id=-1, prompt_ids=list(range(1, 17)), max_new_tokens=5)
    warmup_engine = Engine(model, num_blocks=64, block_size=16, max_batch_tokens=512)
    warmup_engine.add_request(warmup_req)
    t0 = time.time()
    warmup_engine.run()
    warmup_elapsed = time.time() - t0
    step_time_ms = warmup_elapsed / 6 * 1000  # ~6 steps for warmup
    print(f"  Est. step time: {step_time_ms:.1f} ms")

    if args.concurrency_sweep:
        print("\n=== Concurrency sweep ===")
        batch_sizes = [256, 512, 1024, 2048, 4096]
        for mbt in batch_sizes:
            elapsed, metrics = run_benchmark(
                model, requests, args.num_blocks, args.block_size, mbt
            )
            n_out = sum(m["n_out"] for m in metrics.values())
            tps = n_out / elapsed if elapsed > 0 else 0
            print(f"  max_batch_tokens={mbt:5d}: {tps:7.1f} tokens/s  elapsed={elapsed:.1f}s")
    else:
        elapsed, metrics = run_benchmark(
            model, requests, args.num_blocks, args.block_size,
            max_batch_tokens=2048,
        )
        print_report(elapsed, metrics, step_time_ms, label="Benchmark Results")

    print("\nDone. Run with --concurrency-sweep to see degradation curve.")
    print("Compare with vLLM: serve the same model and run the same workload.")


if __name__ == "__main__":
    main()
