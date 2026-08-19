"""
Phase 1.2 Benchmark：理论 FLOPs vs profiler 实测

用法：
    python3 benchmarks/bench_model.py
"""
import os
import sys
import time
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from minilm.model.config import ModelConfig
from minilm.model.model import MiniLM
from minilm.model.counting import count_params, estimate_flops_per_token


def bench(cfg: ModelConfig, seq_len: int = 64, n_warmup: int = 3, n_runs: int = 10):
    model = MiniLM(cfg)
    model.eval()
    n_params = count_params(cfg)
    theo_flops = estimate_flops_per_token(cfg, seq_len)

    input_ids = torch.randint(0, cfg.vocab_size, (1, seq_len))

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(input_ids)

    # Time
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            logits = model(input_ids)
            t1 = time.perf_counter()
            times.append(t1 - t0)

    avg_ms = sum(times) / len(times) * 1000
    tokens_per_s = seq_len / (avg_ms / 1000)

    print(f"\n{'='*60}")
    print(f"Config: hidden={cfg.hidden_size}, layers={cfg.num_layers}, heads={cfg.num_heads}")
    print(f"Params:           {n_params:>12,}")
    print(f"Seq len:          {seq_len:>12}")
    print(f"Avg latency:      {avg_ms:>11.2f} ms")
    print(f"Throughput:       {tokens_per_s:>11.1f} tok/s")
    print(f"Theo FLOPs/tok:   {theo_flops:>12,}")

    # Try torch profiler
    try:
        from torch.profiler import profile, record_function, ProfilerActivity
        with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
            with record_function("model_inference"):
                with torch.no_grad():
                    model(input_ids)
        print("\nProfiler top ops:")
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=5))
    except Exception as e:
        print(f"Profiler not available: {e}")


if __name__ == '__main__':
    # Small model
    cfg_small = ModelConfig(
        vocab_size=8192,
        hidden_size=256,
        intermediate_size=1024,
        num_layers=4,
        num_heads=8,
        num_kv_heads=4,
        head_dim=32,
        max_seq_len=512,
    )
    bench(cfg_small, seq_len=64)

    # Medium model
    cfg_med = ModelConfig(
        vocab_size=32000,
        hidden_size=512,
        intermediate_size=2048,
        num_layers=8,
        num_heads=8,
        num_kv_heads=4,
        head_dim=64,
        max_seq_len=512,
    )
    bench(cfg_med, seq_len=64)
