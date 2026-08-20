"""Phase 4.1 benchmark：acceptance rate 与加速比扫描。

用法（tiny 随机模型，纯看曲线形状）：
    python3 benchmarks/bench_spec.py
真模型对（租卡后，如 Qwen2.5-0.5B draft + 3B target）：
    python3 benchmarks/bench_spec.py --target <dir> --draft <dir>

产出 POSTMORTEM 素材：
  - temperature ∈ {0, 0.5, 1.0, 1.5} 的 acceptance rate 曲线
  - k ∈ {1, 2, 4, 8} 的实际加速比（vs target 单独自回归）
  - 找出「投机反而更慢」的配置并解释
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from minivllm.speculative import speculative_generate


def load_models(args):
    from transformers import AutoModelForCausalLM, Qwen2Config
    if args.target and args.draft:
        target = AutoModelForCausalLM.from_pretrained(args.target, torch_dtype=torch.float32)
        draft = AutoModelForCausalLM.from_pretrained(args.draft, torch_dtype=torch.float32)
    else:
        print("[i] 未指定权重，使用 tiny 随机模型（只看趋势，不代表真实加速比）")
        torch.manual_seed(0)
        target = AutoModelForCausalLM.from_config(Qwen2Config(
            vocab_size=512, hidden_size=256, num_hidden_layers=8,
            num_attention_heads=8, num_key_value_heads=4, intermediate_size=512))
        torch.manual_seed(1)
        draft = AutoModelForCausalLM.from_config(Qwen2Config(
            vocab_size=512, hidden_size=64, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, intermediate_size=128))
    target.eval(); draft.eval()
    return target, draft


@torch.no_grad()
def baseline_generate(target, prompt, n, temperature, generator):
    seq = prompt.clone()
    for _ in range(n):
        logits = target(seq).logits[:, -1, :]
        if temperature == 0:
            nxt = logits.argmax(-1, keepdim=True)
        else:
            probs = torch.softmax(logits / temperature, -1)
            nxt = torch.multinomial(probs, 1, generator=generator)
        seq = torch.cat([seq, nxt], dim=1)
    return seq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=None)
    ap.add_argument("--draft", default=None)
    ap.add_argument("--n-new", type=int, default=64)
    ap.add_argument("--prompt-len", type=int, default=16)
    args = ap.parse_args()

    target, draft = load_models(args)
    vocab = target.config.vocab_size
    torch.manual_seed(42)
    prompt = torch.randint(0, vocab, (1, args.prompt_len))

    print(f"\n{'temp':>6} {'k':>3} {'acc_rate':>9} {'spec(s)':>8} {'base(s)':>8} {'speedup':>8}")
    for temperature in [0.0, 0.5, 1.0, 1.5]:
        g = torch.Generator().manual_seed(0)
        t0 = time.perf_counter()
        baseline_generate(target, prompt, args.n_new, temperature, g)
        base_t = time.perf_counter() - t0

        for k in [1, 2, 4, 8]:
            g = torch.Generator().manual_seed(0)
            t0 = time.perf_counter()
            _, stats = speculative_generate(
                target, draft, prompt, max_new_tokens=args.n_new,
                k=k, temperature=temperature, generator=g)
            spec_t = time.perf_counter() - t0
            print(f"{temperature:>6.1f} {k:>3} {stats.acceptance_rate:>9.3f} "
                  f"{spec_t:>8.2f} {base_t:>8.2f} {base_t / spec_t:>7.2f}x")


if __name__ == "__main__":
    main()
