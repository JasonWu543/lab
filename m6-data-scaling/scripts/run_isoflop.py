"""run_isoflop.py — isoFLOP 实验脚手架（W12 租卡用）

用途：
  在 4 个 FLOPs 预算 × 4 个模型尺寸的网格上分别训练 HF tiny Qwen2 架构
  （TinyStories 数据集），产出 runs.json 喂给 minilaw.isoflop.isoflop_minima。

  本地 --dry-run 模式只打印网格与 GPU 时估算，不真正训练。

用法示例：
  # 本地 dry-run（不需要 GPU）
  python3 scripts/run_isoflop.py --dry-run

  # 完整训练（需要 GPU，W12 云端使用）
  python3 scripts/run_isoflop.py --output runs.json --data-dir /data/tinystories

  # 只跑特定 FLOPs 预算
  python3 scripts/run_isoflop.py --budgets 1e18 1e19 --dry-run
"""

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, asdict
from typing import Optional

# ── 实验网格参数 ────────────────────────────────────────────────────────────

# 4 个 FLOPs 预算（C = 6ND）
DEFAULT_FLOP_BUDGETS = [1e17, 1e18, 1e19, 1e20]

# 4 个模型尺寸（参数量 N，单位：百万参数 M）
DEFAULT_MODEL_SIZES_M = [10, 20, 40, 80]  # 10M ~ 80M

# Tiny Qwen2 架构超参（根据 N 参数量自动推算）
# 参考：Qwen2-0.5B 架构比例，缩放到 tiny 尺寸
ARCH_CONFIGS = {
    10:  dict(hidden_size=256,  num_heads=4,  num_layers=6,  intermediate_size=512),
    20:  dict(hidden_size=384,  num_heads=6,  num_layers=8,  intermediate_size=768),
    40:  dict(hidden_size=512,  num_heads=8,  num_layers=10, intermediate_size=1024),
    80:  dict(hidden_size=640,  num_heads=10, num_layers=12, intermediate_size=1280),
}

# GPU 训练速度估算（A100 80G，MFU ≈ 0.4）
A100_FLOPS_PER_SEC = 312e12  # 312 TFLOPs BF16
MFU = 0.40


# ── 数据结构 ─────────────────────────────────────────────────────────────────

@dataclass
class RunConfig:
    """单次训练配置。"""
    C: float           # 目标 FLOPs 预算
    N: int             # 模型参数量（实际）
    D: int             # 训练 token 数
    model_size_M: int  # 名义尺寸（M）
    hidden_size: int
    num_heads: int
    num_layers: int
    intermediate_size: int
    batch_size: int
    seq_len: int
    max_steps: int
    estimated_gpu_hours: float


def estimate_actual_params(arch: dict) -> int:
    """粗估模型参数量（不含 embedding）。"""
    h = arch["hidden_size"]
    ffn = arch["intermediate_size"]
    L = arch["num_layers"]
    # 每层 = 4*h^2（attention QKV+O）+ 2*h*ffn（FFN up/down） + 2*h（norms）
    per_layer = 4 * h * h + 2 * h * ffn + 2 * h
    return L * per_layer


def build_run_configs(
    flop_budgets: list[float],
    model_sizes_M: list[int],
    seq_len: int = 2048,
    vocab_size: int = 32000,
) -> list[RunConfig]:
    """从 FLOPs 预算 × 模型尺寸网格生成训练配置列表。"""
    configs = []
    for C in flop_budgets:
        for size_M in model_sizes_M:
            arch = ARCH_CONFIGS[size_M]
            N_est = estimate_actual_params(arch)
            # D = C / (6N)，取整到 batch boundary
            D = int(C / (6 * N_est))
            if D < seq_len:
                # token 数太少，跳过
                continue
            batch_size = 16
            max_steps = max(1, D // (batch_size * seq_len))
            # GPU 时估算
            gpu_hours = C / (A100_FLOPS_PER_SEC * MFU * 3600)
            configs.append(RunConfig(
                C=C,
                N=N_est,
                D=D,
                model_size_M=size_M,
                hidden_size=arch["hidden_size"],
                num_heads=arch["num_heads"],
                num_layers=arch["num_layers"],
                intermediate_size=arch["intermediate_size"],
                batch_size=batch_size,
                seq_len=seq_len,
                max_steps=max_steps,
                estimated_gpu_hours=gpu_hours,
            ))
    return configs


def print_grid(configs: list[RunConfig]):
    """打印实验网格（dry-run 时使用）。"""
    print(f"\n{'=' * 72}")
    print(f"{'isoFLOP 实验网格':^72}")
    print(f"{'=' * 72}")
    print(f"{'C (FLOPs)':>14} {'N (params)':>12} {'D (tokens)':>12} "
          f"{'steps':>8} {'est. GPU-h':>10}")
    print(f"{'-' * 72}")
    total_gpu_h = 0.0
    for cfg in configs:
        print(f"{cfg.C:>14.2e} {cfg.N:>12,} {cfg.D:>12,} "
              f"{cfg.max_steps:>8,} {cfg.estimated_gpu_hours:>10.2f}")
        total_gpu_h += cfg.estimated_gpu_hours
    print(f"{'-' * 72}")
    print(f"{'合计':>50} {total_gpu_h:>10.2f} GPU 小时")
    print(f"{'=' * 72}\n")
    print(f"共 {len(configs)} 个训练 run，覆盖 "
          f"{len(set(c.C for c in configs))} 个 FLOPs 预算 × "
          f"{len(set(c.model_size_M for c in configs))} 个模型尺寸。")
    print("\n注：实际 GPU 用时受通信开销、checkpoint IO 等影响，估算仅供参考。")


def train_single_run(cfg: RunConfig, data_dir: str, output_dir: str) -> dict:
    """训练单个 run 并返回结果字典（含最终 loss）。

    真实训练需要 transformers + torch，本函数是脚手架，
    W12 租卡时在此填入实际训练代码。
    """
    try:
        import torch
        from transformers import Qwen2Config, Qwen2ForCausalLM
        from torch.utils.data import DataLoader
    except ImportError:
        raise RuntimeError(
            "训练模式需要 torch 和 transformers。"
            "本地请用 --dry-run 模式。"
        )

    print(f"\n[RUN] C={cfg.C:.2e}, N={cfg.N:,}, D={cfg.D:,}, "
          f"size={cfg.model_size_M}M, steps={cfg.max_steps}")

    # --- 构建 tiny Qwen2 模型 ---
    model_cfg = Qwen2Config(
        hidden_size=cfg.hidden_size,
        num_attention_heads=cfg.num_heads,
        num_hidden_layers=cfg.num_layers,
        intermediate_size=cfg.intermediate_size,
        vocab_size=32000,
        max_position_embeddings=cfg.seq_len,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Qwen2ForCausalLM(model_cfg).to(device)

    # --- 实际参数量 ---
    actual_N = sum(p.numel() for p in model.parameters())

    # --- 数据加载（TinyStories，预 tokenize 为 .bin mmap）---
    # TODO W12: 从 data_dir 加载 TinyStories token 文件
    # 示例：data = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
    raise NotImplementedError(
        "TODO W12: 填入 TinyStories 数据加载 + 训练循环。\n"
        "  参考 nanoGPT 的 train.py 结构：\n"
        "    1. 加载 mmap 数据\n"
        "    2. AdamW (lr=3e-4, cosine decay)\n"
        "    3. 跑 cfg.max_steps 步，记录 final loss\n"
        "    4. 返回 {'C': cfg.C, 'N': actual_N, 'L': final_loss}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="isoFLOP 实验脚手架（HF tiny Qwen2 + TinyStories）"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="只打印网格与 GPU 时估算，不真正训练"
    )
    parser.add_argument(
        "--output", default="runs.json",
        help="输出 runs.json 路径（默认：runs.json）"
    )
    parser.add_argument(
        "--data-dir", default="data/tinystories",
        help="TinyStories 预处理数据目录（含 train.bin）"
    )
    parser.add_argument(
        "--budgets", nargs="+", type=float,
        default=DEFAULT_FLOP_BUDGETS,
        help="FLOPs 预算列表（默认：1e17 1e18 1e19 1e20）"
    )
    parser.add_argument(
        "--sizes", nargs="+", type=int,
        default=DEFAULT_MODEL_SIZES_M,
        help="模型尺寸列表（单位 M，默认：10 20 40 80）"
    )
    parser.add_argument(
        "--seq-len", type=int, default=2048,
        help="序列长度（默认：2048）"
    )
    args = parser.parse_args()

    configs = build_run_configs(args.budgets, args.sizes, seq_len=args.seq_len)

    if args.dry_run:
        print_grid(configs)
        print("\n[dry-run] 不执行实际训练。去掉 --dry-run 并确保 GPU 可用后正式运行。")
        return

    # 正式训练模式
    if not os.path.isdir(args.data_dir):
        print(f"错误：数据目录 {args.data_dir} 不存在。", file=sys.stderr)
        sys.exit(1)

    results = []
    output_dir = os.path.dirname(os.path.abspath(args.output)) or "."

    print_grid(configs)
    print(f"\n开始训练 {len(configs)} 个 run，结果写入 {args.output} ...\n")

    for i, cfg in enumerate(configs):
        print(f"[{i+1}/{len(configs)}] ", end="", flush=True)
        try:
            run_result = train_single_run(cfg, args.data_dir, output_dir)
            results.append(run_result)
        except NotImplementedError as e:
            print(f"\n\n[WARN] train_single_run 尚未实现：\n{e}\n", file=sys.stderr)
            break
        except Exception as e:
            print(f"\n[ERROR] run 失败：{e}", file=sys.stderr)
            continue

        # 中途保存（防崩）
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  -> 已保存 {len(results)} 条结果到 {args.output}")

    print(f"\n完成。共 {len(results)} 条 run 结果写入 {args.output}")
    print("下一步：python3 -c \"from minilaw.isoflop import isoflop_minima; "
          "import json; print(isoflop_minima(json.load(open('runs.json'))))\"")


if __name__ == "__main__":
    main()
