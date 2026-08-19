"""消融实验脚手架（W8-9 租卡跑，不进 correctness 测试）。

四组实验，回答 SPEC §1 的四个问题：
  moe_vs_dense : 等 activated FLOPs 下 MoE vs Dense 的 val loss
  gamma_sweep  : bias_update_speed ∈ {0, 1e-4, 1e-3, 1e-2} 对 expert 负载均衡的影响
  mla_vs_mha   : MLA vs 标准 MHA 的 val loss 与 KV cache 显存
  mtp          : 加/不加 MTP 的 loss 对比

公平性：固定 seed=42、等 token 预算；结果写 results/{exp}.json。

用法：
  python3 scripts/ablate.py --exp moe_vs_dense --token-budget 50000

注意：这是脚手架。训练循环 / 数据加载留了 TODO，学生在跑消融前补齐
（依赖 minimoe/ 全绿 + 一份 tokenized TinyStories）。
"""
from __future__ import annotations

import argparse
import json
import os
import time

SEED = 42


def set_seed(seed: int = SEED):
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_data(token_budget: int):
    """加载统一的 tokenized TinyStories，切成 train/val。

    返回 (train_ids, val_ids)（LongTensor）。TODO：接上真实数据管线；
    当前给一个可跑通脚手架的随机占位（仅用于烟雾测试，不产出可信数字）。
    """
    import torch

    from minimoe.config import MoEConfig

    cfg = MoEConfig()
    # TODO(学生/实验者): 替换为真实 TinyStories tokenized 数据
    n = token_budget
    ids = torch.randint(0, cfg.vocab_size, (n,))
    split = int(n * 0.9)
    return ids[:split], ids[split:]


def batchify(ids, batch_size, seq_len):
    import torch

    n = (ids.numel() - 1) // (batch_size * seq_len)
    ids = ids[: n * batch_size * seq_len + 1]
    x = ids[:-1].view(batch_size, -1)
    y = ids[1:].view(batch_size, -1)
    # 按 seq_len 切块
    for i in range(0, x.shape[1] - seq_len + 1, seq_len):
        yield x[:, i : i + seq_len], y[:, i : i + seq_len]


def train_eval(model, train_ids, val_ids, *, steps, batch_size, seq_len, lr=3e-4, use_mtp=False):
    """通用训练 + 评估循环。返回 dict（含 train/val loss、wall_clock）。

    TODO(实验者): 这是最小可跑框架，按需补 warmup / grad clip / 日志。
    """
    import torch
    import torch.nn.functional as F

    cfg = model.cfg
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    t0 = time.time()
    step = 0
    train_batches = list(batchify(train_ids, batch_size, seq_len))
    while step < steps:
        for x, y in train_batches:
            if step >= steps:
                break
            opt.zero_grad()
            out = model(x)
            if use_mtp:
                logits, mtp_logits = out
                l_main = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), y.reshape(-1))
                # MTP 预测 t+2：与 y 的下一位对齐
                l_mtp = F.cross_entropy(
                    mtp_logits[:, :-1].reshape(-1, cfg.vocab_size), y[:, 1:].reshape(-1)
                )
                loss = l_main + l_mtp
            else:
                logits = out
                loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), y.reshape(-1))
            loss.backward()
            opt.step()
            step += 1
    wall = time.time() - t0

    # 评估
    model.eval()
    with torch.no_grad():
        vloss, nb = 0.0, 0
        for x, y in batchify(val_ids, batch_size, seq_len):
            out = model(x)
            logits = out[0] if isinstance(out, tuple) else out
            vloss += F.cross_entropy(logits.reshape(-1, cfg.vocab_size), y.reshape(-1)).item()
            nb += 1
        val_loss = vloss / max(nb, 1)
    return {"val_loss": val_loss, "wall_clock_s": wall, "steps": step}


# ---------------- 四个实验 ----------------
def exp_moe_vs_dense(args):
    from minimoe.config import MoEConfig
    from minimoe.model import MiniMoELM
    from minimoe.parity import activated_ffn_params, dense_config_matching_flops

    cfg = MoEConfig()
    dense_cfg = dense_config_matching_flops(cfg)
    train_ids, val_ids = load_data(args.token_budget)

    results = {}
    for name, (c, use_moe) in {
        "moe": (cfg, True),
        "dense": (dense_cfg, False),
    }.items():
        set_seed()
        model = MiniMoELM(c, use_moe=use_moe)
        r = train_eval(
            model, train_ids, val_ids,
            steps=args.steps, batch_size=args.batch_size, seq_len=args.seq_len,
        )
        r["activated_ffn_params"] = activated_ffn_params(c, moe=use_moe)
        results[name] = r
    results["fairness"] = "equal activated FFN params (see parity.py)"
    return results


def exp_gamma_sweep(args):
    import torch

    from dataclasses import replace

    from minimoe.config import MoEConfig
    from minimoe.model import MiniMoELM

    train_ids, val_ids = load_data(args.token_budget)
    results = {}
    for gamma in [0.0, 1e-4, 1e-3, 1e-2]:
        set_seed()
        cfg = replace(MoEConfig(), bias_update_speed=gamma)
        model = MiniMoELM(cfg, use_moe=True)
        r = train_eval(
            model, train_ids, val_ids,
            steps=args.steps, batch_size=args.batch_size, seq_len=args.seq_len,
        )
        # 统计各层 router 的最终 bias 与负载分布（用一批 val 数据）
        model.eval()
        with torch.no_grad():
            x = val_ids[: args.batch_size * args.seq_len].view(args.batch_size, args.seq_len)
            xf = model.embed(x).reshape(-1, cfg.hidden_size)
            idx, _ = model.blocks[0].ffn.router(xf)
            load = torch.bincount(idx.flatten(), minlength=cfg.n_routed_experts).float()
            load = load[load > 0]
            r["load_max_min_ratio"] = (load.max() / load.min()).item()
        results[f"gamma_{gamma}"] = r
    results["fairness"] = "same seed / token budget; report expert load max/min ratio"
    return results


def exp_mla_vs_mha(args):
    """MLA vs 标准 MHA：val loss + KV cache 显存对比。

    TODO(实验者): 当前 minimoe 只有 MLA。要跑此实验需先补一个等 heads 的标准 MHA
    baseline（放 scripts/ 或 minimoe/ 均可，不属 correctness 关卡）。
    这里给出 cache 显存对比的闭式部分（无需训练即可产出）。
    """
    import torch

    from minimoe.config import MoEConfig
    from minimoe.mla import MLACache, mha_cache_bytes

    cfg = MoEConfig()
    B, S = args.batch_size, args.seq_len
    cache = MLACache(cfg, B, cfg.max_seq_len, torch.device("cpu"), torch.float32)
    cache.update(
        torch.zeros(B, S, cfg.kv_lora_rank), torch.zeros(B, S, cfg.qk_rope_head_dim)
    )
    results = {
        "mla_cache_bytes": cache.memory_bytes(),
        "mha_cache_bytes": mha_cache_bytes(cfg, B, S, torch.float32),
        "mla_over_mha": cache.memory_bytes() / mha_cache_bytes(cfg, B, S, torch.float32),
        "val_loss_note": "TODO: 需补 MHA baseline 训练以对比 val loss",
        "fairness": "same seed / token budget; cache bytes at seq_len=%d" % S,
    }
    return results


def exp_mtp(args):
    from minimoe.config import MoEConfig
    from minimoe.model import MiniMoELM

    train_ids, val_ids = load_data(args.token_budget)
    results = {}
    for name, use_mtp in [("no_mtp", False), ("with_mtp", True)]:
        set_seed()
        cfg = MoEConfig()
        model = MiniMoELM(cfg, use_moe=True, use_mtp=use_mtp)
        r = train_eval(
            model, train_ids, val_ids,
            steps=args.steps, batch_size=args.batch_size, seq_len=args.seq_len,
            use_mtp=use_mtp,
        )
        results[name] = r
    results["fairness"] = "same seed / token budget; MTP head discarded at eval (main head only)"
    return results


EXPERIMENTS = {
    "moe_vs_dense": exp_moe_vs_dense,
    "gamma_sweep": exp_gamma_sweep,
    "mla_vs_mha": exp_mla_vs_mha,
    "mtp": exp_mtp,
}


def main():
    p = argparse.ArgumentParser(description="Phase 3.0 ablation scaffold")
    p.add_argument("--exp", required=True, choices=list(EXPERIMENTS))
    p.add_argument("--token-budget", type=int, default=50000)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--out-dir", default="results")
    args = p.parse_args()

    set_seed()
    results = EXPERIMENTS[args.exp](args)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{args.exp}.json")
    with open(out_path, "w") as f:
        json.dump({"exp": args.exp, "seed": SEED, "args": vars(args), "results": results},
                  f, indent=2, ensure_ascii=False)
    print(f"[ablate] wrote {out_path}")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
