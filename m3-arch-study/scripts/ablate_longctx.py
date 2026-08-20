#!/usr/bin/env python3
"""Given scaffold for fixed-state memory and toy recall ablations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minikda.config import KDAConfig
from minikda.delta import delta_rule_recurrent
from minikda.tasks import generate_associative_recall_batch


def memory_curve(cfg: KDAConfig, lengths=(128, 512, 2048, 8192)):
    """Return comparable fp32 bytes for standard KV and delta matrix state."""
    fixed = cfg.num_heads * cfg.head_dim * cfg.head_dim * 4
    return [
        {"tokens": t, "mha_kv_bytes": 2 * cfg.num_heads * cfg.head_dim * t * 4,
         "delta_state_bytes": fixed}
        for t in lengths
    ]


def recall_curve(pair_counts=(4, 8, 16, 32), dim=16, seed=321):
    """Measure exact-pair retrieval as associations outgrow a fixed D×D state."""
    generator = torch.Generator().manual_seed(seed)
    rows = []
    for pairs in pair_counts:
        keys = torch.randn(pairs, dim, generator=generator)
        values = torch.randn(pairs, dim, generator=generator)
        # Write all pairs, then read each key without another write (beta=0).
        q = torch.cat([torch.zeros_like(keys), keys]).view(1, 1, 2 * pairs, dim)
        k = torch.cat([keys, keys]).view(1, 1, 2 * pairs, dim)
        v = torch.cat([values, torch.zeros_like(values)]).view(1, 1, 2 * pairs, dim)
        beta = torch.cat([torch.ones(pairs), torch.zeros(pairs)]).view(1, 1, -1)
        output = delta_rule_recurrent(q, k, v, beta)[0, 0, pairs:]
        # Independent nearest-value oracle: the intended answer for query i is value i.
        nearest = torch.cdist(output, values).argmin(-1)
        accuracy = (nearest == torch.arange(pairs)).float().mean().item()
        rows.append({"pairs": pairs, "state_dim": dim, "recall_accuracy": accuracy})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="validate scaffold without training")
    parser.add_argument("--seed", type=int, default=320)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    cfg = KDAConfig()
    tokens, targets = generate_associative_recall_batch(
        batch_size=4, num_pairs=3, vocab_size=16,
        generator=torch.Generator().manual_seed(args.seed),
    )
    result = {
        "mode": "dry-run" if args.dry_run else "local-ablation",
        "seed": args.seed,
        "sample_shape": list(tokens.shape),
        "target_shape": list(targets.shape),
        "memory_curve": memory_curve(cfg),
    }
    if args.dry_run:
        result["note"] = "full long-sequence recall sweep is intended for the rented accelerator"
    else:
        result["recall_curve"] = recall_curve(seed=args.seed + 1)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
