#!/usr/bin/env python3
"""给定脚手架：DSV4 三机制消融入口；本地 ``--dry-run`` 只校验计划。"""

import argparse
import json


EXPERIMENTS = {
    "sparse": {"metric": "peak_memory_and_tokens_per_second", "requires": "cuda"},
    "hyper-connection": {"metric": "deep_layer_gradient_ratio", "requires": "cuda"},
    "muon": {"metric": "loss_at_equal_steps", "requires": "cuda"},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mechanism", choices=[*EXPERIMENTS, "all"], default="all")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    chosen = EXPERIMENTS if args.mechanism == "all" else {args.mechanism: EXPERIMENTS[args.mechanism]}
    if not args.dry_run:
        raise RuntimeError("真实消融需要租用 CUDA；请在训练环境接入数据与计时循环")
    print(json.dumps({"dry_run": True, "experiments": chosen}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
