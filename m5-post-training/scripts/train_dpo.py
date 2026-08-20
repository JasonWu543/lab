"""scripts/train_dpo.py — DPO 完整训练脚手架。

用法：
    python3 scripts/train_dpo.py [--steps 200] [--beta 0.1] [--lr 1e-4]

功能：
    - 生成 toy 偏好数据（prompt + chosen/rejected 对）
    - 初始化 policy 模型和 ref 模型（tiny Qwen2 或外部 ckpt）
    - 训练循环：每步计算 sequence_logprob → dpo_loss → backward → optimizer step
    - 日志输出：margin、BT 准确率、loss（写入 logs/dpo_log.jsonl）

接线处标记了 TODO：调用学生实现的 sequence_logprob / dpo_loss。
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── 学生模块（接线处）────────────────────────────────────────────────────────
# TODO: 下面两行在学生实现完成后自动生效；骨架阶段会 raise NotImplementedError
from minidpo.logprob import sequence_logprob
from minidpo.dpo import dpo_loss

# ── 配置 ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="DPO 训练脚手架")
    p.add_argument("--steps", type=int, default=200, help="训练步数")
    p.add_argument("--beta", type=float, default=0.1, help="DPO beta")
    p.add_argument("--lr", type=float, default=1e-4, help="AdamW 学习率")
    p.add_argument("--log_dir", type=str, default="logs", help="日志目录")
    p.add_argument("--log_interval", type=int, default=10, help="日志间隔步数")
    p.add_argument("--seed", type=int, default=42, help="随机种子")
    return p.parse_args()


# ── 模型 ─────────────────────────────────────────────────────────────────────

def build_model(seed: int = 42):
    """构造 tiny Qwen2 模型（用于本地 correctness 测试）。"""
    try:
        from transformers import Qwen2Config, Qwen2ForCausalLM
    except ImportError:
        raise SystemExit("请安装 transformers: pip install transformers")

    torch.manual_seed(seed)
    cfg = Qwen2Config(
        vocab_size=259,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        pad_token_id=258,
        attn_implementation="eager",
    )
    return Qwen2ForCausalLM(cfg)


# ── toy 偏好数据 ──────────────────────────────────────────────────────────────

def build_toy_preferences(n_pairs: int = 8, seq_len: int = 12, pad_id: int = 258):
    """构造 toy 偏好对（chosen vs rejected token 模式固定）。

    偏好逻辑：
        - prompt: token [5, 6, 7]（固定）
        - chosen suffix: [10+i, 11+i, 12+i]
        - rejected suffix: [20+i, 21+i, 22+i]
    """
    pairs = []
    for i in range(n_pairs):
        prompt = [5, 6, 7]
        chosen_sfx = [10 + i % 5, 11 + i % 5, 12 + i % 5]
        rejected_sfx = [20 + i % 5, 21 + i % 5, 22 + i % 5]

        def pad_seq(seq):
            s = (prompt + seq)[:seq_len]
            return s + [pad_id] * (seq_len - len(s))

        pairs.append({
            "chosen_ids": torch.tensor(pad_seq(chosen_sfx)),
            "rejected_ids": torch.tensor(pad_seq(rejected_sfx)),
            "prompt_len": len(prompt),
        })
    return pairs


# ── 训练循环 ──────────────────────────────────────────────────────────────────

def train(args):
    torch.manual_seed(args.seed)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "dpo_log.jsonl"

    print(f"[DPO] seed={args.seed} steps={args.steps} beta={args.beta} lr={args.lr}")

    # 构造模型
    model = build_model(args.seed)
    ref_model = copy.deepcopy(model)
    ref_model.requires_grad_(False)
    ref_model.eval()
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    pairs = build_toy_preferences(n_pairs=8)

    records = []
    t0 = time.time()

    for step in range(args.steps):
        # 循环取 pair
        pair = pairs[step % len(pairs)]
        chosen_ids = pair["chosen_ids"].unsqueeze(0)    # (1, L)
        rejected_ids = pair["rejected_ids"].unsqueeze(0)  # (1, L)
        prompt_lens = torch.tensor([pair["prompt_len"]])   # (1,)

        # ── TODO: 调用学生实现 ───────────────────────────────────────────────
        # policy logps
        policy_chosen_logps = sequence_logprob(model, chosen_ids, prompt_lens)
        policy_rejected_logps = sequence_logprob(model, rejected_ids, prompt_lens)

        # ref logps（no_grad）
        with torch.no_grad():
            ref_chosen_logps = sequence_logprob(ref_model, chosen_ids, prompt_lens)
            ref_rejected_logps = sequence_logprob(ref_model, rejected_ids, prompt_lens)

        # DPO loss
        loss, chosen_rew, rejected_rew = dpo_loss(
            policy_chosen_logps, policy_rejected_logps,
            ref_chosen_logps, ref_rejected_logps,
            beta=args.beta,
        )
        # ─────────────────────────────────────────────────────────────────────

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # 日志
        margin = float((chosen_rew - rejected_rew).mean().item())
        bt_acc = float((margin > 0))

        if (step + 1) % args.log_interval == 0 or step == 0:
            rec = {
                "step": step + 1,
                "loss": round(float(loss.item()), 6),
                "margin": round(margin, 6),
                "bt_acc": bt_acc,
                "elapsed": round(time.time() - t0, 2),
            }
            records.append(rec)
            print(f"  step={rec['step']:4d}  loss={rec['loss']:.4f}  "
                  f"margin={rec['margin']:+.4f}  bt_acc={rec['bt_acc']:.2f}")

    # 写日志
    with open(log_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"[DPO] 训练完成，日志 → {log_path}")

    # 验证 ref 参数未变
    any_changed = any(
        not torch.equal(ref_model.state_dict()[n], model.state_dict()[n])
        for n in ref_model.state_dict()
    )
    print(f"[DPO] ref 参数是否已与 policy 不同: {any_changed}（应为 True）")


# ── 入口 ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    train(args)
