"""scripts/train_grpo.py — GRPO 完整训练脚手架。

用法：
    python3 scripts/train_grpo.py [--rounds 200] [--G 4] [--lr 5e-4]

功能：
    - 个位数加法任务（a,b∈[1,4]，prompt="Q:{a}+{b}=\\nA:"）
    - SFT 格式预热（可选，--sft_steps 控制）
    - GRPO 训练循环：rollout → reward → group_advantages → grpo_loss → step
    - 日志输出：avg_reward、pg_loss、kl、clip_frac（写入 logs/grpo_log.jsonl）
    - Hacking 监控：每轮记录 format_score_ratio（格式得分占总 reward 的比例），
      用于检测 reward hacking（模型只生成"格式对但值错"的答案来拿 0.1 分）

接线处标记了 TODO：调用学生实现的 rollout / reward_fn / group_advantages / grpo_loss。
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
# TODO: 下面各行在学生实现完成后自动生效；骨架阶段会 raise NotImplementedError
from minigrpo.rollout import rollout
from minigrpo.reward import reward_fn
from minigrpo.advantage import group_advantages
from minigrpo.loss import grpo_loss
from minisft.tokenizer import ByteTokenizer

# ── 配置 ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="GRPO 训练脚手架")
    p.add_argument("--rounds", type=int, default=200, help="GRPO 训练轮数")
    p.add_argument("--G", type=int, default=4, help="每 prompt 的采样数")
    p.add_argument("--lr", type=float, default=5e-4, help="AdamW 学习率")
    p.add_argument("--clip_eps", type=float, default=0.2, help="PPO clip epsilon")
    p.add_argument("--kl_coef", type=float, default=0.04, help="KL 正则系数")
    p.add_argument("--max_new_tokens", type=int, default=8, help="最大生成 token 数")
    p.add_argument("--temperature", type=float, default=1.0, help="采样温度")
    p.add_argument("--sft_steps", type=int, default=100, help="SFT 预热步数（0=跳过）")
    p.add_argument("--sft_lr", type=float, default=1e-2, help="SFT 预热学习率")
    p.add_argument("--log_dir", type=str, default="logs", help="日志目录")
    p.add_argument("--log_interval", type=int, default=5, help="日志间隔轮数")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    return p.parse_args()


# ── 模型 ─────────────────────────────────────────────────────────────────────

def build_model(seed: int = 0):
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
        attn_implementation="eager",
    )
    return Qwen2ForCausalLM(cfg)


# ── SFT 格式预热 ──────────────────────────────────────────────────────────────

def sft_warmup(model, tok: ByteTokenizer, n_steps: int, lr: float):
    """个位数加法 SFT 格式预热（脚手架实现，非学生任务）。"""
    if n_steps <= 0:
        return
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    eos_id = tok.eos_id
    data = []
    for a in range(1, 5):
        for b in range(1, 5):
            prompt_str = f"Q:{a}+{b}=\nA:"
            answer_str = str(a + b)
            prompt_ids = tok.encode(prompt_str)
            answer_ids = tok.encode(answer_str) + [eos_id]
            full_ids = prompt_ids + answer_ids
            data.append((full_ids, len(prompt_ids)))

    model.train()
    for step in range(n_steps):
        idx = step % len(data)
        full_ids, prompt_len = data[idx]
        ids_t = torch.tensor(full_ids).unsqueeze(0)
        logits = model(input_ids=ids_t).logits[0]
        labels = torch.full((len(full_ids),), -100, dtype=torch.long)
        for i in range(prompt_len - 1, len(full_ids) - 1):
            labels[i] = full_ids[i + 1]
        active = labels != -100
        if not active.any():
            continue
        loss = F.cross_entropy(logits[active], labels[active])
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    print(f"  [SFT] 预热完成，{n_steps} 步")


# ── Hacking 检查 ──────────────────────────────────────────────────────────────

def compute_format_ratio(rewards_list: list[float]) -> float:
    """格式 reward 占比：有格式（reward > 0）但值错（reward < 1.0）的比例。

    这是 reward hacking 的信号：模型只学会输出"数字"但不学会算对答案，
    靠大量 0.1 分的格式奖励推高总 reward。
    正常训练：format_ratio 应下降（模型从格式对→答案对）。
    """
    n_total = len(rewards_list)
    if n_total == 0:
        return 0.0
    n_format = sum(1 for r in rewards_list if 0.0 < r < 1.0)
    return n_format / n_total


# ── GRPO 一轮 ────────────────────────────────────────────────────────────────

def grpo_round(model, ref_model, tok, prompts: list[str], optimizer,
               G: int, max_new_tokens: int, temperature: float,
               clip_eps: float, kl_coef: float, generator=None) -> dict:
    """一轮 GRPO：rollout → reward → advantage → loss → step。"""

    # ── TODO: 调用学生实现 ───────────────────────────────────────────────────
    # 1. rollout
    out = rollout(model, tok, prompts, G=G, max_new_tokens=max_new_tokens,
                  temperature=temperature, generator=generator)
    input_ids = out["input_ids"]     # (B, T)
    old_logps = out["old_logps"]     # (B, T)
    mask = out["mask"]               # (B, T)
    completions = out["completions"] # list[str], len=B

    # 2. reward
    flat_prompts = [p for p in prompts for _ in range(G)]
    rewards_list = [reward_fn(flat_prompts[b], completions[b])
                    for b in range(len(flat_prompts))]
    rewards_t = torch.tensor(rewards_list).reshape(len(prompts), G)

    # 3. advantage
    adv_2d = group_advantages(rewards_t)   # (n_prompts, G)
    adv_1d = adv_2d.reshape(-1)            # (B,)

    # 4. 当前策略 logps
    model.train()
    logits = model(input_ids=input_ids).logits        # (B, T, V)
    logp_all = F.log_softmax(logits[:, :-1, :], dim=-1)
    cur_logps = torch.zeros_like(input_ids, dtype=logp_all.dtype)
    cur_logps[:, 1:] = logp_all.gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)

    # 5. ref logps（no_grad）
    with torch.no_grad():
        ref_logits = ref_model(input_ids=input_ids).logits
        ref_logp_all = F.log_softmax(ref_logits[:, :-1, :], dim=-1)
        ref_logps = torch.zeros_like(input_ids, dtype=ref_logp_all.dtype)
        ref_logps[:, 1:] = ref_logp_all.gather(
            -1, input_ids[:, 1:].unsqueeze(-1)
        ).squeeze(-1)

    # 6. loss + step
    loss, stats = grpo_loss(
        cur_logps, old_logps.detach(), ref_logps,
        adv_1d, mask.long(),
        clip_eps=clip_eps, kl_coef=kl_coef,
    )
    # ─────────────────────────────────────────────────────────────────────────

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return {
        "avg_reward": float(torch.tensor(rewards_list).mean().item()),
        "rewards_list": rewards_list,
        "pg_loss": float(stats["pg_loss"].item()),
        "kl": float(stats["kl"].item()),
        "clip_frac": float(stats["clip_frac"].item()),
        "format_ratio": compute_format_ratio(rewards_list),
    }


# ── 训练循环 ──────────────────────────────────────────────────────────────────

PROMPTS = [
    "Q:1+1=\nA:", "Q:2+3=\nA:", "Q:3+4=\nA:", "Q:2+2=\nA:",
    "Q:1+4=\nA:", "Q:3+1=\nA:", "Q:4+2=\nA:", "Q:1+3=\nA:",
]


def train(args):
    torch.manual_seed(args.seed)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "grpo_log.jsonl"

    tok = ByteTokenizer()
    print(f"[GRPO] seed={args.seed} rounds={args.rounds} G={args.G} "
          f"lr={args.lr} clip_eps={args.clip_eps} kl_coef={args.kl_coef}")

    # 构造模型
    model = build_model(args.seed)
    # SFT 预热
    if args.sft_steps > 0:
        sft_warmup(model, tok, args.sft_steps, args.sft_lr)

    # KL 的 reference 必须是 RL 起点（即 SFT 预热后的策略）。
    ref_model = copy.deepcopy(model)
    ref_model.requires_grad_(False)
    ref_model.eval()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    generator = torch.Generator().manual_seed(args.seed)

    records = []
    t0 = time.time()
    reward_history = []

    for round_idx in range(args.rounds):
        info = grpo_round(
            model, ref_model, tok,
            prompts=PROMPTS[:4],  # 每轮 4 prompts
            optimizer=optimizer,
            G=args.G,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            clip_eps=args.clip_eps,
            kl_coef=args.kl_coef,
            generator=generator,
        )
        reward_history.append(info["avg_reward"])

        if (round_idx + 1) % args.log_interval == 0 or round_idx == 0:
            rec = {
                "round": round_idx + 1,
                "avg_reward": round(info["avg_reward"], 4),
                "pg_loss": round(info["pg_loss"], 6),
                "kl": round(info["kl"], 6),
                "clip_frac": round(info["clip_frac"], 4),
                "format_ratio": round(info["format_ratio"], 4),
                "elapsed": round(time.time() - t0, 2),
            }
            records.append(rec)
            print(
                f"  round={rec['round']:4d}  reward={rec['avg_reward']:.4f}  "
                f"pg={rec['pg_loss']:.4f}  kl={rec['kl']:.6f}  "
                f"clip={rec['clip_frac']:.3f}  "
                f"fmt_ratio={rec['format_ratio']:.3f}"
            )

    # 汇总
    if len(reward_history) >= 20:
        first10 = sum(reward_history[:10]) / 10
        last10 = sum(reward_history[-10:]) / 10
        print(f"\n[GRPO] 首 10 轮均值={first10:.4f}  末 10 轮均值={last10:.4f}  "
              f"delta={last10 - first10:.4f}")

    # 写日志
    with open(log_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"[GRPO] 训练完成，日志 → {log_path}")

    # Hacking 检查报告
    if records:
        fmt_ratios = [r["format_ratio"] for r in records]
        print(f"[GRPO] Hacking 检查：format_ratio 首尾 = "
              f"{fmt_ratios[0]:.3f} → {fmt_ratios[-1]:.3f}；"
              f"均值={sum(fmt_ratios)/len(fmt_ratios):.3f}")
        if fmt_ratios[-1] > 0.7:
            print("  ⚠ format_ratio 末尾 >0.7，疑似 reward hacking！"
                  "模型可能只在学习生成数字格式，而非算对答案。")


# ── 入口 ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    train(args)
