"""scripts/train_sft.py — Minimal SFT training script for M5 Phase 5.0.

Usage:
    python3 scripts/train_sft.py \
        --data_file data/sft_train.jsonl \
        --seq_len 512 \
        --steps 1000 \
        --lr 1e-4 \
        --lora_r 8 \
        --lora_alpha 16 \
        --lora_targets q_proj,v_proj \
        --save_path checkpoints/sft_lora.pt

Data format (JSONL):
    Each line is a JSON object with a "messages" key:
    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

Dependencies:
    pip install transformers torch
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from minisft.tokenizer import ByteTokenizer, PAD_ID
from minisft.chat import build_example
from minisft.packing import pack_examples
from minisft.lora import inject_lora, merge_lora


def parse_args():
    p = argparse.ArgumentParser(description="Minimal SFT trainer with LoRA + packing")
    p.add_argument("--data_file", type=str, required=True,
                   help="JSONL file with {messages: [...]} per line")
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2-0.5B",
                   help="HuggingFace model name or local path")
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=float, default=16.0)
    p.add_argument("--lora_targets", type=str, default="q_proj,v_proj",
                   help="Comma-separated list of linear layer name suffixes")
    p.add_argument("--save_path", type=str, default="checkpoints/sft_lora.pt")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--merge_and_save", action="store_true",
                   help="Merge LoRA into base weights before saving")
    p.add_argument("--wrong_mask", action="store_true",
                   help="[对照实验] 关闭 assistant-only masking，对所有 token 计算 loss。"
                        "观察学出的行为与正确 mask 有何不同（POSTMORTEM 必做）。")
    return p.parse_args()


def load_data(data_file: str, tok: ByteTokenizer) -> list[tuple[list[int], list[int]]]:
    """Load and tokenize all examples from JSONL."""
    examples = []
    with open(data_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            ids, labs = build_example(tok, obj["messages"])
            examples.append((ids, labs))
    print(f"Loaded {len(examples)} examples from {data_file}")
    return examples


def compute_loss_manual(model, ids: list[int], labs: list[int]) -> torch.Tensor | None:
    """
    Compute cross-entropy loss using build_example's pre-shifted labels format.
    labels[i] = input_ids[i+1] for active positions — do NOT pass to model(labels=...)
    which would double-shift.
    """
    import torch.nn.functional as F
    ids_t = torch.tensor(ids).unsqueeze(0)
    out = model(input_ids=ids_t)
    logits = out.logits[0]
    labs_t = torch.tensor(labs)
    active = labs_t != -100
    if not active.any():
        return None
    return F.cross_entropy(logits[active], labs_t[active])


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    # ── Load model ──────────────────────────────────────────────────────────
    try:
        from transformers import AutoModelForCausalLM
    except ImportError:
        print("ERROR: transformers not installed. Run: pip install transformers")
        sys.exit(1)

    print(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float32)
    model.train()

    # ── Tokenizer ────────────────────────────────────────────────────────────
    tok = ByteTokenizer()

    # ── Data ─────────────────────────────────────────────────────────────────
    examples = load_data(args.data_file, tok)
    if not examples:
        print("ERROR: No examples loaded")
        sys.exit(1)

    if args.wrong_mask:
        print("[--wrong-mask] 对照模式：labels mask 关闭，所有 token 均参与 loss")
        # Override labels: set labels[i] = input_ids[i+1] for all positions (no -100 masking)
        # TODO: 学生理解：把所有位置都开放 loss，相当于语言模型预训练 loss，不区分角色
        wrong_examples = []
        for ids, _ in examples:
            labs = ids[1:] + [-100]   # 标准 LM shift，无 assistant mask
            wrong_examples.append((ids, labs))
        examples = wrong_examples

    # Pack examples
    packed_batches = pack_examples(examples, seq_len=args.seq_len, pad_id=PAD_ID)
    print(f"Packed into {len(packed_batches)} bins (seq_len={args.seq_len})")

    # ── LoRA ─────────────────────────────────────────────────────────────────
    target_names = [t.strip() for t in args.lora_targets.split(",")]
    n_replaced = inject_lora(model, target_names, r=args.lora_r, alpha=args.lora_alpha)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Injected LoRA into {n_replaced} layers | "
          f"trainable: {n_trainable:,} / {n_total:,} params "
          f"({100*n_trainable/n_total:.2f}%)")

    # ── Optimizer ────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
    )

    # ── Training loop ────────────────────────────────────────────────────────
    print(f"\nTraining for {args.steps} steps...")
    losses = []
    t0 = time.time()

    for step in range(args.steps):
        # Cycle through packed batches
        batch = packed_batches[step % len(packed_batches)]

        # Extract per-example losses using doc_spans + manual CE
        import torch.nn.functional as F
        ids_t = batch["input_ids"].unsqueeze(0)
        out = model(input_ids=ids_t)
        logits = out.logits[0]

        labs = batch["labels"]
        active = labs != -100
        if not active.any():
            continue

        loss = F.cross_entropy(logits[active], labs[active])
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % args.log_every == 0:
            avg = sum(losses[-args.log_every:]) / min(len(losses), args.log_every)
            elapsed = time.time() - t0
            print(f"  step {step+1:5d}/{args.steps} | loss={avg:.4f} | {elapsed:.1f}s")

    # ── Save ─────────────────────────────────────────────────────────────────
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if args.merge_and_save:
        print("Merging LoRA into base weights...")
        merge_lora(model)
        model.save_pretrained(str(save_path.parent / "merged_model"))
        print(f"Saved merged model to {save_path.parent / 'merged_model'}")
    else:
        # Save only LoRA parameters
        lora_state = {
            k: v for k, v in model.state_dict().items()
            if "lora_A" in k or "lora_B" in k
        }
        torch.save(lora_state, save_path)
        print(f"Saved LoRA weights ({len(lora_state)} tensors) to {save_path}")

    print("Done.")


if __name__ == "__main__":
    main()
