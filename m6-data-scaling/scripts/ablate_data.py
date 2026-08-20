"""Phase 6.0 消融实验脚手架 — ablate_data.py

功能：从 TinyStories valid 集采样固定 token 预算，分别构建
  1. raw         — 原始文档
  2. filtered    — 经 quality_filter 过滤后的文档
  3. filtered+deduped — 过滤后再去重

三组各用同一 tiny Transformer 模型与超参训练，比较 val loss。
结果写入 ablation_results.jsonl。

用法：
  python3 scripts/ablate_data.py [--data-dir PATH] [--token-budget N] [--out OUT]

依赖：transformers, datasets（HuggingFace）
若没有 GPU，在 CPU 上用小配置跑；结果用于定性对比，不要求精确数值。
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import time
from typing import Any

import numpy as np

# ─────────────── 可选依赖（HuggingFace）───────────────
try:
    import torch
    from torch.utils.data import DataLoader, Dataset
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False
    print("[警告] torch 未安装，跳过训练部分，仅做数据统计")

# ─────────────── minidata 导入 ────────────────────────
from minidata.filters import apply_filters
from minidata.minhash import dedup

# ═════════════════════════════════════════════════════
# 1. 数据加载
# ═════════════════════════════════════════════════════

def load_tinystories_valid(data_dir: pathlib.Path) -> list[str]:
    """从 data_dir 加载 TinyStories validation 集文档列表。

    支持两种格式：
    - data_dir/tinystories/valid.txt（每篇用 <|endoftext|> 分隔）
    - data_dir/valid.txt（同上）
    - data_dir/TinyStories-valid.txt（同上）

    若找不到文件，打印提示并返回空列表。
    """
    candidates = [
        data_dir / "tinystories" / "valid.txt",
        data_dir / "valid.txt",
        data_dir / "TinyStories-valid.txt",
        data_dir / "tinystories" / "TinyStories-valid.txt",
    ]
    for path in candidates:
        if path.exists():
            print(f"[数据] 加载 {path}")
            text = path.read_text(encoding="utf-8")
            docs = [d.strip() for d in text.split("<|endoftext|>") if d.strip()]
            print(f"[数据] 共 {len(docs)} 篇文档")
            return docs
    print(f"[警告] 未找到 TinyStories valid 集，尝试路径：{[str(c) for c in candidates]}")
    print("[提示] 请将 TinyStories valid 集放到 --data-dir 指定目录")
    return []


def sample_to_token_budget(
    docs: list[str],
    token_budget: int,
    seed: int = 42,
) -> list[str]:
    """从 docs 中按序采样，直到累计 token 数（空白分词估计）达到 token_budget。"""
    rng = random.Random(seed)
    shuffled = docs[:]
    rng.shuffle(shuffled)
    sampled: list[str] = []
    total_tokens = 0
    for doc in shuffled:
        n_tok = len(doc.split())
        if total_tokens + n_tok > token_budget:
            break
        sampled.append(doc)
        total_tokens += n_tok
    print(f"[采样] {len(sampled)} 篇，≈{total_tokens:,} tokens")
    return sampled


# ═════════════════════════════════════════════════════
# 2. 数据管线（三组）
# ═════════════════════════════════════════════════════

def build_groups(
    raw_docs: list[str],
    token_budget: int,
    seed: int = 42,
) -> dict[str, list[str]]:
    """构建 raw / filtered / filtered+deduped 三组，各组 token 数尽量相近。"""
    print("\n[管线] 构建数据组...")

    # raw：直接采样
    raw = sample_to_token_budget(raw_docs, token_budget, seed=seed)

    # filtered
    all_filtered, stats = apply_filters(raw_docs)
    print(f"[过滤] 保留 {stats.kept}/{stats.kept + stats.dropped} 篇"
          f"，drop_reasons={stats.drop_reasons}")
    filtered = sample_to_token_budget(all_filtered, token_budget, seed=seed)

    # filtered + deduped
    if all_filtered:
        kept_indices, dup_pairs = dedup(
            all_filtered, threshold=0.8, num_perm=128, bands=32, k=3
        )
        deduped_docs = [all_filtered[i] for i in kept_indices]
        print(f"[去重] 去重前 {len(all_filtered)}，去重后 {len(deduped_docs)}"
              f"，去除 {len(dup_pairs)} 对")
        deduped = sample_to_token_budget(deduped_docs, token_budget, seed=seed)
    else:
        deduped = []

    groups = {
        "raw": raw,
        "filtered": filtered,
        "filtered+deduped": deduped,
    }
    for name, g in groups.items():
        print(f"  {name}: {len(g)} 篇")
    return groups


# ═════════════════════════════════════════════════════
# 3. 极简 Transformer 训练（字符级 / 词级）
# ═════════════════════════════════════════════════════

class SimpleTokenizer:
    """极简字符级 tokenizer（无需 HuggingFace tokenizer）。"""

    def __init__(self):
        self.vocab: dict[str, int] = {}
        self.inv_vocab: dict[int, str] = {}

    def fit(self, docs: list[str]) -> "SimpleTokenizer":
        chars: set[str] = set()
        for doc in docs:
            chars.update(doc)
        self.vocab = {"<pad>": 0, "<unk>": 1}
        for c in sorted(chars):
            self.vocab[c] = len(self.vocab)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        return self

    def encode(self, text: str, max_len: int = 512) -> list[int]:
        ids = [self.vocab.get(c, 1) for c in text]
        return ids[:max_len]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)


class TextDataset:
    """字符级语言模型数据集（torch Dataset）。"""

    def __init__(self, docs: list[str], tokenizer: SimpleTokenizer, seq_len: int = 256):
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        # 合并所有 token ids
        all_ids: list[int] = []
        for doc in docs:
            all_ids.extend(tokenizer.encode(doc, max_len=100_000))
            all_ids.append(0)  # 文档间 pad

        self.data = all_ids

    def __len__(self) -> int:
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx: int):
        import torch
        chunk = self.data[idx : idx + self.seq_len + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


def build_tiny_model(vocab_size: int, device: str = "cpu"):
    """构建一个极小的 Transformer 语言模型（用于消融对比，不追求绝对性能）。"""
    import torch
    import torch.nn as nn

    class TinyLM(nn.Module):
        def __init__(self, vocab_size: int, d_model: int = 64,
                     n_heads: int = 2, n_layers: int = 2,
                     seq_len: int = 256):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, d_model)
            self.pos_embed = nn.Embedding(seq_len, d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=0.0, batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.lm_head = nn.Linear(d_model, vocab_size)
            self.seq_len = seq_len

        def forward(self, x):
            B, T = x.shape
            pos = torch.arange(T, device=x.device).unsqueeze(0)
            h = self.embed(x) + self.pos_embed(pos)
            # causal mask
            mask = torch.triu(
                torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
            )
            h = self.transformer(h, mask=mask)
            return self.lm_head(h)

    return TinyLM(vocab_size).to(device)


def train_and_eval(
    train_docs: list[str],
    val_docs: list[str],
    tokenizer: SimpleTokenizer,
    device: str = "cpu",
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 3e-4,
    seq_len: int = 256,
    seed: int = 42,
) -> dict[str, float]:
    """训练 tiny LM，返回 {train_loss, val_loss}。"""
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    train_ds = TextDataset(train_docs, tokenizer, seq_len=seq_len)
    val_ds = TextDataset(val_docs, tokenizer, seq_len=seq_len)

    if len(train_ds) == 0 or len(val_ds) == 0:
        return {"train_loss": float("nan"), "val_loss": float("nan")}

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = build_tiny_model(tokenizer.vocab_size, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits.view(-1, tokenizer.vocab_size), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # 评估
    model.eval()
    train_losses, val_losses = [], []
    with torch.no_grad():
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            l = criterion(logits.view(-1, tokenizer.vocab_size), y.view(-1))
            train_losses.append(l.item())
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            l = criterion(logits.view(-1, tokenizer.vocab_size), y.view(-1))
            val_losses.append(l.item())

    return {
        "train_loss": float(np.mean(train_losses)),
        "val_loss": float(np.mean(val_losses)),
    }


# ═════════════════════════════════════════════════════
# 4. 主流程
# ═════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 6.0 数据消融实验")
    parser.add_argument(
        "--data-dir",
        type=pathlib.Path,
        default=pathlib.Path("../m1-foundation/data/tinystories"),
        help="TinyStories valid 集所在目录",
    )
    parser.add_argument(
        "--token-budget",
        type=int,
        default=500_000,
        help="每组训练集 token 预算（空白分词估计，默认 500K）",
    )
    parser.add_argument(
        "--val-budget",
        type=int,
        default=50_000,
        help="验证集 token 预算（默认 50K）",
    )
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=pathlib.Path("ablation_results.jsonl"),
        help="输出 JSONL 路径",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只做数据统计，跳过训练（快速检查管线）",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Phase 6.0 数据消融实验")
    print("=" * 60)

    # 加载数据
    all_docs = load_tinystories_valid(args.data_dir)
    if not all_docs:
        print("[错误] 无法加载数据，退出")
        return

    # 分出验证集（从头取，确保固定）
    rng = random.Random(args.seed)
    rng.shuffle(all_docs)
    # 拆分：前 10% 作验证集，其余为训练候选
    split = max(100, len(all_docs) // 10)
    val_raw = all_docs[:split]
    train_raw = all_docs[split:]

    val_docs = sample_to_token_budget(val_raw, args.val_budget, seed=args.seed)

    # 构建三组训练集
    groups = build_groups(train_raw, args.token_budget, seed=args.seed)

    # 统计
    results: list[dict[str, Any]] = []
    for name, train_docs in groups.items():
        rec: dict[str, Any] = {
            "group": name,
            "n_docs": len(train_docs),
            "approx_tokens": sum(len(d.split()) for d in train_docs),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        print(f"\n[组] {name}：{rec['n_docs']} 篇，≈{rec['approx_tokens']:,} tokens")

        if args.dry_run or not _TORCH_OK:
            rec["train_loss"] = None
            rec["val_loss"] = None
            print("  (dry-run / torch 未安装，跳过训练)")
        else:
            # 用全量文档拟合 tokenizer（保证词表一致）
            tokenizer = SimpleTokenizer()
            tokenizer.fit(all_docs)
            print(f"  tokenizer vocab_size={tokenizer.vocab_size}")

            device = "cuda" if _TORCH_OK and __import__("torch").cuda.is_available() else "cpu"
            print(f"  device={device}, epochs={args.epochs}")

            t0 = time.time()
            metrics = train_and_eval(
                train_docs=train_docs,
                val_docs=val_docs,
                tokenizer=tokenizer,
                device=device,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                seq_len=args.seq_len,
                seed=args.seed,
            )
            elapsed = time.time() - t0
            rec.update(metrics)
            rec["elapsed_s"] = round(elapsed, 1)
            print(f"  train_loss={metrics['train_loss']:.4f}, "
                  f"val_loss={metrics['val_loss']:.4f}, "
                  f"time={elapsed:.1f}s")

        results.append(rec)

    # 写 JSONL
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n[完成] 结果写入 {args.out}")
    print("\n==== 消融结果汇总 ====")
    for r in results:
        vl = f"{r['val_loss']:.4f}" if isinstance(r.get("val_loss"), float) else "N/A"
        print(f"  {r['group']:20s}: n_docs={r['n_docs']:5d}, val_loss={vl}")

    print("\n思考题（写进 POSTMORTEM）：")
    print("  1. 去重对 val_loss 的影响是正是负？与你的预测一致吗？")
    print("  2. 过滤步骤去掉了哪类文档最多？合理吗？")
    print("  3. 如果 token 预算固定，去重后文档数减少意味着什么？")


if __name__ == "__main__":
    main()
