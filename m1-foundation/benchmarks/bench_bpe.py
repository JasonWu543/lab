"""B1 吞吐 benchmark：V1 / V2 / V3(1,4,8 workers) / HF tokenizers 对比。

用法：
    python benchmarks/bench_bpe.py --input data/tinystories/TinyStoriesV2-GPT4-valid.txt
    python benchmarks/bench_bpe.py --input data/tinystories/TinyStoriesV2-GPT4-train.txt --skip-naive

说明：
- V1 naive 在大语料上不可用：默认只在文件前 --naive-mb MB 的子集上跑并线性外推。
- train 阶段的「预分词 vs merge」分段计时由实现内部记录（实现可打印日志），
  本脚本只测端到端 wall time 与 encode 吞吐。
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 允许脚本直跑
from minilm.tokenizer.bpe import BPETokenizer

EOT = "<|endoftext|>"
VOCAB = 8192


def timed(fn, *args, **kw):
    t0 = time.perf_counter()
    out = fn(*args, **kw)
    return out, time.perf_counter() - t0


def subset_file(src: Path, mb: float, dst: Path) -> Path:
    """截取前 mb MB，并回退到最后一个完整 EOT 边界，避免截断文档。"""
    raw = src.read_bytes()[: int(mb * 1e6)]
    cut = raw.rfind(EOT.encode())
    dst.write_bytes(raw[: cut + len(EOT.encode())] if cut > 0 else raw)
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--naive-mb", type=float, default=5.0)
    ap.add_argument("--skip-naive", action="store_true")
    ap.add_argument("--workers", type=int, nargs="+", default=[1, 4, 8])
    args = ap.parse_args()

    size_mb = args.input.stat().st_size / 1e6
    rows = []

    if not args.skip_naive:
        sub = subset_file(args.input, args.naive_mb, args.input.with_suffix(".naive_subset.txt"))
        sub_mb = sub.stat().st_size / 1e6
        _, t = timed(BPETokenizer.train, sub, VOCAB, [EOT], algo="naive")
        rows.append(("V1 naive", f"{sub_mb:.1f}MB 子集", t, f"外推全量 ≈ {t * size_mb / sub_mb / 60:.1f} min"))

    _, t2 = timed(BPETokenizer.train, args.input, VOCAB, [EOT], algo="incremental", num_workers=1)
    rows.append(("V2 incremental", f"{size_mb:.0f}MB", t2, ""))

    for w in args.workers:
        if w == 1:
            continue
        _, t3 = timed(BPETokenizer.train, args.input, VOCAB, [EOT], algo="incremental", num_workers=w)
        rows.append((f"V3 workers={w}", f"{size_mb:.0f}MB", t3, f"vs V2: {t2 / t3:.2f}x"))

    try:
        from tokenizers import Tokenizer, models, pre_tokenizers, trainers

        def hf_train():
            hf = Tokenizer(models.BPE())
            hf.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
            hf.train([str(args.input)], trainers.BpeTrainer(
                vocab_size=VOCAB, special_tokens=[EOT],
                initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
            return hf

        _, th = timed(hf_train)
        rows.append(("HF tokenizers", f"{size_mb:.0f}MB", th, f"vs V2: {t2 / th:.1f}x faster"))
    except ImportError:
        print("(未安装 tokenizers，跳过 HF 对比)")

    # encode 吞吐（用 V2 训练出的 tokenizer 在 10MB 文本上）
    tok = BPETokenizer.train(args.input, VOCAB, [EOT], algo="incremental",
                             num_workers=max(args.workers))
    text = args.input.read_text(encoding="utf-8")[: int(10e6)]
    _, te = timed(tok.encode, text)
    enc_mbs = len(text.encode()) / 1e6 / te

    print(f"\n{'variant':<18}{'corpus':<14}{'train(s)':<12}note")
    for name, corp, t, note in rows:
        print(f"{name:<18}{corp:<14}{t:<12.1f}{note}")
    print(f"\nencode 吞吐: {enc_mbs:.2f} MB/s (10MB 样本)")


if __name__ == "__main__":
    main()
