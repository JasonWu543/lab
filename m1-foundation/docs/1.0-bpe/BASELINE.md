# Phase 1.0 参考实现基线（TinyStories valid, 21MB, vocab 8192）

参考实现（`reference/1.0-bpe/bpe_solution.py`）在 Mac 本地的实测，
你的实现完成后跑 `benchmarks/bench_bpe.py` 对比：

| variant | corpus | train(s) | 备注 |
| --- | --- | --- | --- |
| V1 naive | 2MB 子集 | 33.2 | 线性外推全量 ≈ 6.2 min |
| V2 incremental | 21MB | 34.4 | vs V1 外推 ≈ 11x |
| V3 workers=4 | 21MB | 31.5 | vs V2 仅 1.09x (!) |
| V3 workers=8 | 21MB | 31.2 | vs V2 仅 1.10x (!) |
| HF tokenizers | 21MB | 1.1 | vs V2 约 31x |

encode 吞吐：1.46 MB/s（10MB 样本）。

## 思考题（过关后写进你的 POSTMORTEM）

1. V3 并行预分词为什么几乎没加速（1.09x）？
   分别测出预分词阶段和 merge 循环各占多少秒，用 Amdahl 定律解释。
2. 在 2GB 的 train split 上，V3 相对 V2 的加速比会变大还是变小？为什么？
   （提示：预分词耗时随语料线性涨，merge 循环耗时随 unique word 数
   亚线性涨。）先预测，之后真跑 train split 时验证。
3. HF tokenizers 的 31x 差距来自哪几层？哪些差距是语言层面（Rust vs
   Python），哪些是算法/并行层面？
4. encode 只有 1.46 MB/s，瓶颈在哪个函数？有什么改进方案？
