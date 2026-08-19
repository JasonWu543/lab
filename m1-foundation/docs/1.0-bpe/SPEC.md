# SPEC — Phase 1.0: Byte-level BPE Tokenizer

> 状态：DRAFT（待确认后冻结接口）
> 模式：Foundation —— 训练与 encode 核心由我手写；Agent 先行编写测试与 benchmark 脚手架
> 算力：无 GPU，纯本地
> 工期：约 3 天（W1 前半）

## 1. 问题

从零实现 GPT-2 风格的 byte-level BPE：给定原始文本语料，训练出
vocab + merges；并实现与训练结果一致的 encode/decode。
**效率是一等目标**：在 TinyStories 全量语料上，naive 实现慢到不可用，
本 phase 要亲手经历「naive → 算法优化 → 并行化」的完整提速路径，
并量化每一步的收益。

## 2. 数据集

[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)
（Eldan & Li 2023，约 2M 条 GPT-4 生成的短故事，train 文本 ~2GB）。
下载后放 `m1-foundation/data/tinystories/`（gitignored）。
开发用 valid split（~20MB）快速迭代，正式训练/benchmark 用 train split。

## 3. 范围

- GPT-2 预分词 regex（`regex` 库，含 `\p{L}`/`\p{N}` unicode 属性类）
- byte-level 初始词表（256 个 byte 为基础单元，无 OOV）
- BPE 训练三个版本，逐级提速，结果必须完全一致：
  - **V1 naive**：全量重新统计 pair 频次，每轮 O(corpus)——作为正确性基准
  - **V2 增量**：只更新受本轮 merge 影响的 pair 计数（+ 堆/有序结构取 max）
  - **V3 并行预分词**：大文件按 chunk 切分（在 special token 边界对齐切点，
    保证不切断文档），`multiprocessing` 多进程预分词 + 词频聚合，
    之后的 merge 循环在聚合后的 word-count 表上进行
- special tokens（`<|endoftext|>`）：文档边界标记，encode 时整体保留、不参与合并
- save/load：vocab + merges 落盘为 json，可复现加载
- 吞吐报告：V1/V2/V3/HF tokenizers 四方对比（预分词与 merge 阶段分开计时）

## 4. 非目标（明确不做）

- 不做 GPU/Rust 加速（与 HF tokenizers 的剩余差距解释清楚即可，不追平）
- 不做 Unigram/WordPiece/SentencePiece
- 不做 vocab > 16k 的大规模训练
- 不做流式/增量语料训练

## 5. 冻结接口

```python
# minilm/tokenizer/bpe.py

class BPETokenizer:
    """Byte-level BPE, GPT-2 style."""

    @classmethod
    def train(
        cls,
        input_path: str | Path,          # 原始文本文件（UTF-8）
        vocab_size: int,
        special_tokens: list[str] | None = None,
        num_workers: int = 1,            # 1 = 串行；>1 走 V3 并行预分词
        algo: Literal["naive", "incremental"] = "incremental",
    ) -> "BPETokenizer": ...

    def encode(self, text: str) -> list[int]: ...

    def decode(self, ids: list[int]) -> str: ...

    def save(self, dirpath: str | Path) -> None: ...

    @classmethod
    def load(cls, dirpath: str | Path) -> "BPETokenizer": ...

    @property
    def vocab_size(self) -> int: ...
```

约定（算法歧义处一次性定死，测试将按此执行）：

- 预分词 pattern 使用 GPT-2 原版 regex，逐 pre-token 独立做 BPE，
  合并不跨 pre-token 边界
- 每轮合并选**频次最高**的 pair；频次并列时选**字节序较大**的 pair
  （与 CS336 约定一致，保证训练确定性）
- special tokens 的 id 排在普通 vocab 之后，encode 前先按 special token
  切分文本，special 段直接映射 id
- decode 对非法 byte 序列使用 `errors="replace"`

## 6. 验收标准（tests/test_bpe.py，先于实现存在）

| 编号 | 测试 | 通过条件 |
| --- | --- | --- |
| T1 | round-trip fuzz | 随机 unicode/emoji/中英混排/空白字符，`decode(encode(s)) == s` |
| T2 | 训练确定性 | 同语料同参数训练两次，vocab/merges 完全一致 |
| T3 | 参考对齐 | 小语料上与 HF tokenizers 同参数训练的 merges 序列一致（前 N 条）|
| T4 | special tokens | `<|endoftext|>` 永不被切开；与普通文本混排正确 |
| T5 | save/load | 落盘再加载后 encode 结果逐 id 一致 |
| T6 | 三版一致性 | V1/V2/V3 在同一语料上训练出的 vocab/merges 完全一致（V3 用不同 num_workers 也一致）|
| T7 | chunk 边界 | 并行切分点对齐测试：构造 special token 落在 chunk 边界附近的文件，词频统计与串行一致 |
| B1 | 吞吐 benchmark | TinyStories 上报告 V1(子集外推)/V2/V3(1/4/8 workers)/HF tokenizers 的耗时与 MB/s；预分词与 merge 分开计时；给出加速比曲线与剩余差距解释 |

## 7. 产物

- `minilm/tokenizer/bpe.py`（实现，V1/V2/V3 同一接口）
- `tests/test_bpe.py` + `benchmarks/bench_bpe.py`
- 在 TinyStories train split 上训练的 8k vocab（`<|endoftext|>` 为 special token），
  存至 `m1-foundation/artifacts/tokenizer-8k/`（vocab/merges json 体积小，进 git）
- `docs/1.0-bpe/POSTMORTEM.md`（含提速路径的量化记录：每步优化带来几倍）

## 8. 开发流程（M7 Phase 7.0 演练）

1. 本 SPEC 确认 → 接口冻结
2. Agent 写 `tests/test_bpe.py`（全红）→ 我 review 测试本身
3. 我在 branch `m1/bpe` 上手写实现，小步 commit
4. 全绿后发 PR → Agent review diff → 修改 → merge
5. POSTMORTEM
