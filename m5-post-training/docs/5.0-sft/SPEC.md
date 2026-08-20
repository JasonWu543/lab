# SPEC — Phase 5.0: SFT（chat template / loss mask / packing / LoRA）

> 状态：FROZEN（接口已冻结）
> 模式：Copilot —— loss mask 与 LoRA 数学手写；模板渲染/数据管道提示给足
> 基座：官方 transformers（测试用 tiny 随机 Qwen2Config）。不重写模型结构。
> LoRA **手搓**（不用 peft）——注入/合并本身就是学习点
> 算力：correctness 本地；真实 SFT 一次 S 级（W10）
> 工期：约 0.5 周

## 1. 问题

搭一条正确的 SFT 管线：多轮对话渲染成 chat template、只对 assistant
段计算 loss、sequence packing 不串味、LoRA 低秩微调可合并回主干。

学完必须能回答（写进 POSTMORTEM）：
- 为什么只训 assistant 段？把 user 段也算 loss 会学出什么行为？
  （做完 mask 错误对照实验后回答）
- packing 时如果不隔离文档间 attention，会发生什么泄漏？
- LoRA 为什么用 BA 而不是一个满秩矩阵？alpha/r 起什么作用？
  merge 之后推理开销为什么是零？

## 2. 范围与非目标

范围：固定 chat template（见下）、-100 label mask、贪心装箱 packing +
块对角 attention mask、LoRALinear 注入/合并、错误 mask 对照实验脚本。
非目标：不做真实大数据集清洗（M6 的事）、不做 NEFTune 等 trick、
不做多轮工具调用模板。

## 3. Chat template（冻结，ChatML 风格）

```
<|im_start|>{role}\n{content}<|im_end|>\n
```
整段对话 = 逐消息拼接，结尾不加 generation prompt。
tokenizer 用**给定的** `ByteTokenizer`（脚手架，字节级 + 3 个特殊 token：
`<|im_start|>`=256, `<|im_end|>`=257, `<|pad|>`=258，vocab_size=259）。

## 4. 冻结接口（minisft/）

```python
# minisft/tokenizer.py —— 给定脚手架，不需要实现
class ByteTokenizer:
    def encode(self, text: str) -> list[int]     # 特殊 token 原子化
    def decode(self, ids: list[int]) -> str

# minisft/chat.py
def render_chat(messages: list[dict]) -> str:
    """messages: [{"role": "user"|"assistant"|"system", "content": str}, ...]
    按上方模板渲染成单个字符串。"""

def build_example(tok: ByteTokenizer, messages: list[dict]
                  ) -> tuple[list[int], list[int]]:
    """返回 (input_ids, labels)。labels 与 input_ids 等长：
    仅 assistant 消息的 content 及其后的 <|im_end|> 位置为「下一 token 预测」
    的目标，其余一律 -100。注意 labels 已右移对齐（labels[i] 对应
    input_ids[i] 位置的预测目标），末位无目标为 -100。"""

# minisft/packing.py
def pack_examples(examples: list[tuple[list[int], list[int]]],
                  seq_len: int, pad_id: int
                  ) -> list[dict]:
    """贪心装箱（不切分单条样本；放不下开新箱；超长样本截断）。
    每箱返回 {"input_ids": (S,), "labels": (S,),
             "attention_mask": (1, S, S) bool 块对角（文档间互不可见），
             "doc_spans": [(start, end), ...]}，pad 位置 labels=-100。"""

# minisft/lora.py
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float): ...
    # forward = base(x) + (alpha/r) * B(A(x))；A ~ N(0, 1/r) 初始化，B 零初始化
    # base 权重冻结；只有 A/B 可训练

def inject_lora(model, target_names: list[str], r: int, alpha: float) -> int:
    """把名字以 target_names 之一结尾的 nn.Linear 换成 LoRALinear，
    返回替换数量。其余参数全部 requires_grad=False。"""

def merge_lora(model) -> None:
    """把所有 LoRALinear 的 BA 合并进 base 权重并还原为普通 nn.Linear。"""

# scripts/train_sft.py —— 脚手架（含 --wrong-mask 对照开关），接线处留 TODO
```

## 5. 验收标准（tests/test_sft.py，CPU）

| 编号 | 通过条件 |
| --- | --- |
| T1 | render_chat 快照：多轮（system+user+assistant×2）渲染结果与冻结字符串完全一致 |
| T2 | **labels mask**：多轮对话中恰好 assistant 的 content+<|im_end|> 位置有目标、其余全 -100；右移对齐正确（拿一条手工算的短例逐位断言）|
| T3 | packing：装箱不切样本、pad 的 labels=-100、doc_spans 无重叠覆盖正确；块对角 mask 恰好隔离文档 |
| T4 | **packing 无泄漏**：tiny 模型上验证块对角 mask 的结构效果——第一个文档的 logits 与不加 mask 时完全一致（它前面没有别的文档）、后续文档的 logits 在有/无 mask 下显著不同（证明跨文档可见性确实被切断）。（注：不能直接比 packed vs 单独 forward 的数值——RoPE 的绝对位置不同，这本身是个思考题，答案写进 POSTMORTEM）|
| T5 | LoRA 注入：替换数量正确；可训练参数只有 A/B；注入后（B 零初始化）前向与原模型完全一致 |
| T6 | LoRA merge：训练几步后 merge，merge 前后前向一致（atol 1e-5）且模型里不再有 LoRALinear |
| T7 | 端到端：tiny 模型 + LoRA 在 8 条固定对话上 200 步内 assistant-loss 降 >60%，且 user 段 loss 不下降（证明 mask 生效）|

## 6. 产物

- `minisft/*.py` 全绿
- 错误 mask 对照实验记录（train_sft.py --wrong-mask，观察学出什么）
- `docs/5.0-sft/POSTMORTEM.md`
