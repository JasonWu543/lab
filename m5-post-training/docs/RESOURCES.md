# M4/M5 真实模型权重与后训练数据资源

> 测试关卡全部用 tiny 随机模型（CPU 秒级），**闯关不需要下载任何东西**。
> 本文档服务于通关后的正式实验（S/M 级）：在真实小模型上跑 SFT/DPO/GRPO
> 和推理 benchmark。
>
> 网络提示：直连 HuggingFace 慢的话，先 `export HF_ENDPOINT=https://hf-mirror.com`。

## 1. 模型权重（按算力场景选）

| 模型（HF id） | 参数量 | 用途建议 |
| --- | --- | --- |
| `HuggingFaceTB/SmolLM2-135M` / `-135M-Instruct` | 135M | 本地 Mac 全流程冒烟：CPU/MPS 上 SFT/DPO/GRPO 都能真跑 |
| `Qwen/Qwen2.5-0.5B` | 0.5B | **默认基座**（base 版已在 `m1-foundation/data/qwen2.5-0.5b/`）；M5 从 base 开始 SFT 最有教学价值 |
| `Qwen/Qwen2.5-0.5B-Instruct` | 0.5B | DPO/GRPO 的起点（已会对话，偏好优化效果更可见）；也是 M4 benchmark 的默认模型 |
| `Qwen/Qwen2.5-1.5B` / `-Instruct` | 1.5B | 4090 上的舒适区：full FT bf16 可行，LoRA 更轻 |
| `Qwen/Qwen2.5-3B-Instruct` | 3B | 4090 上 LoRA 后训练 / M4 Phase 4.1 的 target（配 0.5B 做 draft）|

下载方式（权重不进 git，`data/` 已在 .gitignore）：

```python
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen2.5-0.5B-Instruct", local_dir="data/qwen2.5-0.5b-instruct")
```

投机解码（4.1）推荐模型对：draft = `Qwen2.5-0.5B-Instruct`，
target = `Qwen2.5-3B-Instruct`（同族 tokenizer 一致，这是硬要求）。

## 2. SFT 数据（Phase 5.0）

| 数据集（HF id） | 规模 | 说明 |
| --- | --- | --- |
| `HuggingFaceH4/no_robots` | ~9.5k | 人工写的高质量指令数据，小模型 SFT 首选（量小质高，S 级预算够全量过几个 epoch）|
| `yahma/alpaca-cleaned` | 52k | 经典 Alpaca 清洗版，格式简单，适合做数据混合实验 |
| `BelleGroup/train_0.5M_CN` | 0.5M | 中文指令数据（采样 1–5 万条用即可，不要全量）|
| `allenai/tulu-3-sft-mixture` | ~1M | 工业级混合配方参考；只采样子集，重点看它怎么配比 |

```python
from datasets import load_dataset
ds = load_dataset("HuggingFaceH4/no_robots", split="train")
```

## 3. 偏好数据（Phase 5.1 RM/DPO）

| 数据集 | 规模 | 说明 |
| --- | --- | --- |
| `HuggingFaceH4/ultrafeedback_binarized` | ~62k 对 | DPO 事实标准数据集，字段就是 chosen/rejected，开箱即用 |
| `Anthropic/hh-rlhf` | ~161k 对 | RLHF 奠基数据集（helpful/harmless 两轨），适合做 RM 训练对照 |
| `PKU-Alignment/PKU-SafeRLHF` | ~83k | 安全偏好（中英），可选扩展 |

0.5B 模型 + ultrafeedback 采样 5–10k 对、LoRA、1–2 epoch ≈ 单卡 2–4 小时（M 级内）。

## 4. 可验证任务数据（Phase 5.2 GRPO）

| 数据集 | 说明 |
| --- | --- |
| （内置）两位数加法 | SPEC 冻结的默认任务，零下载，reward 严格可验证——**先用它把闭环跑通** |
| `Jiayi-Pan/Countdown-Tasks-3to4` | TinyZero 用的 Countdown 数字游戏，0.5B–3B 上复现 "aha moment" 的社区标准任务 |
| `openai/gsm8k` | 小学数学应用题；0.5B 裸模型太难，建议 1.5B-Instruct + 只取答案数值做 exact-match reward |

## 5. 规模与预算建议（对应 LAB_DESIGN §0.2 的 S/M/L 分级）

- **本地 Mac**：tiny 随机模型过测试（0 成本）→ SmolLM2-135M 真数据冒烟
- **S 级（≤2 GPU·h）**：0.5B + no_robots 全量 SFT（LoRA）
- **M 级（≤12 GPU·h）**：0.5B/1.5B DPO（ultrafeedback 采样）；GRPO 内置任务 → Countdown
- 顺序铁律：本地测试全绿 + 冒烟通过，才允许上云跑正式实验
