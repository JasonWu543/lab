# Lab 阅读清单

> 与 LAB_DESIGN.md 各模块对应。由 subagent 网络检索整理（2026-08），
> 链接均来自搜索结果。使用建议：动手前只精读打 ★ 的 1–2 篇，
> 其余在实现中遇到问题时查阅 —— 不要用读论文推迟写代码。

---

## M1 — 模型基础

### Phase 1.0 BPE Tokenizer

- ★ [Neural Machine Translation of Rare Words with Subword Units (Sennrich et al., 2016)](https://arxiv.org/abs/1508.07909) — paper — BPE 用于 NLP 的原论文，算法基础，必读。
- ★ [karpathy/minbpe](https://github.com/karpathy/minbpe) — repo — 200 行 Python 从零实现 BPE，配套视频 "Let's build the GPT Tokenizer"，工程细节最清晰的教学实现。
- [Byte-level BPE 讲解（HF LLM Course, Ch6）](https://huggingface.co/learn/llm-course/en/chapter6/5) — course — GPT-2 byte-level BPE 的深入讲解，为何以 byte 为基本单元可消除 OOV。
- [Tokenizer summary（HF Transformers Docs）](https://huggingface.co/docs/transformers/tokenizer_summary) — docs — BPE / WordPiece / SentencePiece / Unigram 系统对比速查。
- [Tokenization is Killing our Multilingual LLM Dream](https://huggingface.co/blog/omarkamali/tokenization) — blog — tokenization 对多语言的不公平影响与工程隐患。

### Phase 1.1 Autograd

- ★ [Automatic Differentiation in ML: a Survey (Baydin et al., JMLR 2018)](https://arxiv.org/abs/1502.05767) — paper — 自动微分最权威综述，forward/reverse mode 的数学本质。
- ★ [karpathy/micrograd](https://github.com/karpathy/micrograd) — repo — 150 行标量级 autograd 引擎，理解"为什么保存前向激活"最直观。
- [Overview of PyTorch Autograd Engine](https://pytorch.org/blog/overview-of-pytorch-autograd-engine/) — blog — 官方 autograd 引擎深度讲解：计算图、backward node、JVP。
- [PyTorch Internals (Edward Z. Yang)](https://blog.ezyang.com/2019/05/pytorch-internals/) — blog — Tensor/stride/dispatcher 的 C++ 层实现细节，理解 view 与内存的最佳材料。
- [A Gentle Introduction to torch.autograd](https://docs.pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html) — docs — 快速对接实践。

### Phase 1.2 Transformer from scratch

- ★ [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762) — paper — 起点。
- ★ [karpathy/build-nanogpt](https://github.com/karpathy/build-nanogpt) — repo — 从空文件逐 commit 构建 GPT-2(124M)，最高质量的工程教学。
- [RoFormer: Rotary Position Embedding (Su et al., 2021)](https://arxiv.org/abs/2104.09864) — paper — RoPE 原论文，现代 decoder 位置编码工业标准。
- [Rotary Embeddings: A Relative Revolution (EleutherAI)](https://blog.eleuther.ai/rotary-embeddings/) — blog — RoPE 直观解释与 vs ALiBi/T5 bias 实验对比。
- [GQA (Ainslie et al., 2023)](https://arxiv.org/abs/2305.13245) — paper — MQA/GQA 如何降 KV cache 而不明显损失质量。
- [The Llama 3 Herd of Models (2024)](https://arxiv.org/abs/2407.21783) — paper — 现代架构选择（GQA/RoPE/SwiGLU）与大规模预训练工程实践的参考文档。

### Phase 1.3 训练框架

- ★ [Mixed Precision Training (Micikevicius et al., 2018)](https://arxiv.org/abs/1710.03740) — paper — master weight / loss scaling / FP32 accumulation，BF16 训练的前置知识。
- ★ [OPT 技术报告 + 训练 logbook (Zhang et al., 2022)](https://arxiv.org/abs/2205.01068) — paper — 附真实训练日志，数十次硬件故障与梯度爆炸重启，训练稳定性最真实的工程文档。
- [PaLM (Chowdhery et al., 2022)](https://arxiv.org/abs/2204.02311) — paper — MFU 指标的出处，loss spike 处理记录。
- [Spike No More (Takase et al., 2023)](https://arxiv.org/abs/2312.16903) — paper — loss spike 根因分析与初始化/embedding 改进。
- [Fault-tolerant Training with torchrun](https://docs.pytorch.org/tutorials/beginner/ddp_series_fault_tolerance.html) — docs — checkpoint 容错的工程落地参考。

---

## M2 — Kernel 与硬件

- ★ [Triton 官方 Tutorials](https://triton-lang.org/main/getting-started/tutorials/) — docs — 从 vector add 到 matmul 的核心编程模式，边看边写。
- ★ [FlashAttention-2 (Dao, 2023)](https://arxiv.org/abs/2307.08691) — paper — IO-aware fused kernel 的经典案例，SRAM tiling 规避 HBM 瓶颈。
- [Triton: An Intermediate Language and Compiler (Tillet et al., 2019)](https://www.semanticscholar.org/paper/661d142c23cb2a3207d5f1ba2ac7ff61f2d4fb2f) — paper — Triton 的 tile-based IR 设计。
- [Roofline model（Modal GPU Glossary）](https://modal.com/gpu-glossary/perf/roofline-model) — blog — arithmetic intensity 与 compute/memory-bound 判断，快速建立直觉。
- [Fused kernel 实战：Linear + CrossEntropy 省 84% 显存](https://towardsdatascience.com/cutting-llm-memory-by-84-a-deep-dive-into-fused-kernels/) — blog — 可跑的 fused kernel 实战。
- [Nsight Compute](https://developer.nvidia.com/nsight-compute) — docs — kernel profiling 标准工具（roofline 可视化、warp stall 分析）。

---

## M3 — 架构机制研究

### Phase 3.0 MoE / DeepSeek-V3

- ★ [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) — paper — MLA、DeepSeekMoE、aux-loss-free balancing、MTP 四大机制的核心一手文献。
- ★ [DeepSeekMoE (Dai et al., 2024)](https://arxiv.org/abs/2401.06066) — paper — 细粒度专家切分 + shared expert 隔离的设计动机。
- [DeepSeek-V2 (2024)](https://arxiv.org/abs/2405.04434) — paper — MLA 的原始提出（低秩 KV 压缩）。
- [Auxiliary-Loss-Free Load Balancing (Wang et al., 2024)](https://arxiv.org/abs/2408.15664) — paper — 用 bias 替代 aux loss 的均衡策略，V3 直接采用。
- [Multi-token Prediction (Gloeckle et al., 2024)](https://arxiv.org/abs/2404.19737) — paper — MTP 收益的原始出处（Meta）。
- [Switch Transformers (Fedus et al., 2021)](https://arxiv.org/abs/2101.03961) — paper — Top-1 routing 与容量因子，MoE 经典基线背景。

### Phase 3.1/3.2 长上下文与新型 attention（backlog）

- [DeltaNet: Parallelizing Linear Transformers with the Delta Rule (2024)](https://arxiv.org/abs/2406.06484) — paper — delta rule 线性 attention 的硬件高效并行化。
- [Gated DeltaNet (2024)](https://arxiv.org/abs/2412.06464) — paper — 门控 delta rule，系统超越 Mamba2。
- [Native Sparse Attention (DeepSeek, 2025)](https://arxiv.org/abs/2502.11089) — paper — 硬件对齐、可原生训练的稀疏 attention。
- [Kimi K2 Technical Report (2025)](https://arxiv.org/abs/2507.20534) — paper — 工业级长上下文 + agentic 系统参考。
- [RULER (2024)](https://arxiv.org/abs/2404.06654) — paper — 长上下文评测标准（needle 的系统化扩展）。

---

## M4 — 推理系统

### Phase 4.0 mini-vLLM

- ★ [PagedAttention / vLLM (Kwon et al., SOSP'23)](https://arxiv.org/abs/2309.06180) — paper — OS 分页思想引入 KV cache 管理，本 phase 的目标蓝图。
- ★ [Continuous Batching 解析（Anyscale）](https://www.anyscale.com/blog/continuous-batching-llm-inference) — blog — iteration-level scheduling 的直观解释与实测数据。
- [SGLang / RadixAttention (2023)](https://arxiv.org/abs/2312.07104) — paper — 跨请求 prefix KV cache 复用。
- [System-Aware KV Cache Optimization Survey (2026)](https://arxiv.org/abs/2607.08057) — paper — KV cache 优化全景地图。
- [LLM 推理 benchmark 指标详解（TTFT/TPOT/percentile）](https://neelmishra.github.io/blog/mlops/llm-inference/inference-benchmarking.html) — blog — 为什么报告必须带输入长度与百分位。

### Phase 4.1 投机解码

- ★ [Speculative Decoding (Leviathan et al., ICML'23)](https://arxiv.org/abs/2211.17192) — paper — 原论文：draft + verify 范式与无损性证明。
- ★ [Speculative Sampling (Chen et al., DeepMind 2023)](https://arxiv.org/abs/2302.01318) — paper — acceptance-rejection 的严格推导，比 Leviathan 版更清晰。
- [EAGLE (2024)](https://arxiv.org/abs/2401.15077) — paper — feature 层自回归草稿，draft-model-free 的高效方案。
- [Medusa (2024)](https://arxiv.org/abs/2401.10774) — paper — 多解码头 + tree attention 验证。
- [FasterDecoding/Medusa](https://github.com/FasterDecoding/Medusa) — repo — tree speculation 完整实现，可对照复现。

---

## M5 — 后训练

- ★ [InstructGPT (Ouyang et al., 2022)](https://arxiv.org/abs/2203.02155) — paper — SFT → RM → PPO 三阶段范式的奠基论文。
- ★ [DPO (Rafailov et al., 2023)](https://arxiv.org/abs/2305.18290) — paper — 把 RLHF 化为分类损失的闭式推导，工业标准方法。
- ★ [DeepSeekMath / GRPO (2024)](https://arxiv.org/abs/2402.03300) — paper — GRPO 来源：去 critic、组内均值作 baseline。
- [DeepSeek-R1 (2025)](https://arxiv.org/abs/2501.12948) — paper — 纯 RL 涌现推理行为的标志性工作。
- [Illustrating RLHF（HF blog）](https://huggingface.co/blog/rlhf) — blog — RLHF 全流程图解与主流库梳理，工程入门起点。
- [Reward Hacking (Lilian Weng, 2024)](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/) — blog — reward hacking 成因、表现与缓解，Phase 5.2 验收项的理论依据。

---

## M6 — 数据与 Scaling

- ★ [Chinchilla (Hoffmann et al., 2022)](https://arxiv.org/abs/2203.15556) — paper — compute-optimal 配比，Phase 6.1 直接复刻其方法论。
- ★ [Deduplicating Training Data Makes LMs Better (Lee et al., 2022)](https://arxiv.org/abs/2107.06499) — paper — 去重影响的系统量化，Phase 6.0 核心实验的参照。
- [Scaling Laws for Neural LMs (Kaplan et al., 2020)](https://arxiv.org/abs/2001.08361) — paper — 幂律关系的原始确立。
- [FineWeb 数据卡与技术说明](https://huggingface.co/datasets/HuggingFaceFW/fineweb) — docs — 15T token 数据集的完整过滤/去重管线，工业级参考。
- [RedPajama (Weber et al., 2024)](https://arxiv.org/abs/2411.12372) — paper — 开源数据管线与质量信号设计。
- [Does Data Contamination Detection Work? (Fu et al., 2024)](https://arxiv.org/abs/2410.18966) — paper — 主流检测方法接近随机猜测的警示，做评测前必读。

---

## M7 — Agent 软件工程

- ★ [Building Effective Agents（Anthropic）](https://www.anthropic.com/engineering/building-effective-agents) — blog — workflow vs agent 的区分与可组合模式，最权威实战参考。
- ★ [How we built our multi-agent research system（Anthropic）](https://www.anthropic.com/engineering/multi-agent-research-system) — blog — orchestrator + 并行 subagent 架构拆解，Phase 7.1 的直接参照。
- [Effective harnesses for long-running agents（Anthropic）](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents) — blog — 跨 context window 的长时 agent 工程。
- [SWE-bench (Jimenez et al., ICLR 2024)](https://arxiv.org/abs/2310.06770) — paper — coding agent 能力的标准评测，度量指标设计可借鉴。
- [Context Engineering (Philipp Schmid)](https://www.philschmid.de/context-engineering) — blog — 把瓶颈从模型归因到上下文管理，Phase 7.3 的方法论。
- [My LLM Coding Workflow Going into 2026 (Addy Osmani)](https://addyosmani.com/blog/ai-coding-workflow/) — blog — specs-first、小步迭代、AI-on-AI review 的一手工作流经验。
