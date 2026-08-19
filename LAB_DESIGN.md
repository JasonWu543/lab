# AI Systems Lab — 总体设计

> 版本 v0.1 · 2026-08 制定
> 目标：通过一套自设计的 lab 体系，建立完整的个人技术闭环 ——
> 能写核心算法、能搭训练系统、能优化 kernel、能部署推理、能做可信实验，
> 并能像小型 Tech Lead 一样调度 Agent 并发交付。

---

## 0. 全局约定

### 0.1 组织方式：独立模块 + 模块内 phase 串联

七个大模块**相互独立**（各自一个目录/仓库，代码不互相 import）。
模块内部的 lab 按 phase 递进，后一个 phase 在前一个 phase 的代码上生长，
以此训练"抽象被后续需求检验"的能力。

模块之间只通过**标准产物**衔接，不共享代码：

- 模型权重：HF 格式 checkpoint（safetensors + config.json）
- 数据：tokenized 后的标准格式（如 .bin + index）
- 结果：统一的 benchmark JSON / markdown 报告

例如：M4 推理模块默认加载官方 Qwen 小模型权重，
也可以加载 M1 自己训出来的 checkpoint —— 但两个模块的代码互不依赖。

```text
lab/
├── LAB_DESIGN.md            # 本文档
├── READINGS.md              # 各模块阅读清单（论文/文章，含精读标记）
├── m1-foundation/           # 模型基础：Tensor → Transformer → Trainer
├── m2-kernels/              # Triton / TileLang kernel
├── m3-arch-study/           # 架构机制研究：DeepSeek / Kimi / MoE
├── m4-inference/            # 推理系统：mini-vLLM → 投机解码 → PD
├── m5-post-training/        # SFT → DPO → GRPO
├── m6-data-scaling/         # 数据工程 + Scaling law
└── m7-agent-engineering/    # Agent 并发开发 / Review / Context 实验
```

### 0.2 算力策略：本地开发，云端实验

- **本地 Mac（MPS/CPU）**：写代码、跑单测、correctness 级小实验（≤10M 参数、玩具数据）。所有代码必须支持 `--device cpu/mps/cuda` 切换。
- **云端租用单卡 4090/5090（24–32GB）**：只跑正式实验。上云前 checklist：本地单测全绿、config 固定、预估 GPU 小时。

算力分级（每个实验必须事先标注级别）：

| 级别 | 预算 | 用途 |
| --- | --- | --- |
| S | ≤2 GPU·h | correctness 验证、冒烟测试 |
| M | ≤12 GPU·h | 正式消融实验 |
| L | ≤48 GPU·h | 阶段性 capstone（每模块最多 1 个）|

### 0.3 三种开发模式（古法 vs vibe 的比例）

| 模式 | 我负责 | Agent 负责 | 适用 |
| --- | --- | --- | --- |
| Foundation | 核心实现、推导、debug | Review、提问、补测试 | 第一次实现某机制 |
| Copilot | 架构、接口、关键路径 | boilerplate、测试、文档 | 第二次工程化 |
| Lead | Spec、拆解、验收、Review | 多分支并发实现 | 第三次扩展/系统集成 |

原则：同一机制第一次手搓，第二次让 Agent 加速，第三次只控制接口与验收。
每个 phase 在 SPEC 里事先声明用哪种模式，事后在 postmortem 里复盘是否合适。

### 0.4 统一验收模板

每个 phase 完成时必须具备（小 phase 可精简，但不可为零）：

1. `SPEC.md` — 问题、范围、非目标、验收标准（动手前写）
2. `DESIGN.md` — 接口、关键选择、备选方案（可与 SPEC 合并）
3. Correctness tests（与 PyTorch/参考实现对齐）
4. Performance benchmark（有 baseline、有数字）
5. 至少一个 ablation 或 failure injection
6. PR + Review 记录（哪怕是自己 review 自己）
7. `POSTMORTEM.md` — 错误、返工、学到什么

实验纪律（来自全局框架，此处落地）：
固定 seed、记全 config、一次只改一个变量、警惕数值不稳与数据泄露。

---

## 1. M1 — 模型基础（Foundation）

**回答的问题**：能不能独立写出并理解一个现代 Transformer 及其训练系统。
**模式**：以 Foundation 为主。这是整个 lab 里手搓比例最高的模块。
**单元级拆解与验收测试见 `m1-foundation/PLAN.md`**（v0.2 起 phase 顺序调整为
BPE → Tensor → Transformer → Trainer，BPE 作为第一周热身兼 PR 闭环演练）。

### Phase 1.0 — BPE Tokenizer

GPT-2 风格 regex 预分词 + BPE 训练（vocab/merges）+ encode/decode +
效率优化。与 HF tokenizers 对齐验证，round-trip fuzz 测试。
产出 8k–16k vocab 供后续 phase 使用。

### Phase 1.1 — Tensor 与 Autograd（最小可用版）

最小 Tensor 系统：shape/stride/view/broadcast、elementwise/matmul/reduction、
反向传播计算图、Linear/RMSNorm/SwiGLU，与 PyTorch forward/backward 对齐。

- 验收：随机 shape 对齐、gradcheck、非连续 tensor 测试；
  故意注入 broadcast/stride/梯度累积 bug 再定位修复。
- 理解目标：view 为何不复制内存、broadcast 梯度如何归约、
  inplace 为何破坏 autograd、backward 到底保存了什么。
- 算力：纯本地。

### Phase 1.2 — Qwen-like Transformer from scratch

（从这里起切换到 PyTorch，Tensor phase 的产物封存为独立成果）

Tokenizer、RMSNorm、RoPE、GQA、SwiGLU、causal attention、KV cache、
weight tying、训练/生成/checkpoint。

- 规模：correctness 用 10–30M；正式 50–150M，BF16，seq 512–2048。
- 验收：32 样本稳定过拟合；中断恢复轨迹一致；
  cached/non-cached decoding logits 对齐；
  给出参数量/FLOPs/MFU/显存的手算估算并与实测对比。
- 附加：加载转换后的官方 Qwen 小模型权重，验证模块映射正确。
- 算力：S 级 + 一次 M 级正式训练。

### Phase 1.3 — 训练框架

在 1.2 的训练脚本上生长出真正的 Trainer：
配置系统、DataLoader/packing/masking、grad accumulation、BF16、clipping、
LR scheduler、checkpoint/resume（含 RNG state）、日志与异常检测、
evaluation hook、profiler、OOM 处理。

- **Failure injection 是本 phase 的灵魂**：训练中途 kill、checkpoint 写一半、
  数据 NaN、loss spike、optimizer state 丢失 —— 每种都要能解释、恢复、复现。
- 验收标准不是"能跑"，是"出问题时能快速解释和恢复"。
- 模式：Copilot（架构和关键路径自己写，boilerplate 交给 Agent）。
- 产物：一个可复用的 Trainer，M3/M6 的实验都用它跑。

**M1 capstone（L 级）**：用自己的全套代码在云端单卡完整训练一个 ~100M 模型，
产出 HF 格式 checkpoint + 训练报告（loss 曲线、MFU、中断恢复演示）。

---

## 2. M2 — Kernel 与硬件（Kernels）

**回答的问题**：能不能定位瓶颈并写出有效的 GPU 优化。
**模式**：Foundation（kernel 本体手搓）+ Agent 负责 benchmark 脚手架。
**依赖**：无（独立模块，用 PyTorch 官方算子做 baseline）。

### Phase 2.0 — 入门三件套
Fused RMSNorm → Fused SwiGLU → Softmax/CrossEntropy（Triton）。

### Phase 2.1 — 进阶
RoPE → blockwise attention（FlashAttention 思想的简化版）。

### Phase 2.2 — 面向 MoE
Grouped GEMM（为 M3 的 MoE 实验提供性能直觉）。

每个 kernel 统一要求：

- forward + backward correctness（多 shape、FP32/BF16）
- warmup 后稳定 benchmark vs PyTorch baseline
- roofline 或至少 bandwidth/compute bound 判断
- **解释**：为什么快、什么 shape 快、什么 shape 反而慢、
  瓶颈是 launch / memory / compute
- 算力：S 级为主（kernel benchmark 很省卡时，但必须在真卡上跑，MPS 无意义）

---

## 3. M3 — 架构机制研究（Arch Study）

**回答的问题**：能不能把论文创新变成公平、可信的消融实验。
**模式**：Copilot。模型骨架可复用 M1 经验重写一份轻量版（本模块独立，不 import M1）。
**命名纪律**：这些是 "mechanism study"，不宣称"复现 XX 模型"。

### Phase 3.0 — MoE / DeepSeek-V3 机制（旗舰，与现有研究互补）
MLA、fine-grained + shared experts、aux-loss-free load balancing、MTP。
在 30–100M 规模回答：

- 等 activated FLOPs 下 MoE 是否优于 Dense？
- bias 更新速度如何影响 expert collapse？
- shared expert 学到了什么？
- MLA 省多少 KV cache、损失多少质量？
- MTP 改善的是 next-token 还是更远预测？

### Phase 3.1 — DeepSeek-V4 机制
三个独立单卡消融：dense vs compressed/sparse attention（长序列显存/吞吐/召回）；
standard residual vs mHC（深层梯度范数与稳定性）；AdamW vs Muon（等 token 收敛速度）。

### Phase 3.2 — Kimi 长上下文机制
Delta attention / attention residuals / 状态压缩。
观察：KV·state memory 随序列长度变化、needle/copy/长程依赖任务、
短上下文是否退化。

- 算力：每个消融 M 级；全模块最多一个 L 级。
- 公平性要求：对照组严格等算力（activated FLOPs 或 wall-clock 二选一并声明）。

---

## 4. M4 — 推理系统（Inference)

**回答的问题**：能不能理解 KV cache、调度、吞吐与延迟的系统权衡。
**模式**：Lead 倾向 —— 这是 Agent 并发开发（M7 Lab B）的主战场。
**基座模型**：官方 Qwen 0.5B 级权重 + transformers 官方实现，
**不自己写模型结构** —— 本模块的训练对象是模型之外的系统层
（cache manager、scheduler、batching、sampler）。
唯一允许碰模型代码的场景：为接入 paged KV cache 对 attention 的
cache 读写做最薄的 adapter 层，且必须先用官方实现验证 logits 一致。

### Phase 4.0 — 从 naive 到 mini-vLLM
递进实现：无 cache 解码 → KV cache → static batching →
continuous batching → paged KV cache → request scheduler →
prefix cache → streaming API。

- 全程记录：TTFT、TPOT、tokens/s、p50/p95/p99、KV 占用、并发退化曲线。
- 终点：用 vLLM/SGLang 部署同一模型，解释自己的实现慢在哪。

### Phase 4.1 — 投机解码
draft/target、acceptance-rejection 的严格分布校正、dynamic speculative length。

- 验收：采样分布与 target 一致（统计检验）；acceptance rate 曲线；
  不同 temperature 下加速比；找出"投机反而更慢"的边界。

### Phase 4.2 — PD 分离（功能模拟版）
单卡做功能与调度模拟：两类请求建模、双 worker、KV transfer API、
chunked prefill、workload replay。
性能版（双卡）列为**可选扩展**，不进主线。

- 算力：4.0/4.1 各一次 M 级正式 benchmark；开发调试尽量本地小模型。

---

## 5. M5 — 后训练（Post-Training）

**回答的问题**：能不能搭起 SFT → 偏好优化 → 在线 RL 的完整闭环并理解每步的坑。
**模式**：Copilot；RL loss 核心部分 Foundation。
**基座**：官方 0.5B 级模型 + transformers/PEFT + LoRA（单卡可承受），
同样不自己写模型结构，练的是训练管线与算法本身。

### Phase 5.0 — SFT
chat template、loss mask、sequence packing、assistant-only loss、
LoRA vs full FT、数据混合、catastrophic forgetting 观察。

- 验收：小数据过拟合；packing 前后 loss 对齐；
  **错误 mask 的对照实验**（故意 mask 错，看模型学出什么）。

### Phase 5.1 — 偏好优化
Reward model + DPO：chosen/rejected 数据检查、reference model、KL 的作用。

### Phase 5.2 — GRPO / Online RL
可验证任务（数学/格式约束/简单代码执行）上实现：
rollout、group advantage、reward normalization、KL regularization、
rollout freshness、reward hacking 检查。

- 训练与 rollout 分阶段执行，不追求并发。
- 算力：5.0 S 级；5.1/5.2 各 M 级。

---

## 6. M6 — 数据与 Scaling（Data & Scaling）

**回答的问题**：算力有限时如何做实验、数据如何决定模型。
**模式**：Copilot。训练部分用 M1 产出的 Trainer 思路重写轻量版或直接借用产物。

### Phase 6.0 — 数据工程
tokenizer 训练、质量过滤、MinHash 去重、contamination 检测、packing、
数据混合、DataLoader 吞吐。

- 核心实验：模型/token 数/算力全部固定，只改数据处理，
  看 validation loss 与下游任务变化。

### Phase 6.1 — Scaling law
固定总 FLOPs，训练 10/20/40/80M 若干组合，拟合 compute-optimal frontier，
**外推下一组最优配置并真实训练验证外推是否成立**。

- 算力：本模块是算力大户，6.1 整体按一个 L 级预算规划。
- 顺序建议：放在 M1 完成之后、与 M3 可并行。

---

## 7. M7 — Agent 软件工程（Agent Engineering）

**回答的问题**：什么任务可以并发、接口如何冻结、如何减少集成返工。
**特殊性**：本模块不是"学一个技术"，而是**演练场 + 度量体系**，
贯穿其他模块执行，同时有自己的独立实验。

### Phase 7.0 — 单 Agent PR 闭环（最先做）
在任一模块的真实需求上严格执行：
Issue → Design → Branch → Impl → Tests → Commit → PR → Review → Fix → Merge → Postmortem。
训练：写验收标准、小 commit、diff review、要求 Agent 给证据而非"完成了"。

### Phase 7.1 — 并发模块开发（主实验，挂靠 M4）
mini-vLLM 天然可拆：Agent1 KV cache manager / Agent2 scheduler /
Agent3 sampler / Agent4 benchmark+测试。
我负责：接口定义、所有权划分、公共数据结构冻结、integration checkpoint、Review、合并。

度量指标（每次演练记录）：
并行加速比、merge conflict 数、接口返工次数、首次 CI 通过率、
人工干预次数、逃逸 bug、token 成本、Issue→Merge 时长。

### Phase 7.2 — 快速 Review 训练
构造含 15–20 个已知问题的 PR（数值错误、race、silent shape bug、
不公平 benchmark、性能回退、覆盖缺口……），
多个 Review Agent 分维度审（correctness/perf/design/test），
我合并意见、去重、定优先级。演练材料放独立子目录，不污染真实代码。

### Phase 7.3 — Context Engineering 对照实验
同一任务在不同 context 策略下重复：无指令 / 长 AGENTS.md / 分层指令 /
精简根指令+模块局部指令 / 全量上下文 / 只给接口+测试。
记录 token、时间、错误数、无关修改、首次测试通过率。

- 算力：不需要 GPU，消耗的是 token 预算。

---

## 8. 执行路线（12 周冲刺版，每周 15–25h）

> v0.2 调整：总工期压缩到 3 个月。代价是明确砍量（见 8.2）
> 并提高 Agent 杠杆：Foundation 模式只保留给真正的核心路径，
> 其余一律 Copilot/Lead。

### 8.1 十二周计划

模块独立，允许并行，但同一时间**最多两条活跃线**（一主一副）。

| 周 | 主线 | 副线 | 出口标志（不达标不进下一周）|
| --- | --- | --- | --- |
| W1 | M1: 1.0 BPE → 1.1 Tensor/Autograd（最小版）| M7: 7.0 用 BPE 演练 PR 闭环 | BPE round-trip + gradcheck 通过 + 第一个完整 PR |
| W2–3 | M1: 1.2 Qwen-like Transformer | — | 过拟合 + cached/non-cached logits 对齐 |
| W4 | M1: 1.3 Trainer + 云端训 ~50–100M | — | 中断恢复演示 + checkpoint 产出 |
| W5–6 | M4: 4.0 mini-vLLM | M7: 7.1 四 Agent 并发开发（主实验）| continuous batching 跑通 + vLLM 对比报告 |
| W7 | M4: 4.1 投机解码 | M2: 2.0 fused RMSNorm/SwiGLU | 分布一致性检验通过 |
| W8–9 | M3: 3.0 MoE/DeepSeek 机制 | M7: 7.2 Review 训练 | MoE vs Dense 等算力结论 |
| W10 | M5: 5.0 SFT → 5.1 DPO | M6: 6.0 数据工程（压缩版）| SFT/DPO checkpoint + mask 对照实验 |
| W11 | M5: 5.2 GRPO | — | 可验证任务上 reward 曲线上升且无 hacking |
| W12 | 缓冲 + 总结报告 | 可选：补砍掉的项 | 全部 postmortem 汇总成一篇总结 |

### 8.2 为压缩到 3 个月砍掉的内容（记入 backlog，不删设计）

- M3: 3.1（DeepSeek-V4 机制）、3.2（Kimi 机制）→ 全部延后
- M4: 4.2（PD 分离模拟）→ 延后
- M2: 2.1（attention kernel）、2.2（grouped GEMM）→ 只保 2.0 三件套
- M6: 6.1（scaling law）→ 降级为 W12 可选项；6.0 压缩为
  「tokenizer + 去重 + 一个数据消融」
- M7: 7.3（context engineering 对照）→ 融入日常，不单独立项
- Phase 1.1 Tensor 从「完整系统」降为「最小可用」：
  只做 matmul/elementwise/broadcast/reduction + 反向 + 三个 module，
  一周内完成，点到为止

### 8.3 弹性原则

- 每周日做一次 15 分钟 review：本周 postmortem、下周计划微调。
- 出口标志是硬门槛：没达标就砍下周的副线来补，不顺延总工期。
- 若 W6 结束时进度落后超过一周，优先再砍 M3 的消融数量，保 M5 闭环。

---

## 8A. 单个 phase 的标准执行流程

每个 phase 都走同一个循环（这本身就是 M7 在练的东西）：

```text
1. SPEC     我和 Agent 讨论 → Agent 起草 SPEC.md → 我修改确认
            （范围、非目标、验收标准、S/M/L 算力级别、开发模式）
2. DESIGN   接口和数据结构先定死（Lead/Copilot 模式下这是冻结线）
3. IMPL     按声明的模式分工：
            Foundation → 我写核心，Agent review + 补测试
            Copilot    → 我写关键路径，Agent 写 boilerplate/测试/脚手架
            Lead       → Agent 并发实现，我只看 diff 和接口
4. VERIFY   跑 SPEC 里写好的验收项，全绿才算完成；
            云端实验前先过本地 checklist（0.2 节）
5. MERGE    PR + review 记录 → merge → tag
6. POST     POSTMORTEM.md：返工点、Agent 用得好/坏的地方、下次改什么
```

节奏参考：小 phase（如 2.0 的单个 kernel）整循环 1–2 天；
大 phase（如 4.0）把 IMPL 拆成多个子 PR，每个子 PR 走 3–5 的小循环。

---

## 9. 明确不做的事（Non-goals）

- 不做规模复现：不训练 >200M 的模型，不宣称"复现 DeepSeek/Kimi"。
- 不做多卡分布式训练（TP/PP/EP）：留待有多卡条件后另立模块。
- 不重写完整 PyTorch：Phase 1.0 的 Tensor 系统点到为止。
- 不追新：模块进行中不因新模型发布而改需求，新技术记入 backlog，月度 review 时再决定。
- 不做产品化：所有 serving 实验止于 benchmark 与报告，不做前端/运维。

---

## 10. 下一步

1. 确认本文档（可迭代修订，版本号递增）。
2. 建立 `m1-foundation/` 仓库骨架 + 第一份 SPEC（Phase 1.0 Tensor）。
3. 同步启动 M7 Phase 7.0：Phase 1.0 本身就用完整 PR 闭环来做。
