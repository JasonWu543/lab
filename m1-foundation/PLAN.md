# M1 Foundation — 模块开发计划

> 对应 LAB_DESIGN.md 的 M1，工期 W1–W4（约 4 周，半强度）。
> 本文档回答三个问题：分成哪些单元、每个单元怎么开发（谁写什么）、
> 每个单元怎么算"通过"。

---

## 0. 开发方法：测试先行的闯关制

**角色分工（师生模式）**：Agent 是出题人 + 助教 —— 负责 SPEC、骨架代码、
测试关卡、参考答案、review 和分级提示；**你是实现者** —— 手写核心代码闯关。
参考答案放 `reference/`，规则：卡住超过 30 分钟才看，看完要能说出
自己的版本差在哪。

借鉴 CS336 的作业组织方式，但反过来由我们自己造关卡：

1. **接口先冻结**：每个单元开工前，先在 `SPEC.md` 里定死函数签名/类接口。
2. **测试先写**：Agent 根据接口写好 `tests/test_<unit>.py`（对照 PyTorch
   或参考实现的数值对齐测试），此时测试全红。
3. **实现闯关**：按声明的模式实现，直到 `pytest tests/test_<unit>.py` 全绿。
4. **过关动作**：PR → review → merge → 在本文档的进度表打勾。

这样"完成"不由感觉定义，由测试定义；同时 Agent 写的测试本身也要 review
（测试写错比实现写错更隐蔽）。

**设备约定**：所有代码 `--device cpu/mps/cuda` 可切换；
本模块除 Phase 1.3 的正式训练外全部本地完成。

```text
m1-foundation/
├── PLAN.md              # 本文档
├── pyproject.toml
├── minilm/              # 主包
│   ├── tokenizer/       # Phase 1.0
│   ├── tensor/          # Phase 1.1（独立小系统，不被后续 import）
│   ├── model/           # Phase 1.2（PyTorch）
│   └── training/        # Phase 1.3
├── tests/               # 每个单元一个测试文件，先于实现存在
├── benchmarks/          # 吞吐/显存测量脚本
├── configs/             # 训练配置（yaml）
└── docs/                # 各 phase 的 SPEC / POSTMORTEM
```

---

## Phase 1.0 — BPE Tokenizer（W1 前半，约 3 天）

**为什么从这里开始**：纯算法 + 纯 Python，无框架依赖；边界条件多，
非常适合第一次完整演练「SPEC → 测试先行 → PR 闭环」（M7 Phase 7.0）。

**模式**：Foundation（BPE 训练与 encode 核心手搓；Agent 写测试和 benchmark 脚手架）。

**数据集**：TinyStories（~2GB train / ~20MB valid），效率优化是一等目标 ——
naive 版在全量语料上慢到不可用，亲手走完「算法优化 → 并行化」的提速路径。

| 单元 | 内容 | 验收测试 |
| --- | --- | --- |
| U0.1 BPE 训练 V1 | naive 版：GPT-2 regex 预分词 → 全量统计 pair 频次 → 迭代合并 | 小语料上与 HF tokenizers 同参数训练的 merges 一致；tie-breaking 规则明确并测试 |
| U0.2 encode/decode | 应用 merge 规则编码；byte-level 回退；special tokens | round-trip fuzz；`<|endoftext|>` 不被切开；save/load 一致 |
| U0.3 效率 V2 | 增量 pair 计数（只更新受 merge 影响的部分）+ 堆取 max | 与 V1 输出完全一致；量化提速倍数 |
| U0.4 并行 V3 | multiprocessing 预分词：chunk 在 special token 边界对齐切分 → 多进程词频统计 → 聚合后 merge | 与 V1/V2 一致（不同 worker 数亦一致）；chunk 边界构造测试；1/4/8 workers 加速比曲线 |

**过关标准**：四个单元测试全绿 + benchmark 报告（V1/V2/V3/HF 四方对比，
预分词与 merge 分开计时）+ `docs/1.0-bpe/POSTMORTEM.md`
（重点复盘：第一次 PR 闭环里 Agent 用得好/坏的地方 + 每步优化的量化收益）。

**产物**：TinyStories 上训练的 8k vocab，供 Phase 1.2/1.3 使用。

---

## Phase 1.1 — 最小 Tensor 与 Autograd（W1 后半–W2 初，约 4 天）

**范围控制**（12 周版已降级为最小可用）：不做完整 PyTorch，只做到
能解释「view/broadcast/backward 到底发生了什么」。numpy 做底层存储。

**模式**：Foundation（这是全 lab 手搓浓度最高的单元）。

| 单元 | 内容 | 验收测试 |
| --- | --- | --- |
| U1.1 前向 | Tensor 包装 numpy：elementwise、matmul、reduction、broadcast、view/transpose/reshape | 随机 shape 与 PyTorch forward 对齐（rtol=1e-6）；非连续 tensor 用例 |
| U1.2 反向 | 计算图构建、拓扑排序 backward、broadcast 梯度归约、梯度累积 | 有限差分 gradcheck；与 PyTorch backward 对齐；二次 backward 不要求 |
| U1.3 模块层 | Module 基类 + Linear/RMSNorm/SwiGLU + SGD | 三个 module 前反向与 PyTorch 对齐；用自己的系统在玩具数据上把一个 2 层 MLP 训到收敛 |
| U1.4 找 bug 演练 | Agent 在实现里注入 3 个 bug（broadcast 归约错、stride 错、inplace 破坏图），我只看失败测试定位 | 三个 bug 全部定位并修复，写出每个 bug 的机制解释 |

**过关标准**：gradcheck 全绿 + MLP 收敛 + bug 演练报告。
**封存**：本 phase 产物独立成档，后续 phase 用 PyTorch，不 import 它。

---

## Phase 1.2 — Qwen-like Transformer from scratch（W2–W3，约 1.5 周）

**模式**：Foundation（attention/RoPE/KV cache 手搓）+ Copilot（其余组件、生成循环脚手架）。

| 单元 | 内容 | 验收测试 |
| --- | --- | --- |
| U2.1 基础组件 | Config dataclass、Embedding、RMSNorm、SwiGLU MLP（PyTorch 重写，允许参考 1.1 的推导） | 与 `torch.nn` / 官方实现数值对齐 |
| U2.2 RoPE + GQA | 旋转位置编码；分组查询注意力；causal mask | 与 `F.scaled_dot_product_attention` 对齐；GQA 在 n_kv_heads=n_heads 时退化为 MHA 的等价性测试；RoPE 的相对位置平移不变性测试 |
| U2.3 组装 | Decoder block 堆叠、weight tying、初始化；**手算**参数量/FLOPs/显存并写成函数 | 手算参数量 == `sum(p.numel())`；FLOPs 估算与 profiler 实测差距 <2x 并解释 |
| U2.4 生成 + KV cache | temperature/top-p 采样；增量解码 KV cache | **cached 与 non-cached 解码 logits 逐位置 allclose**（本 phase 最重要的测试）；greedy 下两者输出序列完全一致 |
| U2.5 权重映射验证 | 转换官方 Qwen 最小尺寸模型权重加载进自己的实现 | 同一 prompt 下与 transformers 官方实现 logits 对齐（BF16 容差内）→ 证明架构理解无偏差 |
| U2.6 过拟合冒烟 | 用 1.0 的 tokenizer + 10–30M 模型 | 32 条样本 loss 降到 ~0；训练曲线无 NaN/spike |

**过关标准**：U2.4 与 U2.5 两个对齐测试是硬门槛，其余全绿。

---

## Phase 1.3 — 训练框架（W4，约 1 周）

**模式**：Copilot（架构、checkpoint 逻辑、failure 处理我写；
dataloader/日志/配置 boilerplate 交 Agent）。

| 单元 | 内容 | 验收测试 |
| --- | --- | --- |
| U3.1 数据管线 | tokenized 语料的 memmap 存储、sequence packing、causal masking、DataLoader | packing 前后每 token loss 对齐；打乱可复现（seed 固定）|
| U3.2 训练循环 | BF16 autocast、grad accumulation、clipping、cosine scheduler + warmup、wandb/本地日志 | grad accum N 步 == 大 batch 单步（数值容差内）；lr 曲线快照测试 |
| U3.3 checkpoint/resume | model/optimizer/scheduler/RNG/step 全量保存；原子写入 | **中断-恢复后 loss 轨迹与不中断基线逐步一致**（本 phase 最重要的测试）|
| U3.4 failure injection | 中途 kill、checkpoint 写一半、数据 NaN、loss spike、optimizer state 丢失 | 每个场景：能检测 → 能报告 → 能恢复或干净失败；写成自动化测试而非手工演示 |
| U3.5 云端 capstone（M→L 级）| 租卡训 ~50–100M，用 1.0 tokenizer + 真实语料 | 产出 HF 格式 checkpoint + 训练报告（loss/MFU/显存/一次真实的中断恢复记录）|

**过关标准**：U3.3/U3.4 全绿 → 才允许上云跑 U3.5（省算力铁律）。

---

## 进度表（过关打勾）

- [ ] 1.0 BPE：U0.1 / U0.2 / U0.3 + postmortem
- [ ] 1.1 Tensor：U1.1 / U1.2 / U1.3 / U1.4 + postmortem
- [ ] 1.2 Transformer：U2.1 / U2.2 / U2.3 / U2.4 / U2.5 / U2.6 + postmortem
- [ ] 1.3 Trainer：U3.1 / U3.2 / U3.3 / U3.4 / U3.5 + postmortem
- [ ] M1 总结：capstone checkpoint 发布 + 模块 review

## 时间风险与裁剪顺序

进度落后时按此顺序砍：U0.3（效率优化）→ U1.4（bug 演练）→
U2.5（权重映射，降级为可选）→ U3.5 从 100M 降到 50M。
U2.4（KV cache 对齐）和 U3.3（恢复一致性）任何情况下不砍。
