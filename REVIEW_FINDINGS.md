# Review Findings — 2026-08-20

## 汇总

- 审计范围：M1–M7 的参考答案、验收测试、学生骨架提示、实验脚手架、14 份 SPEC、根文档与网页。
- 方法：按 `CODEX_REVIEW.md` §3 分两批并行审计；所有结论均有代码推导、最小反例或实测输出。
- Finding：35 项（P0 20、P1 13、P2 2）；34 项已修复。另 1 项是 M7 教具中答案卷未记录的意外缺陷，按 §0 只补记答案卷、不修埋雷代码。
- 修改均未 commit；最终工作区为学生骨架状态。默认 `python3` 指向无 pytest 的 Python 3.11，因此验证统一使用 `/opt/anaconda3/bin/python3`（Python 3.12.2）。

## P0（参考答案错误 / 测试无效 / 会误导学生的泄题）

### F-01 GPT-2 regex 把数字错误切成三位一组 [FIXED]

- 位置：`m1-foundation/reference/1.0-bpe/bpe_solution.py:26`、`minilm/tokenizer/bpe.py:40`、`tests/test_bpe.py:121`
- 类别：参考答案 bug / 测试盲点
- 证据：transformers 4.52.4 的 GPT-2 pattern 为 ` ?\p{N}+`，原仓库为 `\p{N}{1,3}`。语料 `1234` 重复 50 次且只做一次 merge 时，正确规则选并列最大 pair `34`，旧规则因切成 `123|4` 选 `23`；原 HF alignment 只测 4 个英文单词。
- 修复：参考答案和给定 tokenizer pattern 改为连续数字；新增数字公共接口回归。
- 修复后验证：M1 最终 128 项中，参考覆盖 `127 passed, 1 deselected`（唯一 deselected 为未改动 slow overfit）；骨架 121 failed + 7 errors。

### F-02 同一 Tensor 图重复 backward 会传播陈旧中间梯度 [FIXED]

- 位置：`m1-foundation/reference/1.1-tensor/tensor_solution.py:109-141`、`tests/test_tensor.py:371`
- 类别：参考答案 bug / 测试盲点
- 证据：`r=(a*a+a).sum(); r.backward(); r.backward()` 原实现得到 3 倍而非 2 倍梯度；旧测试明确重建计算图，规避了缺陷。
- 修复：每次反传前清空非叶节点 grad、保留叶节点累积；补同图重复反传及 `requires_grad=False` 根节点回归。
- 修复后验证：同 F-01。

### F-03 KV-cache 测试漏掉多 token suffix mask [FIXED]

- 位置：`m1-foundation/tests/test_transformer.py:387`
- 类别：测试盲点
- 证据：原测试只覆盖 prefill + 单 token decode；“cache 非空且 `T_new>1` 时让每个 query 看完整 suffix”的错误 mask 可全过。
- 修复：prefix=3 后一次追加 suffix=4，逐位置比较 cached 与 full logits。
- 修复后验证：参考答案通过新增回归；骨架保持全红。

### F-04 Trainer 的“bit 级恢复”只比较近似 loss [FIXED]

- 位置：`m1-foundation/tests/test_trainer.py:424-500`
- 类别：测试无效
- 证据：原断言仅 `abs(loss_a-loss_b)<1e-5`；参数或 optimizer state 出现小漂移仍能通过。
- 修复：逐步 loss 精确相等，最终 model tensor 用 `torch.equal`，递归逐位比较 optimizer state；保留 grad accumulation 与跨 epoch 场景。
- 修复后验证：参考答案严格通过；骨架全红。

### F-05 Foundation 骨架直接给出核心导数与更新公式 [FIXED]

- 位置：`m1-foundation/minilm/tensor/tensor.py:176`、`tensor/optim.py:5`、`model/model.py:32`
- 类别：泄题
- 证据：骨架直接列出 div/pow/exp/log/sigmoid/maximum 导数、momentum 完整更新式、RoPE/RMSNorm 实现式；这些不属于 SPEC 冻结公开公式。
- 修复：替换为推导、shape 与有限差分引导问题；所有 `NotImplementedError` 原样保留。
- 修复后验证：骨架共 128 项全红。

### F-06 CUDA 显存测试把上一轮 autograd 图算进 fused 峰值 [FIXED]

- 位置：`m2-kernels/tests/test_kernels.py:289-325`、`docs/2.0-kernels/SPEC.md:67`
- 类别：测试无效
- 证据：旧测试保留 baseline logits、reference logits 和 grad，仅 `empty_cache()`；活动 tensor 不会释放。`4096×32768×4` 单张量即 512 MiB，峰值比例不是两种实现的可比量。
- 修复：两轮独立输入，记录分配输入后的 baseline，比较增量峰值；轮次间删除完整图、GC、清 cache；SPEC 明确“增量峰值”。
- 修复后验证：静态编译通过；无 CUDA 为 1 skipped。

### F-07 CE 的 bf16 参数化用例实际在 kernel 前升级成 fp32 [FIXED]

- 位置：`m2-kernels/tests/test_kernels.py:231-268`
- 类别：测试无效
- 证据：原 `logits.clone().float()` 使 `torch.bfloat16` 从未进入 Triton kernel；只正确实现 fp32 的错误 kernel 可通过。
- 修复：kernel 输入保留参数 dtype，oracle 从同一量化输入转 fp32；检查 loss dtype，并真正覆盖默认 `ignore_index=-100`。
- 修复后验证：py_compile 通过；无 CUDA 为 1 skipped。

### F-08 CE 全部 target 被 ignore 时 backward 除零 [FIXED]

- 位置：`m2-kernels/reference/2.0-kernels/cross_entropy_solution.py:124-150`、`tests/test_kernels.py:271-284`
- 类别：参考答案 bug
- 证据：`logits=(8,1024)` 且 targets 全为 `-100`；PyTorch 返回 loss=NaN、梯度全 0，原 backward 的标量除法会除零。
- 修复：forward 保留 `0/0→NaN` 语义，backward 在 `N_valid==0` 时显式零缩放；补回归。
- 修复后验证：同 F-06。

### F-09 标准 MHA cache 闭式把 K+V 重复乘 2 [FIXED]

- 位置：`m3-arch-study/minimoe/mla.py:51-55`、`reference/3.0-moe/mla_solution.py:89-93`、`tests/test_moe.py:223-234`
- 类别：参考答案 bug / 错误 oracle
- 证据：`qk_nope + qk_rope + v_head` 已是 K 与 V 维度之和，旧函数再乘 2。测试配置正确值 10,240 bytes，旧值 20,480；MLA/MHA 比例被从 25% 错报为 12.5%。
- 修复：删除额外乘 2；在原 T5 内加入独立闭式 oracle，避免测试与实现共错。
- 修复后验证：参考 16 passed；骨架 16 failed。

### F-10 MoE 均衡测试忽略零负载 expert，完全塌缩可伪装成 1.0 [FIXED]

- 位置：`m3-arch-study/tests/test_moe.py:140-152`
- 类别：测试盲点
- 证据：旧测试先 `load=load[load>0]`；错误 `update_bias` 若把所有 token 压到单一 expert，`max/min` 仍为 1。
- 修复：收敛测试先断言每个 expert 负载大于 0，再计算完整负载比。
- 修复后验证：参考 16 passed；骨架 16 failed。

### F-11 完整 prompt 命中 prefix cache 时 prefill 收到空输入 [FIXED]

- 位置：`m4-inference/reference/4.0-minivllm/cache_adapter_solution.py:73-79`、`engine_solution.py:147-159`
- 类别：参考答案 bug / 测试盲点
- 证据：block_size=4，先跑 `[1..8]` 再提交相同 prompt；旧实现得到 `input_ids=[]`，reshape 0 元素时报错。
- 修复：prefill cache 最多复用 `num_tokens()-1`，保留最后一个 prompt token 产生 next-token logits；同步脚手架提示并补回归。
- 修复后验证：参考 40 passed；骨架 40 failed。

### F-12 prefill 未预留首个 decode slot，完成请求仍申请 block [FIXED]

- 位置：`m4-inference/reference/4.0-minivllm/block_manager_solution.py:106-138`、`engine_solution.py:76-103`
- 类别：参考答案 bug
- 证据：`num_blocks=1, block_size=4, prompt=[1,2,3,4], max_new_tokens=1`；生成唯一输出后旧代码先 `append_slot` 再判断结束，错误报无空 block。
- 修复：准入和分配按 `prompt_len+1` 预留；先判断 EOS/长度结束，仅继续运行时 append；补边界与过度准入回归。
- 修复后验证：参考 40 passed；骨架 40 failed。

### F-13 SFT 训练脚本完全没使用 packing attention mask [FIXED]

- 位置：`m5-post-training/scripts/train_sft.py:160-165`
- 类别：脚手架 bug / 测试盲点
- 证据：`pack_examples` 正确构造块对角 mask，但训练只调用 `model(input_ids=ids_t)`；改变 packed doc A 会影响 doc B logits。
- 修复：bool `(1,S,S)` mask 转 `(1,1,S,S)` additive mask 并传给模型。
- 修复后验证：参考全套 98 passed；骨架 98 failed。

### F-14 DPO sequence_logprob 把右 padding 计入 response [FIXED]

- 位置：`m5-post-training/reference/5.1-dpo/logprob.py:18-34`、`tests/test_dpo.py:102-111`、`scripts/train_dpo.py:57-67`
- 类别：参考答案 bug / 测试盲点
- 证据：旧 mask 只有 `pos>=prompt_len`；给相同序列右侧追加 pad=258 会改变 sequence log-prob，违反 SPEC。
- 修复：从 `model.config.pad_token_id` 构造 attention mask，并从 response mask 排除 pad；将不变量并入 T1。
- 修复后验证：参考 98 passed；骨架 98 failed。

### F-15 DPO reference“不变”断言与自身比较且含恒真表达式 [FIXED]

- 位置：`m5-post-training/tests/test_dpo.py:385`、`:441-445`
- 类别：测试无效
- 证据：训练后才创建 `ref_snapshot` 并立即与自身比较；`torch.equal(...) is False or True` 对任何结果恒真。错误实现可在每步修改 ref_model 而通过。
- 修复：训练前 clone reference 参数，训练后逐位与初始快照比较，删除恒真断言。
- 修复后验证：参考 98 passed；骨架 98 failed。

### F-16 GRPO reference 在 SFT 预热前快照，KL 锚错策略 [FIXED]

- 位置：`m5-post-training/scripts/train_grpo.py:218-227`、`tests/test_grpo.py:546-558`
- 类别：脚手架 bug / 错误 oracle
- 证据：旧脚本先复制随机模型为 reference，再 SFT policy。修正 reference 但保留 100 步预热时，实测 reward delta 仅 0.1313，说明旧 T6 通过依赖错误锚点。
- 修复：SFT 后快照 reference；预热改为实测稳定且保留 RL 学习空间的 35 步，未放松 `delta>0.3`、60 轮与 NaN 断言。
- 修复后验证：T6 单测通过；参考全套 98 passed；骨架 98 failed。

### F-17 MinHash 空签名让任意短文档互相“完全重复” [FIXED]

- 位置：`m6-data-scaling/reference/6.0-data/minhash_solution.py:127-151`、`tests/test_data.py:455-460`
- 类别：参考答案 bug / 测试盲点
- 证据：旧实现对 `['alpha','beta']`、`['','different']` 均返回 `kept=[0]`，因为空 shingle 的全 MAX 哨兵签名逐位相等。
- 修复：空 shingle 文档不进入 LSH 近重复判定；SPEC 与骨架提示澄清哨兵语义；补短文档回归。
- 修复后验证：参考 61 passed；骨架 49 failed + 12 errors。

### F-18 isoFLOP 近线性数据可外推出 N_opt=inf 和巨大负 loss [FIXED]

- 位置：`m6-data-scaling/reference/6.1-scaling/isoflop_solution.py:39-52`、`tests/test_scaling.py:271-279`
- 类别：参考答案 bug / 数值稳定性
- 证据：三点 `(N,L)=(1e6,2.2),(1e7,2.1),(1e8,2.0)` 因 polyfit 数值噪声得到极小正曲率；旧实现输出 `N_opt=inf, L_min≈-3.97e13`。
- 修复：只接受开口向上且顶点落在观测 `log N` 区间的谷底，否则退化到实测最小端点；补回归。
- 修复后验证：参考 61 passed；骨架 49 failed + 12 errors。

### F-19 Foundation isoFLOP 骨架直接给出二次式顶点公式 [FIXED]

- 位置：`m6-data-scaling/minilaw/isoflop.py:10-15`
- 类别：泄题
- 证据：Foundation 模式要求学生手推，旧骨架直接给 `x_opt=-b/(2a)` 与 `N_opt=exp(x_opt)`，SPEC 未公开该具体公式。
- 修复：替换为关于导数、开口方向、空间变换和退化策略的引导问题；保留 `NotImplementedError`。
- 修复后验证：骨架仍全红。

### F-20 M7 假 PR 用 42 处 `BUG #n` 注释直接公布答案 [FIXED]

- 位置：`m7-agent-engineering/exercises/7.2-review/pr/{sampling,kv_utils,metrics,bench,test_pr}.py`
- 类别：泄题
- 证据：`rg 'BUG #|BUG NOTE'` 原命中 42 处，包含 bug 编号、机制和正确修法，45 分钟 review 练习无需审查即可抄满 18 项。
- 修复：只清理/改写泄题注释和输出文案，不改变 18 个埋雷的执行逻辑；同步答案卷行号。
- 修复后验证：`rg` 0 命中；教具仍 41 passed。

## P1（应修，但不阻塞闯关）

### F-21 Tensor reshape/transpose 声称 view 却无条件复制 [FIXED]

- 位置：`m1-foundation/reference/1.1-tensor/tensor_solution.py:77-82`、`tests/test_tensor.py:438`
- 类别：参考答案 bug
- 证据：旧构造器 `astype(np.float64)` 默认 copy，`np.shares_memory` 为 False。
- 修复：使用 `copy=False`，新增 reshape/transpose 共享存储断言。
- 修复后验证：同 F-01。

### F-22 generate(max_new_tokens=0,use_cache=True) 仍生成一个 token [FIXED]

- 位置：`m1-foundation/reference/1.2-transformer/generate_solution.py:45`、`tests/test_transformer.py:490`
- 类别：参考答案 bug
- 证据：最小反例即 `max_new_tokens=0`，旧 cache 路径仍进入首次 decode。
- 修复：立即返回输入 clone，覆盖 cache on/off。
- 修复后验证：同 F-01。

### F-23 load_qwen 缺少冻结接口中的 device/dtype [FIXED]

- 位置：`m1-foundation/reference/1.2-transformer/convert_qwen_solution.py:32-61`、`minilm/model/convert_qwen.py:37`
- 类别：一致性
- 证据：旧实现不能调用 `load_qwen(path, device='cpu', dtype=torch.float16)`。
- 修复：补默认参数，构模后 `.to(device,dtype)`，本地 Qwen T7 校验签名。
- 修复后验证：本地权重测试通过；骨架全红。

### F-24 RMSNorm wrapper 忽略非连续 weight 的元素 stride [FIXED]

- 位置：`m2-kernels/reference/2.0-kernels/rmsnorm_solution.py:121-165`、`kernels/rmsnorm.py:164-212`
- 类别：参考答案 bug / 脚手架 bug
- 证据：`weight=storage[::2]` 仍为 `(H,)`，但 `W_ptr+offsets` 按连续元素读取相邻 storage。
- 修复：wrapper 将 weight contiguous，并把该 tensor 存入 ctx；补回归，不实现学生 kernel TODO。
- 修复后验证：静态编译通过；无 CUDA 1 skipped。

### F-25 参考 SwiGLU 缺失 shape 校验，可能越界读取 [FIXED]

- 位置：`m2-kernels/reference/2.0-kernels/swiglu_solution.py:86-100`、`tests/test_kernels.py:210-214`
- 类别：参考答案 bug
- 证据：gate `(2,8)`、up `(2,7)` 时 grid 按 gate.numel 启动并读取较短 up；学生给定 wrapper 反而已有 assert。
- 修复：参考答案补相同 shape assert 及 mismatch 回归。
- 修复后验证：同 F-24。

### F-26 speculative acceptance 的闭区间与 epsilon 改变目标分布 [FIXED]

- 位置：`m4-inference/reference/4.1-speculative/speculative_solution.py:145-173`
- 类别：参考答案 bug
- 证据：令 draft token 在 target 下概率为 0、mock `torch.rand=0`，旧 `u<=alpha` 会接受不可能 token；`p/(q+1e-12)` 还会压低小 q 的接受率。
- 修复：判据改 `u<alpha`，采样 token 的正 q 直接算 `p/q`，仅残差和确实 `<=0` 时回退；补 RNG 端点回归。
- 修复后验证：参考 40 passed；骨架 40 failed。

### F-27 k3 在 policy≈reference 时发生消去并出现负值 [FIXED]

- 位置：`m5-post-training/reference/5.2-grpo/loss.py:25-29`、`tests/test_grpo.py:365-384`
- 类别：参考答案 bug / 数值稳定性
- 证据：float32 直接算 `exp(x)-x-1`，`x∈[-1e-3,1e-3]` 有 406 个负值，最小 `-5.9604645e-08`。
- 修复：用数学等价的 `torch.expm1(x)-x`；加入近零非负回归。
- 修复后验证：参考 98 passed；骨架 98 failed。

### F-28 dedup 跳过已删除候选，重复链不会归成一个簇 [FIXED]

- 位置：`m6-data-scaling/reference/6.0-data/minhash_solution.py:142-150`、`tests/test_data.py:462-473`
- 类别：参考答案 bug / 测试盲点
- 证据：构造 A~B、B~C 均约 0.82，而 A~C≈0.67；旧实现先删 B，随后跳过 B，错误保留 C，得到 `[0,2]`。
- 修复：仍用所有早期候选建立重复边，重复链只保留最早成员；补确定性链式回归。
- 修复后验证：参考 61 passed；骨架 49 failed + 12 errors。

### F-29 全局文档测试计数长期漂移 [FIXED]

- 位置：`README.md:21-31`、`docs/index.html:218-334`
- 类别：文档
- 证据：网页 aria 仍写 61、总数 330、M2 显示 5 skips，均与实际收集不符。
- 修复：同步 M1=128、M2=1 skipped、M3=16、M4=40、M5=98、M6=61；学生红灯总数 343，M7 41 passed 明确不计入。
- 修复后验证：逐模块 collect 与实测摘要一致。

### F-30 M1 PLAN 的 U3.1 验收内容已不对应冻结测试 [FIXED]

- 位置：`m1-foundation/PLAN.md:115`
- 类别：文档 / 一致性
- 证据：旧文档写 sequence packing/packing loss，实际 SPEC/tests 验收 memmap roundtrip、PackedDataset 右移/边界与 seed 重现。
- 修复：按冻结验收项改写；同时修复 §5 已知的 U0.4 漏列（不重复计 finding）。
- 修复后验证：14 份 SPEC 的每个 T 编号均有对应测试组。

### F-31 LAB_DESIGN 把恢复测试挂到 Transformer phase 且错写 phase 名 [FIXED]

- 位置：`LAB_DESIGN.md:113`、`:387`
- 类别：文档 / 一致性
- 证据：1.2 下误列训练中断恢复；另一处把 Phase 1.0 写成 Tensor。
- 修复：1.2 改为 cache、参数量、Qwen logits 硬门槛；Phase 1.0 改 BPE Tokenizer。
- 修复后验证：与 SPEC/tests 逐项核对一致。

### F-32 M7 答案卷多处机制、概率与复现不准确 [FIXED]

- 位置：`m7-agent-engineering/reference/7.2-review/ANSWERS.md`
- 类别：教具一致性
- 证据：包括 H=3 的 perplexity 倍率误写 8x（实为 3.77x）、stochastic 失败概率误写、top-p 测试用过滤后 softmax（恒重归一到 1）、mutable-default 复现用默认 penalty=1 导致提前返回、把 batched topk 误称 512x/改变复杂度、把单层 zero-fill 误述成 32 层 64 次分配等。
- 修复：逐项重跑最小复现，修正文案、概率、真实异常、正确测试方法和行号；不修改 18 个故意 bug。
- 修复后验证：教具 41 passed。

### F-33 M7 batch_prefill_cache 还有答案卷未记录的输入/device/dtype 缺陷 [NOT-FIXED]

- 位置：`m7-agent-engineering/exercises/7.2-review/pr/kv_utils.py:204-219`、`reference/7.2-review/ANSWERS.md` 的 accidental issue 附录
- 类别：意外 bug / 教具一致性
- 证据：keys 长度 2、values 长度 1 时静默返回第二行全零；float32 key + float64 value 会把 V 静默降成 float32；CUDA 输入因 `torch.zeros` 默认建在 CPU 而设备错误。这与已知 Bug 14 的性能问题不同。
- 修复：按 §0 禁止修改埋雷代码；已在答案卷新增“不计入 18 分”的 accidental issue、最小反例与处置建议。
- 修复后验证：pr 逻辑未改，41 passed；需要出题人决定将其纳入、替换或显式排除。

## P2（打磨项）

### F-34 Scaling SPEC 的指数边界与实现、测试注释不一致 [FIXED]

- 位置：`m6-data-scaling/docs/6.1-scaling/SPEC.md:46`、`minilaw/fit.py:64`、`tests/test_scaling.py:85`
- 类别：一致性
- 证据：实现与骨架使用 alpha/beta 下界 0.01，SPEC 写 0；测试标题写 E 误差 0.1，实际断言与 SPEC 为 0.2。
- 修复：文字统一为 `[0.01,2.5]` 与 E 误差 0.2，未改冻结签名。
- 修复后验证：参考 61 passed。

### F-35 M7 INSTRUCTIONS 指向不存在的 POSTMORTEM [FIXED]

- 位置：`m7-agent-engineering/exercises/7.2-review/INSTRUCTIONS.md:106-107`
- 类别：文档
- 证据：仓库不存在 `docs/7.2/POSTMORTEM.md` 或所称模板，课后步骤不可执行。
- 修复：改为填写现存 `review_template.md` 的 Self-assessment。
- 修复后验证：本地链接目标存在；教具 41 passed。

## [未验证] 猜测清单

- 无。M2 真卡 codegen、显存比例与性能仍受环境限制，但这是任务书已知环境事实，不作为新 finding。
- 租卡优先验证顺序：① CE T4 增量峰值与 16 GiB 压力；② SM75 bf16 load/store；③ CE `V=32768` JIT/寄存器压力；④ online logsumexp 在真卡近似数学下的 `1e-5` loss 容差；⑤ RMSNorm H=4096 backward 资源占用。

## 各模块最终状态

| 模块 | 参考答案 / 专项验证 | 恢复骨架 |
| --- | --- | --- |
| M1 | 修改前完整 122 passed（含 slow/Qwen，436.38s）；修改后最终 non-slow 127 passed、1 deselected（32.53s） | 121 failed + 7 errors（128/128 全红） |
| M2 | `py_compile` 通过；无 CUDA `1 skipped` | 6 个学生 `NotImplementedError` 保留 |
| M3 | 16 passed | 16 failed |
| M4 | 40 passed | 40 failed |
| M5 | 98 passed、2 warnings（33.73s） | 98 failed、2 warnings |
| M6 | 61 passed | 49 failed + 12 errors |
| M7 | 41 passed；18 个故意 bug 保持原样 | 不适用 |

最终检查：`git diff --check` 通过；学生包无 `_solution` import 残留；M1–M6 的 `NotImplementedError` 均保留；工作区所有修改未提交。
