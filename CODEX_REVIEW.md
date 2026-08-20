# Codex Review 任务书 — AI Systems Lab 全仓库审计

> 面向：Codex（或任何外部 AI reviewer）。可以开启 subagent、鼓励高并发
> （建议按下方「并发分工」拆分）。产出统一写入根目录 `REVIEW_FINDINGS.md`。

---

## 0. 必读：这个仓库的特殊性（不懂这节会产生大量错误操作）

这是一套**师生模式的课程 lab**（类比 CS336 大作业）：

- `m*/minilm|minimoe|minivllm|minisft|minidpo|minigrpo|minidata|minilaw|kernels/`
  下的文件是**学生答题骨架**：函数体是 `raise NotImplementedError` + 中文引导提示。
  **这是设计状态，不是 bug。绝对不许把实现补进去。**
- `m*/reference/` 下是参考答案（学生卡 30 分钟才看）。参考答案必须能让测试全绿。
- `m*/tests/` 是验收关卡：**骨架状态下全红 = 正确；参考答案覆盖后全绿 = 正确。**
- `m7-agent-engineering/exercises/7.2-review/pr/` 是**故意埋了 18 个已知 bug 的
  演练材料**（答案在 `m7-agent-engineering/reference/7.2-review/ANSWERS.md`）。
  这些 bug 是教具，**不要作为 finding 上报，更不要修**。你对它的审计任务见 §3.7。
- 每个 phase 的规格说明在 `m*/docs/<phase>/SPEC.md`，接口已冻结。

### 修改权限（已授权：审计 + 修复）

**允许修改**（确认属实的 finding 直接修）：

- `m*/reference/` 参考答案的 bug
- `m*/tests/` 的测试缺陷：恒真断言、盲点补测、错误的 oracle、不稳定的容差
- `m*/docs/` 与根目录文档的错误/过期内容、`scripts/`、`benchmarks/` 脚手架
- 学生骨架**仅限**：删除泄题内容（换成引导式问题）、修正错误的提示/注释、
  修正给定脚手架代码（如给定的数据类/工具函数）里的 bug

**禁止修改**（碰了就是事故）：

1. **不许实现学生骨架的任何 TODO**——`raise NotImplementedError` 必须原样保留。
   即使它让测试变红，红就是设计状态。
2. **不许为了让测试通过而放松测试**（扩容差/删断言/加 skip）。测试只能变严
   或修正错误，任何容差调整必须在报告里给出数值依据。
3. **不许修 7.2 的 18 个已埋 bug**（`m7-.../exercises/7.2-review/pr/`）。
   若发现答案卷行号漂移或答案卷未记录的意外 bug，改 `ANSWERS.md` 使其准确，
   代码里的雷保持原样。
4. 不许改各 SPEC 的冻结接口签名（文字勘误可以）。

**每次修改后必须维持的不变量**（改完当场验证，写进报告）：

- 受影响模块：参考答案覆盖 → 全绿；恢复骨架 → 全红（m2 除外：py_compile +
  无 CUDA 全 skip）；期望数值见 §2 的表（若你修复导致合理变化，更新报告说明）。
- 收尾时工作区必须处于**骨架状态**（不是参考答案覆盖状态）。

**Git**：不 commit / push / branch。所有修改留在工作区，出题人（Claude）
会对 diff 做二审后再提交。

其他：不下载大文件（权重/数据集）。已有本地权重：`m1-foundation/data/qwen2.5-0.5b/`。

### 环境事实

- Mac（无 CUDA）：`m2-kernels` 的测试会整体 skip，**只能静态审 Triton 代码**——
  这正是它风险最高的原因（从未在真卡上跑过），静态审它是本次的高价值任务。
- Python 3.12，torch / transformers==4.52.4 / numpy / scipy / regex 已装。
- 各模块测试独立：`cd m<X>-... && python3 -m pytest tests/ -q`。
  全部 CPU；最慢的 m5 约 45s（参考答案覆盖时）。

---

## 1. 审计目标（按价值排序）

1. **测试的正确性与盲点**（最高价值——测试写错比实现写错更隐蔽）：
   恒真断言、oracle 与实现犯同一个错（自己对拍自己）、容差松到测不出错、
   随机不固定 seed、参数组合覆盖缺口、骨架状态下就能通过的空转测试。
2. **参考答案的正确性**：数学/算法错误、边界条件、与 SPEC 冻结接口的偏差。
   历史战绩供校准：此前人工 review 抓到过 BPE tie-breaking 比较对象错误、
   RoPE buffer 持久化导致官方权重加载失败、resume 快进漏乘 grad_accum_steps。
   这个级别的 bug 是你的目标。
3. **骨架泄题**：学生骨架的提示里出现了本该学生自己推导的核心公式/实现。
   判断标准看各 SPEC 的模式声明（Foundation 单元提示只许引导式问题；
   Copilot 脚手架可以给密）。注意：SPEC 里冻结公开的公式不算泄题。
4. **m2-kernels 静态审计**（专项，见 §3.2）。
5. **SPEC ↔ 代码 ↔ 测试三方一致性**：签名、默认值、验收表编号与实际测试对应。
6. **文档一致性**：README/PLAN/LAB_DESIGN 的进度表、路径、测试计数是否与
   实际相符；文档互链是否断链。
7. 低优先：风格一致性（已知 m1 的 1.3 骨架是英文，其余是中文——已记录，
   不必重复上报此条本身，但可以报同类新发现）。

## 2. 验证协议（报 finding 前必须做的）

- **每个 finding 必须带证据**：file:line + 复现命令/最小反例/推导。
  没有证据的猜测标注 `[未验证]` 并单独分区。
- 声称「测试有盲点」时，给出一个**能通过现有测试但错误的实现片段**（思想实验
  级别的具体描述即可，不必真写入仓库）。
- 声称「参考答案有 bug」时，先跑参考答案覆盖验证（下方命令），
  再给出触发错误的最小输入。

### 参考答案覆盖验证命令（按模块）

```bash
# 通用模式（以 m3 为例）：备份 → 覆盖 → 去后缀 → 测试 → 恢复
cd m3-arch-study
mkdir -p /tmp/rv-m3 && cp minimoe/*.py /tmp/rv-m3/
for f in config moe mla mtp model parity; do
  cp reference/3.0-moe/${f}_solution.py minimoe/${f}.py; done
LC_ALL=C sed -i '' 's/_solution//g' minimoe/*.py
python3 -m pytest tests/ -q        # 期望：16 passed
cp /tmp/rv-m3/*.py minimoe/        # 恢复！
```

各模块的映射与期望值：

| 模块 | 覆盖映射 | 参考答案期望 | 骨架期望 |
| --- | --- | --- | --- |
| m1 | reference/1.0-bpe/bpe_solution.py→minilm/tokenizer/bpe.py；1.1 的 {tensor,nn,optim}→minilm/tensor/；1.2 的 {config,model,generate,counting,convert_qwen}→minilm/model/；1.3 的 {data,scheduler,checkpoint,trainer}→minilm/training/（均需 sed 去 `_solution`）| 128 passed（含 T7 需本地 Qwen 权重；non-slow 为 127+1 deselected）| 121 failed + 7 errors |
| m2 | 不可运行，静态审 | —（tests 全 skip）| 1 skipped |
| m3 | 见上方示例 | 16 passed | 16 failed |
| m4 | reference/4.0-minivllm/*_solution.py→minivllm/（sed 去后缀）；4.1 的 speculative_solution.py→minivllm/speculative.py | 40 passed | 40 failed |
| m5 | 5.0 的 {chat,packing,lora}_solution.py→minisft/（sed）；5.1 的 {logprob,rm,dpo}.py→minidpo/（**无后缀直接拷**）；5.2 的 {reward,advantage,loss,rollout}.py→minigrpo/ | 98 passed | 98 failed |
| m6 | 6.0 的 {filters,minhash,contamination}_solution.py→minidata/；6.1 的 {fit,optimal,isoflop}_solution.py→minilaw/（均 sed）| 61 passed | 49 failed + 12 errors |
| m7 | 不覆盖。`python3 -m pytest exercises/7.2-review/pr/test_pr.py -q` | 41 passed（**故意的**）| 同左 |

## 3. 并发分工建议（每个 subagent 一个独立任务）

1. **m1-foundation**（最大，可再拆 1.0/1.1 与 1.2/1.3 两个 agent）：
   四个 phase 的测试盲点 + 参考答案审计。重点：test_transformer 的 KV cache
   对齐是否真能抓 mask off-by-one；test_trainer 的恢复一致性是否真是 bit 级。
2. **m2-kernels 专项**：逐行审三个 Triton kernel 的参考答案与测试。
   重点：指针步长单位、mask 边界、fp32 累加是否贯彻、bf16 store cast、
   `tl.full/tl.max` 对全 mask 块的行为、autograd.Function 的 ctx 保存、
   T4 显存断言的计量方式。给出「租卡验证时最可能爆的前 5 个点」排序。
3. **m3 + m6**：MoE/MLA/MTP 数学审计（bias 只影响选择、MLA scale、
   MTP 梯度路径）；MinHash 无偏性与溢出、闭式最优配比推导复核。
4. **m4**：BlockManager refcount/prefix hash 的正确性；engine 与串行 oracle
   等价性测试的覆盖；speculative 的分布校正（对照 Chen et al. 2023）。
5. **m5**：labels 右移与 mask 边界；packing 块对角 mask 的构造与测试策略；
   DPO/GRPO 公式对照论文（DPO: Rafailov 2023；GRPO: DeepSeekMath 2024；
   k3 估计: Schulman blog）；GRPO T6 收敛测试的稳定性风险。
6. **文档横切**：所有 SPEC 验收表 ↔ 实际测试函数一一对应；README/PLAN/
   LAB_DESIGN/网页(docs/index.html) 的计数与状态是否一致；互链有效性。
7. **m7 材料审计**：ANSWERS.md 的 18 个问题逐个核对行号与真实性
   （每个 bug 按答案卷的复现片段跑一遍）；检查是否存在答案卷**没记录**的
   意外 bug（那才是 finding）；INSTRUCTIONS 的评分规则是否可执行。

## 4. 产出格式（REVIEW_FINDINGS.md）

```markdown
# Review Findings — <日期>
## 汇总
- 审计范围/跑过的验证命令/finding 总数（按严重度）/其中已修复数
## P0（参考答案错误 / 测试无效 / 会误导学生的泄题）
### F-01 <标题>  [FIXED | NOT-FIXED]
- 位置：file:line
- 类别：测试盲点|参考答案bug|泄题|一致性|文档
- 证据：<复现命令与输出 / 最小反例 / 推导>
- 修复：<改了什么（文件+要点）；NOT-FIXED 则写原因（如需真卡/需出题人决策）>
- 修复后验证：<受影响模块的 覆盖→绿 / 骨架→红 实测结果>
## P1（应修，但不阻塞闯关）
## P2（打磨项）
## [未验证] 猜测清单
## 各模块最终状态（覆盖→绿数、恢复→红数 的实测输出摘要）
```

## 5. 已知问题（不必重复上报）

- m1 Phase 1.3 骨架注释为英文，与其他 phase 中文风格不一致（已记录待改）。
- m2 整体未经真卡验证（已在 SPEC 与 README 声明，租卡时跑 validate_reference.sh）。
- m6 6.1 T2 的 E 容差 0.2（可识别性问题，SPEC 已注明）。
- m1 PLAN.md 进度表行 `1.0 BPE：U0.1 / U0.2 / U0.3` 漏列 U0.4。
- 学生骨架全红、7.2 的 41 绿、GRPO 测试 149s 内的耗时——均为设计如此。
```
