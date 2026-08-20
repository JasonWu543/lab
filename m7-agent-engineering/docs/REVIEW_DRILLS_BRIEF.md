# 材料规格 — 7.2 系列新增 Review 演练（7.2b / 7.2c）

> 状态：FROZEN（材料契约）
> 性质：同 7.2 —— 故意埋雷的假 PR 教具，学生限时 45 分钟 review。
> 这不是学生答题骨架，没有 TODO；埋的 bug 本身是教具，测试全绿是设计目标。

## 通用要求（两个演练相同）

- 目录结构对齐 7.2：`exercises/<name>/{INSTRUCTIONS.md, PR_DESCRIPTION.md,
  review_template.md, pr/*.py}` + `reference/<name>/ANSWERS.md`。
- **15–18 个埋雷**，严重度分布约 P0×4 / P1×8 / P2×5，类别覆盖：
  数值/数学错误、边界条件、静默错误吞掉、可变默认参数或 in-place 污染、
  性能陷阱、测试自身缺陷（弱断言/漏 seed/oracle 与实现共错）——
  其中「PR 自带测试全绿但掩盖 bug」至少 4 项。
- pr/ 自带 pytest **必须全绿**（这是陷阱的一部分），CPU < 10s。
- **零泄题**：`rg -i 'bug|fixme|hack|wrong|broken|leak|注意|故意'` 在 pr/ 下
  0 命中（docstring 用正常工程口吻撒谎，如 7.2 现状）；测试名/注释不得暗示
  缺陷机制（7.2 曾因 `BUG #n` 注释翻车，引以为戒）。
- ANSWERS.md 按 7.2 现行格式：逐项 file:line、机制解释、最小复现片段、
  正确修法、severity；结尾 summary 表。**每个复现片段必须实际跑过**。
- INSTRUCTIONS.md：45 分钟规程 + 评分表（对齐 7.2 现行版），并注明
  「发现答案卷之外的真实缺陷可获口头加分但不计入总分」。

## 7.2b — 训练循环 PR（`exercises/7.2b-review-training/`）

假 PR 主题：「给 trainer 加 mixed precision + grad accumulation + cosine 调度
+ 断点续训」。文件建议：`trainer.py, scheduler.py, checkpoint.py, test_train.py`。
埋雷素材方向（builder 自选组合，不必全用）：
- grad accum 平均时机错误（clip 在 unscale 前 / loss 除 accum 的位置错）
- AMP 下 GradScaler 的 step/update 顺序错、fp16 里算 loss 累计
- cosine 调度 off-by-one（warmup 边界、按 epoch 还是按 step）
- resume 漏恢复 scheduler/scaler state、快进不乘 accum（对，就是我们自己
  犯过的那个——见 m1 1.3 的历史，学生应该能抓出来）
- 测试用 1 step 断言「resume 一致」形同虚设、固定输入导致梯度恒零

## 7.2c — 数据管线 PR（`exercises/7.2c-review-data/`）

假 PR 主题：「给数据管线加 tokenize + pack + shuffle + train/val 切分」。
文件建议：`packing.py, splitting.py, sampling.py, test_data.py`。
埋雷素材方向：
- **时序/泄露类至少 3 项**（对齐本 lab 的量化背景）：val 集统计量用全量数据算
  （归一化泄露）、按样本 shuffle 后再切分时序数据、packing 跨文档 attention
  无 mask
- labels 右移 off-by-one、pad 计入 loss、EOS 处理不一致
- shuffle 无固定 seed 但测试比较两次输出（测试恰好没测到）
- buffer 复用导致 batch 间数据污染（in-place）

## 验证协议（builder 自验，写进 BUILD_REPORT）

1. `pytest exercises/<name>/pr/ -q` 全绿；
2. ANSWERS.md 每项复现片段实跑并粘输出；
3. 泄题扫描命令输出 0 命中；
4. 用「不知情的 subagent 扮演学生」盲测 45 分钟规程一次，报告其命中数
   （期望 6–12 项命中；全中说明太明显，<4 说明太隐蔽，需调难度）。
