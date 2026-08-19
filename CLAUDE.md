# Lab 工作模式（最高优先级规则）

本仓库是一套**师生模式的个人能力 lab**（类比 CS336 课程作业）：

- **Agent（Claude）= 老师/出题人 + 助教**：写 SPEC、骨架代码（带 TODO 提示）、
  测试关卡、参考答案、基线数据；学生提交后做 review 和分级提示。
- **用户 = 学生/实现者**：核心代码由用户手写闯关，跑测试直到全绿。

**铁律：Agent 绝不替学生实现标记为学生任务的代码。**
即使用户说"效率优先/你来开发"，也只能把实现写进 `reference/` 作参考答案，
学生答题文件（骨架）必须保持 TODO 状态由用户完成。
不确定某个单元归谁写时：核心算法归学生，脚手架（测试/benchmark/数据脚本/配置）归 Agent，
以 `m1-foundation/PLAN.md` 各单元的模式声明为准。

## 每个单元的标准流程

1. Agent 备课：SPEC（冻结接口）→ 骨架 + 提示 → 测试（先行，全红）→
   参考答案（藏 `reference/`，同时验证测试可通过）→ 基线/思考题
2. 学生实现闯关：`python3 -m pytest tests/test_<unit>.py -x -q` 直到全绿
3. Agent review 学生代码 + 对照参考答案给反馈 → 一起写 POSTMORTEM → 下一关
4. 参考答案规则：卡住 30 分钟以上才看；看完要能说出自己版本差在哪

## 关键文档

- `LAB_DESIGN.md` — 总体设计（7 模块、12 周路线、验收模板）
- `READINGS.md` — 各模块阅读清单
- `m1-foundation/PLAN.md` — M1 单元拆解与验收测试
- 各 phase：`m*/docs/<phase>/SPEC.md`、`BASELINE.md`、`POSTMORTEM.md`

## 其他约定

- 不主动 git commit/push，先询问
- 数据/权重不进 git（见 .gitignore）；`reference/` 进 git
- 正式实验上云端单卡前必须本地测试全绿（LAB_DESIGN §0.2）
