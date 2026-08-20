# Codex 开发任务书 — 六个新 phase 的课件建造

> 面向：Codex。可以开启 subagent、鼓励高并发（六个任务相互独立，见 §4 分工）。
> 你的角色是**助教/建造者**：按已冻结的 SPEC 建骨架、测试、参考答案并自验证。
> SPEC 由出题人（Claude）撰写且已冻结——**接口签名和验收表不许改**；
> 发现 SPEC 有不可实现的缺陷时停下该项，写进报告由出题人裁决，不要自行变通。

## 0. 必读：仓库规则（与 CODEX_REVIEW.md §0 相同的红线）

师生模式课程 lab。每个 phase 交付四件套：

1. **学生骨架**（`m*/<包名>/*.py`）：冻结接口的空实现——docstring +
   `raise NotImplementedError` + 中文引导提示。**绝不写入任何实现逻辑。**
   提示密度按 SPEC 头部的模式声明：Foundation = 只许引导式问题（不给公式、
   不给伪代码）；Lead 倾向 = 可给较详细的契约与结构提示（仍不给逐行实现）。
   SPEC 里公开冻结的公式可以引用。
2. **测试**（`m*/tests/test_<phase>.py`）：按 SPEC 验收表逐条落地，每个 T 至少
   一个测试函数（可拆多个）。骨架状态必须全红（m2 例外：无 CUDA 全 skip）。
   随机必须固定 seed；oracle 不许调用被测实现（独立闭式/手算/torch 对拍）；
   禁止恒真断言；容差要有数值依据（写在断言旁注释）。
3. **参考答案**（`m*/reference/<phase>/<模块名>_solution.py`）：能让测试全绿的
   完整实现，import 用相对路径或自包含，保证「拷贝去 `_solution` 后缀」即可覆盖。
4. **给定脚手架**（SPEC 标注「给定」的类/函数/脚本）：写完整实现，
   放学生包里（这不算替学生答题——SPEC 已声明归属）。

**红线**（碰了就是事故）：不填学生 TODO；不放松/绕过验收表；不改冻结签名；
不动本任务书范围外的既有文件（README/index.html 等计数文档由出题人事后统一改）；
不 git commit/push/branch；不下载权重/数据集。

## 1. 环境事实

- Mac 无 CUDA。m2 的两个任务只能：`py_compile` + 测试 CUDA-skip 结构自查 +
  逐行静态推演 kernel 正确性（把推演写进 BUILD_REPORT）。
- 测试用 `/opt/anaconda3/bin/python3 -m pytest`（Python 3.12，torch/transformers
  /numpy/scipy 已装；Triton 不可用也没关系，import 放 CUDA guard 后面，
  参照 m2 2.0 现行写法）。
- 每个任务的自验证协议：参考答案覆盖 → 全绿；恢复骨架 → 全红；
  **收尾必须处于骨架状态**。每个任务在报告里粘两种状态的 pytest 输出。

## 2. 任务清单（SPEC 均已冻结，先通读再动工）

| # | 任务 | SPEC | 交付位置 | 验证方式 |
| --- | --- | --- | --- | --- |
| A | 4.2 PD 分离模拟 | `m4-inference/docs/4.2-pd-sim/SPEC.md` | `m4-inference/minipd/` + `tests/test_pd.py` + `reference/4.2-pd-sim/` + `benchmarks/bench_pd.py` | CPU 全闭环 |
| B | 3.1 DSv4 机制 | `m3-arch-study/docs/3.1-dsv4/SPEC.md` | `m3-arch-study/minidsv4/` + `tests/test_dsv4.py` + `reference/3.1-dsv4/` + `scripts/ablate_dsv4.py` | CPU 全闭环 |
| C | 3.2 长上下文 | `m3-arch-study/docs/3.2-longctx/SPEC.md` | `m3-arch-study/minikda/` + `tests/test_longctx.py` + `reference/3.2-longctx/` + `scripts/ablate_longctx.py` | CPU 全闭环（T8 < 60s）|
| D | 2.1 进阶 kernel | `m2-kernels/docs/2.1-advanced/SPEC.md` | `m2-kernels/kernels/{rope,block_attn}.py` + `tests/test_kernels_21.py` + `reference/2.1-advanced/` + 扩展 `scripts/validate_reference.sh` | 静态（见 §1）|
| E | 2.2 Grouped GEMM | `m2-kernels/docs/2.2-grouped-gemm/SPEC.md` | 同上模式，`test_kernels_22.py`、`reference/2.2-grouped-gemm/` | 静态 |
| F | 7.2b + 7.2c 演练 | `m7-agent-engineering/docs/REVIEW_DRILLS_BRIEF.md` | 按 brief 的目录契约 | pr 测试全绿 + 复现实跑 + 泄题扫描 + 盲测 |

注意跨任务契约：B 的 toy 模型与 3.0 的 `minimoe` 风格对齐；C 的 `state_bytes`
与 3.0 `mha_cache_bytes` 同口径；E 的 T4 要 import m3 参考实现对拍（路径处理
写清楚）；D/E 的 bf16 容差推导参照 2.0 的 F-07 先例（REVIEW_FINDINGS.md 可查）。

## 3. 质量清单（每个任务收尾自查）

- [ ] 骨架无实现残留、无泄题（Foundation 单元逐条提示自问：学生照抄能写出来吗？
      能 → 重写成问题）；`rg 'BUG|TODO: 实现已给'` 类自查
- [ ] 测试红/绿两态实跑输出已存档；无恒真断言（把参考答案故意改错一处，
      确认测试能抓到，再改回来——每个 T 至少做一次这种变异自检）
- [ ] 参考答案 docstring 含关键推导注释（它是答案卷，要能教人）
- [ ] SPEC 验收表编号 ↔ 测试函数名可对应（docstring 标注 T 编号）
- [ ] 新文件不碰既有测试的收集（pytest 各模块计数变化只来自新增文件）

## 4. 并发分工建议

六个任务完全独立，六路并行。B、C 数学密度高建议各配独立 subagent 先写
参考答案再倒推骨架；F 的两个演练可再拆两路。A 最简单可先跑通全流程当样板。

## 5. 产出

- 全部代码（未 commit，留工作区）
- 根目录 `BUILD_REPORT.md`：每任务的红/绿实测输出、变异自检记录、
  静态推演（D/E）、盲测结果（F）、遇到的 SPEC 疑义与自行决策清单
