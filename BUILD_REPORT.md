# 六个新 phase 建造报告

日期：2026-08-20  
环境：macOS / Python 3.12（`/opt/anaconda3/bin/python3`）/ 无 CUDA  
状态：所有交付均留在工作区，未 commit；A–E 收尾均为学生骨架状态。

## 总览

| 任务 | 参考态 | 骨架态 | 结论 |
| --- | --- | --- | --- |
| A — 4.2 PD 分离模拟 | 9 passed | 9 failed | 完成 |
| B — 3.1 DSv4 | 8 passed, 1 failed | 9 failed | 冻结 SPEC 的 T6 数值门槛不可满足，待裁决 |
| C — 3.2 长上下文 | 16 passed | 16 failed | 完成 |
| D — 2.1 进阶 kernel | 无 CUDA，模块 skip | 无 CUDA，模块 skip | 静态完成；需首张 NVIDIA GPU 动态验收 |
| E — 2.2 Grouped GEMM | 无 CUDA，模块 skip | 无 CUDA，模块 skip | 静态完成；需首张 NVIDIA GPU 动态验收 |
| F — 7.2b / 7.2c review drills | PR 自带测试全绿 | 不适用 | 完成；两套盲测均命中 12/17 |

## A — 4.2 PD 分离模拟

交付：`m4-inference/minipd/`、`tests/test_pd.py`、
`reference/4.2-pd-sim/`、`benchmarks/bench_pd.py`、
`docs/4.2-pd-sim/POSTMORTEM.md`。

参考态实测：

```text
.........                                                                [100%]
9 passed in 0.02s
```

恢复骨架后实测：

```text
FFFFFFFFF                                                                [100%]
9 failed in 0.06s
```

变异自检：T1 将向上取整改成向下取整；T2 忽略 chunk 上限；T3 迁移耗时加一；
T4 允许同 tick 补 decode slot；T5 忽略 workload seed；T6 禁用 chunk；T7 让
decode 每 20 tick 才运行；T8 在 prefill 完成时错误记录首 token；T9 删除单 token
分母保护。T1–T9 均由对应测试捕获，随后恢复参考实现。

决策：arrival 在到达 tick 末入队；transfer 从下一 tick 开始，倒计时归零的 tick
可进入 decode；prefill 按 FIFO round-robin，每请求每 tick 最多一个 chunk；
`total_ticks = last_finish_time + 1`。

## B — 3.1 DSv4 机制

交付：`m3-arch-study/minidsv4/`、`tests/test_dsv4.py`、
`reference/3.1-dsv4/`、`scripts/ablate_dsv4.py`、
`docs/3.1-dsv4/POSTMORTEM.md`。

参考态实测：

```text
.....F...                                                                [100%]
1 failed, 8 passed
```

失败仅为冻结 T6。严格使用 SPEC 给出的系数、5 次 Newton–Schulz 迭代与
`X0 = G / ||G||F`，固定矩阵上的正交误差约为 `0.3714`（门槛 `< 0.05`），
相对 SVD 极分解误差约为 `0.2146`（门槛 `< 0.1`）。另一独立 seed 也得到
`0.3648 / 0.2126`。因此正确实现不能通过两个冻结阈值，未放松测试、未替换公式。

排除冻结矛盾项的参考结果：

```text
........                                                                 [100%]
8 passed, 1 deselected
```

恢复骨架后实测：

```text
FFFFFFFFF                                                                [100%]
9 failed in 0.83s
```

变异自检：T1 sparse attention 清零；T2 放行未来块；T3 错误构造块并集；
T4/T5 破坏 HyperConnection 的 identity 系数；T7 接受非 2D 参数；T8 将 Muon
替换为 SGD，均被对应测试捕获。T6 的“清零 msign”也失败，但正确基线本身已失败，
所以该项不能算有效的 mutation 区分能力，须在 SPEC 修订后重做。

疑义：冻结构造签名没有 `identity_init` 参数，但文字要求 identity init；实现采用
无参数的默认 identity 初始化。除此之外未改变冻结接口。

## C — 3.2 长上下文

交付：`m3-arch-study/minikda/`、`tests/test_longctx.py`、
`reference/3.2-longctx/`、`scripts/ablate_longctx.py`、
`docs/3.2-longctx/POSTMORTEM.md`。

参考态实测：

```text
................                                                         [100%]
16 passed, 1 warning in 2.00s
```

warning 仅来自临时 overlay 目录没有读取仓库的 pytest marker 配置；仓库已注册
`slow`。T8 固定 seed、双层、严格 200 步，整套参考测试约 2 秒，满足 `<60s`。

恢复骨架后实测：

```text
FFFFFFFFFFFFFFFF                                                         [100%]
16 failed in 0.83s
```

变异自检：T1 chunk 输出清零；T2 删除 overwrite/erase；T3 recurrent 输出清零；
T4 忽略 alpha；T5 step 输出清零；T6 `state_bytes=0`；T7 删除块内因果 mask；
T8 attention 输出清零。T1–T8 均被对应测试捕获。

决策：SPEC 未给 cfg 与 `DeltaAttention.__init__` 结构，因此新增与仓库风格一致的
`KDAConfig` 和 `DeltaAttention(cfg)`；状态按冻结的 fp32 累加口径计字节，即便
输入 dtype 为 fp16；chunk 参考实现使用块内单位下三角 solve，不包装 recurrent。

## D — 2.1 进阶 kernel

交付：`kernels/rope.py`、`kernels/block_attn.py`、`tests/test_kernels_21.py`、
`reference/2.1-advanced/`、`benchmarks/bench_advanced.py`、
`docs/2.1-advanced/POSTMORTEM.md`，并扩展 `scripts/validate_reference.sh`。

本机验证：

```text
$ python3 -m py_compile ...
(no output)
$ python3 -m pytest tests/test_kernels_21.py -q
1 skipped in 0.70s
```

参考态和骨架态都会在 module-level CUDA guard 处 skip；因此本机不能声称参考态
全绿，也不能实际执行 mutation。首张真卡必须运行扩展后的
`scripts/validate_reference.sh`。脚本现有 EXIT trap，测试失败或中断也会恢复骨架。

逐行静态推演：

- RoPE 将 `(B,H,T,D/2)` pair 展平；`vector = pair // half`，
  `token = vector % T`，高半索引为低半加 `D/2`，与 front/back-half 契约一致。
  cos/sin 与输入提升到 fp32 做旋转，store 时回到输入 dtype；反向使用 `sign=-1`
  即正交旋转的转置。
- BlockAttention 的 program 轴为 `(B*H, ceil(T/BLOCK_M))`。每个 key tile 更新
  online-softmax 的 running max、重标定旧 accumulator、累加新指数与分母；causal
  mask 和 K/Q 尾 mask 分离。合法 causal 行的首块至少含 key 0；padding query 即使
  产生 NaN 也被最终 store mask 隔离。
- 两次 `tl.dot` 明确使用 `input_precision="ieee"`，避免 Ampere TF32 破坏 fp32
  `1e-5` 门槛；T4 的 common `+80` score 用独立 SDPA oracle 检查稳定性。
- 内核没有分配 `(T,T)` score；T5 用 T=4096 的峰值显存上界检查这一点。
- benchmark 同时打印 Triton/eager RoPE 与 Triton/SDPA 中位耗时；交叉审阅发现的
  eager 基线遗漏已修复，并加强 T6 输出断言。

静态 mutation 矩阵：T1 错误 pair/token 索引；T2 去掉 fp32 提升；T3 去掉 causal
或尾 mask；T4 改回直接 `exp(score)`；T5 物化 score；T6 删除 eager/SDPA 基线。
测试分别针对这些失效机制，但因无 CUDA 未执行，不能冒充动态 mutation 结果。

## E — 2.2 Grouped GEMM

交付：`kernels/grouped_gemm.py`、`tests/test_kernels_22.py`、
`reference/2.2-grouped-gemm/`、`benchmarks/bench_grouped_gemm.py`、
`docs/2.2-grouped-gemm/POSTMORTEM.md`，并纳入同一参考验证脚本。

本机验证：

```text
$ python3 -m py_compile ...
(no output)
$ python3 -m pytest tests/test_kernels_22.py -q
1 skipped in 0.67s
```

逐行静态推演：

- host 以向量运算从 group offsets 得到各组 row-tile 前缀和；Grouped GEMM 本体只
  launch 一次。grid 的 `N+E` 是 tile 总数安全上界，多余 program 由
  `TILE_OFFSETS[E]` 屏蔽。
- 每个 program 在 device 端扫描 tile 前缀和定位 expert；空组占零 tile；
  `row_end`、N、M、K 四类尾界分别进入 load/store mask。
- K 分块以 fp32/IEEE 累加，最终按输出 dtype 存储；全空 N=0、单行组、非 tile
  整除尾块现已对 E=1/8/64 显式参数化，不再依赖随机 seed 偶然覆盖。
- `moe_ffn_grouped` 是 gate/up 两次 grouped GEMM、给定 SwiGLU、down grouped GEMM；
  T4 通过 `parents[2]/m3-arch-study/reference/3.0-moe` 导入独立 SwiGLU 参考对拍。
- benchmark 比较 grouped、Python expert loop、padding-bmm 三条路径。

静态 mutation 矩阵：T1 错误 tile→expert/尾 mask；T2 不支持空组或极端倾斜；
T3 放松 offsets 合同；T4 丢 gate/up/down 任一路；T5 删除任一 benchmark 基线。
测试覆盖这些机制，但同样因无 CUDA 未执行。首张真卡还应重点确认 Triton 对动态
`range` 的编译，以及 E=1/8/64、所有 D/BLOCK 配置。

## F — 7.2b / 7.2c Review 演练

交付：`m7-agent-engineering/exercises/7.2b-review-training/`、
`exercises/7.2c-review-data/` 及各自 `reference/<name>/ANSWERS.md`。

PR 自带测试：

```text
................                                                         [100%]
16 passed in 1.40s
```

泄题扫描：

```text
$ rg -n -i 'bug|fixme|hack|wrong|broken|leak|注意|故意' exercises/7.2b-review-training/pr exercises/7.2c-review-data/pr
(no output; 0 matches)
```

两套各埋 17 项，分布均为 P0×4 / P1×8 / P2×5。training 的 5 项、data
的 4 项属于测试自身缺陷；data 的时序/隔离类为 items 1、3、4，满足至少 3 项。
两份 `ANSWERS.md` 的每个 item 都有 file:line、机制、severity、正确修法、可复制
Python 最小复现与实际输出（合计 34 项均实跑）。

答案侧高价值暴露测试在原始假 PR 上的结果：

```text
FFFF                                                                     [100%]
4 failed in 1.15s
```

它们分别覆盖 training 的累积等价性、scheduler checkpoint 恢复，以及 data 的
next-token labels、时间 holdout。对临时正确修复副本联合运行：

```text
....                                                                     [100%]
4 passed in 1.22s
```

随后丢弃临时副本，原始假 PR 保持埋雷状态。两个文件使用唯一模块名，根目录联合
collect 为 `4 tests collected in 0.70s`，不存在 import-file-mismatch。

盲测均由未读 reference/ANSWERS、未修改文件的独立 subagent 执行：

- data 首轮严格映射命中答案 items 1–12，即 `12/17`；另发现 1 个性能候选，不计分。
- training 首版 raw 命中过多，按 brief 判定过显后调整再盲测。v2 严格映射同样命中
  items 1–12，即 `12/17`；答案外的 device、overflow-step、RNG、超 total 振荡
  只按 INSTRUCTIONS 记口头加分，不计总分。5 个测试缺陷没有被逐项命中。

两套最终命中率均处于冻结 brief 期望的 6–12 上界。

## 全局收尾检查

- 未修改 README、index 或课程计数文件；未 commit/push/branch。
- A–E 学生包处于骨架状态，Foundation 提示保持为问题式，不含答案公式/伪码。
- 新增测试按 T 编号命名，固定随机 seed；oracle 使用手算、PyTorch 独立算子或
  独立参考模型，不调用被测实现。
- `bash -n m2-kernels/scripts/validate_reference.sh`、新 Python 文件
  `py_compile`、`git diff --check` 均通过。
- B/T6 是唯一已知冻结验收阻塞；D/E 的 CUDA 动态结果明确留待 NVIDIA GPU。
