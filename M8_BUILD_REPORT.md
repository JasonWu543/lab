# M8 — 训练系统与效率：建造报告

日期：2026-08-20

环境：macOS、Python 3.12、PyTorch 2.5.1、CPU gloo、无 CUDA

状态：M8 独立提交；学生目录保持骨架态。

## 交付总览

| Phase | 学生包 | 测试 | 参考答案 | 参考态 | 骨架态 |
| --- | --- | --- | --- | --- | --- |
| 8.0 DDP / ZeRO | `minidist/` | `tests/test_dist.py` | `reference/8.0-ddp-zero/` | 8 passed | 6 failed, 2 errors |
| 8.1 TP / PP | `minitp/`, `minipp/` | `tests/test_tp_pp.py` | `reference/8.1-tp-pp/` | 10 passed | 8 failed, 2 errors |
| 8.2 低精度 / FP8 | `minilp/` | `tests/test_lowprec.py` | `reference/8.2-low-precision/` | 11 passed | 11 failed |

三个 phase 均含只提问、不含答案的 `POSTMORTEM.md`。8.0 的进程管理是冻结标注的
给定脚手架；其他学生接口以 docstring、问题式提示和 `NotImplementedError` 收尾。

## 8.0 — 梯度桶 DDP、ZeRO-1、ZeRO-2

交付文件：

- `minidist/{__init__,comm,bucket,zero}.py`
- `reference/8.0-ddp-zero/{bucket,zero}_solution.py`
- `tests/test_dist.py`
- `docs/8.0-ddp-zero/POSTMORTEM.md`

给定 `run_distributed` 使用 spawn + gloo + file rendezvous。每次运行有 60 秒外层
watchdog，process group 也设置 60 秒 timeout；父进程在等待 worker 的同时持续排空
结果 Queue，避免大返回值填满 pipe 后出现“先 join、后 get”的死锁。

参考态实测：

```text
........                                                                 [100%]
8 passed in 9.61s
```

初版冻结门槛曾要求对单进程整批 oracle `torch.equal`。实测 T2 三个 bucket cap 与
T3 的 20 步轨迹最大绝对差均为 `2.9802322387695312e-08`：整批 reduction 与两个
半批分别求梯度再 SUM/2 的 fp32 结合顺序不同。经授权修订后，对单进程 oracle 使用
`rtol=1e-6, atol=1e-7`；各 rank 之间和三个 bucket cap 之间仍必须逐位一致。

骨架态实测：

```text
FFEFEFFF                                                                 [100%]
6 failed, 2 errors in 6.87s
```

全套正好 6 次 spawn：T2 一次；T3/T5 共享一次 module fixture；T6 一次；T7 一次；
T8 两次。ZeRO-1 state 闭式总计 544 B，两 rank 分别 284 B / 260 B。T6 使用一份
故意不同于 `shard_params` 默认结果的合法 ownership，确保实现真的消费传入 shards。

Mutation 记录：

- T1：改成正序装桶或错误处理 cap 边界，closed-form 失败。
- T2：漏除 world size / 只保留单 rank，数值 oracle 或 rank/cap bitwise 门失败。
- T3：漏 broadcast / 所有 rank 都更新，rank 一致性或容差轨迹门失败。
- T4：删除 numel 降序或改变 tie-break，精确 shards 失败。
- T5：漏算每参数 step tensor，字节闭式失败。
- T6：不平均、保留非 owned、忽略传入 shards，None/独立 oracle 失败。
- T7：错误地固定除 world size，world_size=1 退化门失败。
- T8：引入非确定随机扰动，两次 rank 返回值不一致。

## 8.1 — Tensor Parallel 与 Pipeline Parallel

交付文件：

- `minitp/{__init__,layers}.py`
- `minipp/{__init__,schedule,runner}.py`
- `reference/8.1-tp-pp/{layers,schedule,runner}_solution.py`
- `tests/test_tp_pp.py`
- `docs/8.1-tp-pp/POSTMORTEM.md`

TP 参考实现的 Column forward / input-gradient backward 与 Row partial-output forward
分别由两个 autograd 原语表达。测试从各 rank 的实际权重分片重建完整 eager MLP，
避免强制 SPEC 没有冻结的 RNG 分布、`0.02` 缩放或 row `seed+1` 细节；同时统计
模型调用区间内 collective，要求恰好两次 all-reduce、零次 all-gather。

GPipe 测试机器检查依赖、每 stage 每 tick 单操作、flush、操作唯一性、总 tick 和实际
空闲槽比例；没有强制 SPEC 未声明的 backward micro-batch 顺序。

参考态实测：

```text
..........                                                               [100%]
10 passed in 5.00s
```

初版 T4 的“逐位一致”在 m=2/4 时出现约 `7.45e-09` 至 `2.98e-08` 的正常 fp32
归约差异，而且会反向奖励“忽略 micro-batch、直接整批 backward”的绕过实现。
修订后 m=1 保留 `torch.equal`，m>1 使用 `rtol=1e-6, atol=1e-7`。测试同时记录
loss_fn 看到的 batch sizes，强制实际执行 m 个大小为 B/m 的 micro-batch，并核对
返回的全批 mean loss；另有独立 mean-gradient 语义门检查 `/m` 缩放。

骨架态实测：

```text
EEFFFFFFFF                                                               [100%]
8 failed, 2 errors in 3.23s
```

Mutation 记录：

- T1：删除 Row forward all-reduce，forward 30/30 元素不匹配。
- T2：删除 Copy backward all-reduce，输入梯度不匹配；冗余 all-gather 也被计数门捕获。
- T3：错误 bubble 分母或破坏依赖/tick，调度门失败。
- T4：删除 micro-loss `/m`，独立 mean-scaling 门出现约 3 倍梯度；忽略切分由调用
  batch-size 记录捕获；m=1 exact、m>1 容差门分别验收确定性与数值等价。
- T5：接受 empty Sequential stage 或不可整除 batch，边界门失败。
- T6：使用非确定 seed，双跑 TP/PP 返回值不一致。

## 8.2 — 混合精度、FP8 与 delayed scaling

交付文件：

- `minilp/{__init__,fp8,scaler,train}.py`
- `reference/8.2-low-precision/{fp8,scaler,train}_solution.py`
- `tests/test_lowprec.py`
- `docs/8.2-low-precision/POSTMORTEM.md`

参考态实测：

```text
...........                                                              [100%]
11 passed in 1.79s
```

骨架态实测：

```text
FFFFFFFFFFF                                                              [100%]
11 failed in 1.41s
```

实现与测试要点：

- `fp8_finfo` 从 exponent/mantissa/bias 闭式推导；AST 门禁止直接调用
  `torch.finfo`，而测试仍用 `torch.finfo` 作独立 oracle。
- E4M3FN / E5M2 在 cast 前显式 saturate，避免原生超界 cast 产生 NaN/inf。
- delayed scaling 先读取过去窗口 scale，再记录当前 amax；spike 当步饱和，下一步适配。
- `SimpleGradScaler` 按 optimizer 保存 unscaled/stepped/found-inf 状态；一个 optimizer
  的 inf 只跳过它自己的 step，scale update 再汇总本轮所有 found-inf；重复 step 拒绝。
- fp32 master weights 累积小于 fp16 ULP 的更新，再同步低精度副本。
- T7 固定 seed 跑 5 步：fp32 loss `3.875589 → 1.515319`，模拟 FP8
  `3.875589 → 1.820055`，最终相对差约 `20.11% < 30%`；重复 5 次稳定。
- 定点 `1.125 × 1.0` 在 E4M3 输出 1.125、错误 E5M2 输出 1.0，明确验收 forward
  必须用 E4M3。交叉审阅前 E5M2 mutant 曾全绿；补门后 T7 会失败。
- `margin` 冻结为非负整数；margin=m 时验证
  `amax*scale ∈ (max/2^(m+1), max/2^m]`，非法类型、bool 与负数均拒绝。

Mutation 记录：T1 调用 `torch.finfo`；T2 删除 clamp；T3 错误 floor/异常 amax；
T4 不淘汰旧窗口最大值；T5 错误 step/update、全局 found-inf、重复 step；T6 直接在
fp16 更新；T7 改用 E5M2 或绕过量化；T8 当前步抢先使用 spike scale。补强后均被
对应门捕获。

## 已授权的 SPEC 修订

1. **8.0 T2/T3**：单进程整批 oracle 改为 `rtol=1e-6, atol=1e-7`；rank 之间、
   bucket cap 之间保留 bitwise。
2. **8.1 初始化**：冻结 CPU Generator、`torch.randn * 0.02`、无 bias，以及 TPMLP
   Column/Row 的 `seed` / `seed+1` 规则。
3. **8.1 T4/T5**：m=1 exact、m>1 使用同一明确容差；增加真实 micro-batch 调用门；
   将无法由现有接口一般判断的“stage 数>层数”具体化为拒绝 empty Sequential stage。
4. **8.2 T3**：冻结 margin 为非负整数，并把范围公式推广到一般 margin。

## 全局检查

- 经用户授权修订三份 M8 SPEC；未修改 README 或既有 M1–M7 文件。
- 多进程测试均使用 CPU gloo + file rendezvous；8.0 spawn≤6，8.1 spawn=3。
- `py_compile`、`git diff --check`、骨架泄题扫描通过。
- 最终工作区为学生骨架；参考实现只在临时副本或 `M8_REFERENCE=1` 下运行。
