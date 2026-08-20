# SPEC — Phase 2.0: Triton 入门三件套

> 状态：FROZEN（接口已冻结）
> 模式：Foundation（kernel 本体手搓；autograd.Function 包装与 benchmark 脚手架给足提示）
> 算力：**必须 NVIDIA 卡**（S 级；kernel 开发极省卡时，但 MPS/CPU 无意义）
> 工期：约 1 周（W7 副线）
>
> ⚠️ 验证状态：本作业包在无 GPU 环境下备课，测试与参考答案**未经真卡验证**。
> 第一次租卡时先跑 `scripts/validate_reference.sh`（把参考答案覆盖到骨架上
> 跑全部测试）——全绿后才开始闯关；有红说明是出题人的 bug，先报给出题人修。

## 1. 问题

用 Triton 写三个训练中真实使用的 fused kernel（各含 forward + backward），
与 PyTorch eager 数值对齐，并能解释每个 kernel 为什么快/慢、
瓶颈是 memory 还是 compute。

学完必须能回答（写进 POSTMORTEM）：
- 这三个 kernel 都是 memory-bound 的——用 roofline 说明为什么，
  以及 fusion 到底省了什么（读写次数怎么数）；
- BLOCK_SIZE 怎么选？太大/太小各发生什么？
- 为什么 softmax/cross-entropy 需要 online 算法才能单 pass？
- bf16 下容差为什么要放宽？累加为什么必须在 fp32 做？

## 2. 范围与非目标

范围：单卡、推理+训练（fwd+bwd）、fp32 与 bf16、行数/维度非 2 的幂也要对。
非目标：不做 matmul/attention kernel（2.1 的事，本期 backlog）、
不做 autotune 搜索（固定几组 BLOCK_SIZE 即可）、不做多卡。

## 3. 冻结接口（kernels/）

```python
# kernels/rmsnorm.py
def rmsnorm(x: Tensor, weight: Tensor, eps: float = 1e-6) -> Tensor:
    """x: (..., H) 任意前缀维；Triton fwd+bwd（torch.autograd.Function 包装）。
    与 minilm 1.2 的 RMSNorm 语义一致：fp32 内部计算，输出原 dtype。"""

# kernels/swiglu.py
def swiglu_mul(gate: Tensor, up: Tensor) -> Tensor:
    """fused silu(gate) * up（逐元素部分融合；两侧 matmul 留在 torch）。
    fwd 单 kernel；bwd 单 kernel 同时算 d_gate 和 d_up。"""

# kernels/cross_entropy.py
def fused_cross_entropy(logits: Tensor, targets: Tensor,
                        ignore_index: int = -100) -> Tensor:
    """logits: (N, V), targets: (N,) → 标量 mean loss（忽略 ignore_index）。
    fwd：online softmax（单 pass 求 max 与 logsumexp），不物化 probs；
    bwd：原地利用保存的 logsumexp 重算 softmax，直接写出 dlogits。
    这是显存收益最大的一个：对照 torch 版本要物化 (N, V) 的 probs。"""
```

约定：

- 每个 kernel 一个 `torch.autograd.Function`；对外只暴露上面三个函数；
- 所有累加（sum/logsumexp）在 fp32 中进行，无论输入 dtype；
- grid 按行划分（每行一个 program），H/V 大于 BLOCK_SIZE 时循环分块；
- 不许调用 torch 的对应算子作为实现（它们是测试里的对照物）。

## 4. 验收标准（tests/test_kernels.py，全部 @cuda-only）

| 编号 | 通过条件 |
| --- | --- |
| T1 | rmsnorm fwd/bwd 与 torch 参考实现对齐：fp32 rtol 1e-5 / bf16 rtol 2e-2；shape 覆盖 (B,T,H) 与 (N,H)，H ∈ {512, 768, 1000(非2幂), 4096} |
| T2 | swiglu_mul fwd/bwd 对齐（同上容差与 shape 策略）|
| T3 | fused_cross_entropy 与 F.cross_entropy 对齐：loss fp32 atol 1e-5；dlogits 对齐；含 ignore_index 用例 |
| T4 | 显存：V=32k、N=4096 时 fused CE 的增量峰值显存 < torch 版的 60%（输入分配后记录基线，以 `torch.cuda.max_memory_allocated - baseline` 对比，并在两轮间释放 autograd 图）|
| T5 | 数值稳定：logits 含 ±1e4 极值时 loss 有限且与 fp64 参考一致（rtol 1e-3）|

benchmark（`benchmarks/bench_kernels.py`，不进测试）：warmup 后测
kernel vs torch eager 的耗时与实测带宽（GB/s），对照卡的理论带宽给出
利用率百分比——这是 POSTMORTEM 第一问的素材。

## 5. 产物

- `kernels/{rmsnorm,swiglu,cross_entropy}.py`（学生实现）
- 三份 benchmark 数字表 + `docs/2.0-kernels/POSTMORTEM.md`（含第 1 节四问）
