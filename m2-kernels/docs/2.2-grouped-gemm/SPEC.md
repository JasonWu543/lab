# SPEC — Phase 2.2: Grouped GEMM（面向 MoE）

> 状态：FROZEN（接口已冻结）
> 模式：同 2.0 —— kernel 学生手写；wrapper / 数据准备给定
> ⚠️ 同 2.1：本地无 CUDA，测试全 CUDA-skip，租卡跑 validate_reference.sh

## 1. 问题

MoE 的 expert FFN 是一批**行数不等的小 GEMM**（每个 expert 分到的 token 数不同，
且随 routing 波动）。逐 expert 循环调 `torch.mm` 会付出 E 次 kernel launch 且
小 expert 打不满 SM；padding 成等大 batch 则浪费算力。Grouped GEMM 用单个
kernel 处理所有组：每个 program 先查自己属于哪个组，再做该组的 tile。

学完必须能回答（写进 POSTMORTEM）：
- launch overhead 和 tail effect 各在什么 (E, tokens/expert) 区间主导？
- 组边界不对齐 tile 时浪费在哪里？block size 怎么权衡？
- 与 m3 的 MoE dispatch 如何衔接（token 排序 + offsets 就是本 kernel 的输入契约）？

## 2. 冻结接口（kernels/grouped_gemm.py）

```python
def grouped_gemm(x: torch.Tensor, weights: torch.Tensor,
                 group_offsets: torch.Tensor) -> torch.Tensor:
    """x: (N_total, K) —— 已按 expert 排序的 token 激活；
    weights: (E, K, M) —— 每 expert 一个权重矩阵；
    group_offsets: (E+1,) int32/int64，第 e 组行区间 [offsets[e], offsets[e+1])，
    允许空组（区间为空）。
    返回 (N_total, M)：out[s:e] = x[s:e] @ weights[g]。
    fp32 累加，输出 dtype 同 x；单 kernel launch（不许 host 端逐组循环）。"""

def moe_ffn_grouped(x, w_gate, w_up, w_down, group_offsets) -> torch.Tensor:
    """给定 wrapper：grouped_gemm ×3 + 2.0 的 fused SwiGLU 拼成完整 expert FFN，
    与 m3 minimoe 的 dispatch 输出契约对齐。"""
```

学生 kernel 设计要求（骨架提示允许给结构，不给完整索引算式）：
tile 到组的映射需在 device 端由 program_id 解出（提示允许给「前缀和查找」
思路）；K 维循环 fp32 累加；组尾不足 tile 的行用 mask。

## 3. 验收标准（tests/test_kernels_22.py，无 CUDA 全 skip）

| 编号 | 通过条件 |
| --- | --- |
| T1 | 对拍逐组 `torch.mm` 循环 oracle：随机组大小（含 0 行空组、1 行组、非 tile 对齐组）× E ∈ {1, 8, 64} × fp32/bf16（bf16 容差写明推导依据）|
| T2 | 极端分布：全部 token 集中单一 expert / 均匀分布，两种极端下均正确 |
| T3 | group_offsets 契约：非递增 offsets 必须 raise；offsets[-1] != N_total 必须 raise（wrapper 层校验，给定）|
| T4 | moe_ffn_grouped 端到端对拍 m3 minimoe 参考实现的 expert FFN 输出（同权重注入，atol 1e-4）|
| T5 | benchmark：vs 逐组 mm 循环、vs padding-bmm 两条 baseline 的耗时表（warmup 后中位数；不做硬性加速断言，真卡填 BASELINE）|

## 4. 产物

- `kernels/grouped_gemm.py` 静态自查通过 + 真卡全绿
- `scripts/validate_reference.sh` 扩展含 2.2
- `docs/2.2-grouped-gemm/POSTMORTEM.md`
