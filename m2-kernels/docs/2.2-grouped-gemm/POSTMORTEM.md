# Phase 2.2 Postmortem

> 完成真卡验收后填写。

## 正确性

- 空组、单行组和 tile 尾块分别暴露了哪些边界错误？
- Grouped GEMM 与 m3 MoE dispatch 的数据契约如何衔接？

## 性能

- 填写 grouped / 逐组 mm / padding-bmm 的 warmup 后中位耗时。
- 在什么 expert 数量与 tokens/expert 区间，launch overhead 或 tail effect 主导？
- row/column/K block 的选择如何影响空算、占用率和寄存器压力？

## 决策与遗留项

- tile→group 查找的代价是多少？是否需要更高效的映射表？
- 下一步是否需要 fused grouped SwiGLU / down projection？
