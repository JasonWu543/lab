# Phase 2.1 Postmortem

> 完成真卡验收后填写；不要只粘 benchmark 数字，要解释机制。

## Fused RoPE

- 从二维旋转矩阵推导 backward。为什么角度取负即可复用 forward kernel？
- fp32 与 bf16 的最大误差分别是多少？与验收容差是否一致？
- eager / Triton 的 warmup 后中位耗时与加速比：

## Blockwise Attention

- 推导 online softmax 更新时旧分母与旧 accumulator 的重标度。
- T=4096 的增量峰值显存与显式 score 闭式各是多少？
- 哪些 `(T,D)` 下 blockwise 反而慢于 SDPA？结合 launch、tile 利用率解释。

## 决策与遗留项

- RoPE backward 选择复用还是独立 kernel，理由是什么？
- 真 backward kernel（bonus）是否值得实现？
