# Phase 8.2 Postmortem — 低精度训练

> 完成实现并通过测试后填写。请结合自己的失败样例和数值记录回答，不要只复述 SPEC。

## Loss scaling 解决什么

- 哪类 tensor 在哪一步会下溢？loss scaling 为什么能保住这些值，却不能修复
  forward 溢出？
- 为什么 bf16 通常不需要同样的 loss scaling？它为动态范围牺牲了什么？
- 写出你实现的完整状态机，并记录一次注入 inf 后的参数、scale 与 growth tracker。

## E4M3 与 E5M2 的职责

- 从 exponent/mantissa 位数推导两个格式的最大有限值和最小正规数。
- 为什么 forward 激活/权重与 backward 梯度通常选择不同格式？用一次量化误差实验
  支持你的解释。

## Delayed scaling 为什么工作

- 为什么过去窗口的 amax 可以预测下一步尺度？读取 scale 与记录当前 amax 的先后
  顺序会怎样改变 spike batch？
- 记录 T8 中 spike 前、spike 当步和下一步的 scale、量化值与反量化值。
- margin 提供了什么 headroom？它如何改变冻结公式中的范围断言？

## 实现取舍与验证

- 对比直接 fp16 SGD 与 fp32 master weight 的 200 步轨迹，并解释停滞点。
- 记录 toy 回归中 fp32 与模拟 FP8 的 loss 曲线、最终相对差和是否出现非有限值。
- 本 phase 的 CPU 模拟能验证哪些机制？哪些吞吐与稳定性结论必须留到真 FP8 硬件？

## Backlog

真 H100 FP8 tensor core、Transformer Engine 对照、per-tensor/per-channel 粒度、
分布式 amax 归约、随机舍入以及真实 Transformer 收敛实验留待后续 phase。
