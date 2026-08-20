# Phase 8.1 Postmortem

> 完成实验后填写。请给出推导，不要只写结论。

## 1. Column Parallel 的输入梯度

设完整权重沿输出维切为 $W=[W_0;\ldots;W_{p-1}]$，写出本 rank 的
$Y_r=XW_r^T$，再从链式法则推导 $\partial L/\partial X$。解释为什么各 rank
算出的项必须 all-reduce，以及这里为什么是求和而不是求平均。

## 2. Row Parallel 的前向通信

设权重与输入沿输入维切分，写出完整线性层输出如何由每个 rank 的局部部分积
组成。说明 backward 对局部输入和局部权重为什么不需要额外 gather。

## 3. TP 与 PP 的通信量

分别列出 TP collective 与 PP stage-boundary activation 的张量形状。讨论通信量
随 batch、sequence、hidden width、stage/tensor-parallel degree 如何增长，以及宽模型
和深模型分别更适合哪种并行。

## 4. GPipe 与 1F1B

推导 GPipe bubble fraction $(p-1)/(m+p-1)$。解释 1F1B 相对 flush GPipe
主要减少的是 bubble，还是峰值激活显存，并说明原因。

## 5. Mean loss 与 micro-batch 累积

写出全 batch mean loss 与 $m$ 个等大小 micro-batch mean loss 的关系。记录若忘记
对每个 micro-batch loss 除以 $m$，参数梯度会发生什么变化。

## 6. Backlog

- [ ] 在真实多卡上测 TP collective 与 PP point-to-point 吞吐。
- [ ] 实现并对拍 1F1B 调度。
