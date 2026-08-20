# Phase 3.1 POSTMORTEM

完成实验后，请结合测量结果回答，而不是只复述定义。

1. Block sparse attention 分别省下哪些 FLOPs 与 KV 存取？为什么 decode 和
   prefill 的瓶颈与收益结构不同？dense-mask 参考实现又为什么不代表真实加速？
2. Hyper-Connections 如何用多条 residual stream 改善深层梯度传播？冻结的恒等
   初始化为何保证训练起点与标准 Pre-Norm residual 一致？
3. `msign(G)` 的更新几何与 SGD、Adam 有何区别？为何 Muon 只接管二维矩阵参数，
   而向量/标量参数应交给 AdamW？
