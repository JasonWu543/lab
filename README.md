# AI Systems Lab

一套 CS336 式的个人能力 lab：Claude 出题（SPEC / 骨架 / 测试关卡 / 参考答案），
我手写核心实现闯关。从 BPE、Autograd、Transformer 一路到推理系统、后训练与 kernel。
七个独立模块，十二周完成。

**项目主页：<https://jasonwu543.github.io/lab/>**

## 导航

- [LAB_DESIGN.md](LAB_DESIGN.md) — 总体设计（7 模块、12 周路线、验收模板）
- [READINGS.md](READINGS.md) — 各模块阅读清单
- [m1-foundation/PLAN.md](m1-foundation/PLAN.md) — M1 逐单元拆解与验收测试
- `m*/docs/<phase>/` — 各 phase 的 SPEC / BASELINE / POSTMORTEM
- `m*/reference/` — 参考答案（学生卡住 30 分钟以上才看）

## 当前进度

| Phase | 状态 |
| --- | --- |
| 1.0 BPE Tokenizer | 备课完成，13 tests 红，闯关中 |
| 1.1 Tensor & Autograd | 备课完成，51 tests 红 |
| 1.2 Qwen-like Transformer | 备课完成，38 tests 红 |
| 1.3 训练框架 Trainer | 备课完成，26 tests 红 |
| 2.0 Triton 三件套 / 2.1 进阶 / 2.2 Grouped GEMM | 备课完成（本机无 CUDA 时 3 skipped；⚠️ 均未真卡验证，首次租卡先跑 `m2-kernels/scripts/validate_reference.sh`）|
| 3.0 MoE / DeepSeek 机制 | 备课完成，16 tests 红 |
| 3.1 DSv4 机制（sparse attn / Hyper-Connections / Muon）| 备课完成，9 tests 红 |
| 3.2 长上下文（Delta rule / 状态压缩）| 备课完成，16 tests 红 |
| 4.0 mini-vLLM | 备课完成，23 tests 红 |
| 4.1 投机解码 | 备课完成，17 tests 红 |
| 4.2 PD 分离模拟 | 备课完成，9 tests 红 |
| 5.0 SFT / 5.1 DPO / 5.2 GRPO | 备课完成，43+26+29 tests 红（真实模型/数据见 `m5-post-training/docs/RESOURCES.md`）|
| 6.0 数据工程 / 6.1 Scaling law | 备课完成，37+24 tests 红 |
| 7.2 / 7.2b / 7.2c Review 演练 | 材料就绪：三个埋雷假 PR（18/17/17 个已知问题，各限时 45 分钟），教具自测 41+16 passed（见 `m7-agent-engineering/exercises/`）|

```bash
cd m1-foundation
python3 -m pytest tests/test_bpe.py -x -q   # 从这里开始
```
