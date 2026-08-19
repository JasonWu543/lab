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
| 1.0 BPE Tokenizer | 备课完成，12 tests 红，闯关中 |
| 1.1 Tensor & Autograd | 备课完成，49 tests 红 |
| 1.2 Qwen-like Transformer | 备课完成，35 tests 红 |
| 1.3 训练框架 Trainer | 备课完成，26 tests 红 |

```bash
cd m1-foundation
python3 -m pytest tests/test_bpe.py -x -q   # 从这里开始
```
