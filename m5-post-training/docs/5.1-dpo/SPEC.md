# SPEC — Phase 5.1: 偏好优化（Reward Model + DPO）

> 状态：FROZEN（接口已冻结）
> 模式：Foundation —— RM 的 BT loss 与 DPO loss 全部手写（这是本 phase 的全部意义）
> 基座：官方 transformers tiny 模型；不用 trl
> 算力：correctness 本地；真实 DPO 一次 M 级（W10）
> 工期：约 0.5 周

## 1. 问题

实现偏好学习的两条路：显式 Reward Model（Bradley-Terry）与隐式的 DPO。
核心是把 DPO 论文的公式一行行写对，并用「隐式 reward margin 上升、
reference 不动」这两个不变量验证。

学完必须能回答（写进 POSTMORTEM）：
- DPO 的 loss 是怎么从 RLHF 目标闭式推出来的？β 控制什么？
- reference model 在里面起什么作用？去掉它（β→∞ 或 ref=均匀）会怎样？
- 为什么 DPO 里 chosen 和 rejected 的 logp 都可能下降，但 margin 上升？
- RM 路线和 DPO 路线各自的失效模式是什么？

## 2. 范围与非目标

范围：sequence log-prob（带 prompt mask）、BT reward loss、DPO loss、
隐式 reward、toy 偏好数据上的训练收敛验证。
非目标：不做 PPO（5.2 用 GRPO）、不做 IPO/KTO 变体（backlog）、
不做真实偏好数据清洗。

## 3. 冻结接口（minidpo/）

```python
# minidpo/logprob.py
def sequence_logprob(model, input_ids: Tensor,      # (B, T)
                     prompt_lens: Tensor            # (B,)
                     ) -> Tensor:                   # (B,)
    """每条序列 response 部分（位置 >= prompt_len）的 log p 之和。
    注意 off-by-one：位置 t 的 logits 预测 t+1；pad 用 attention_mask 之外
    直接按长度处理（本 phase 统一右 pad，pad 不计入）。"""

# minidpo/rm.py
class RewardModel(nn.Module):
    def __init__(self, base_model): ...
    # base 最后隐层 → Linear(H, 1)，取每条序列**最后一个非 pad token** 的标量
    def forward(self, input_ids: Tensor, seq_lens: Tensor) -> Tensor:  # (B,)

def bt_loss(chosen_rewards: Tensor, rejected_rewards: Tensor) -> Tensor:
    """Bradley-Terry：-log σ(r_c - r_r)，返回 batch 均值。"""

# minidpo/dpo.py
def dpo_loss(policy_chosen_logps: Tensor, policy_rejected_logps: Tensor,
             ref_chosen_logps: Tensor, ref_rejected_logps: Tensor,
             beta: float = 0.1) -> tuple[Tensor, Tensor, Tensor]:
    """返回 (loss 均值, chosen_implicit_reward, rejected_implicit_reward)。
    隐式 reward = beta * (policy_logp - ref_logp)（返回 detach 后的向量，
    用于监控 margin）。"""

# scripts/train_dpo.py —— 脚手架：toy 偏好对生成 + 训练循环，接线留 TODO
```

## 4. 验收标准（tests/test_dpo.py，CPU）

| 编号 | 通过条件 |
| --- | --- |
| T1 | sequence_logprob：与手工用 log_softmax gather 计算的值一致（含 off-by-one 与 prompt mask 的手算短例逐位验证）；prompt 部分确实不计入（改 prompt 内容不影响 response logp 的梯度路径除外——用数值断言 response 相同时改 prompt_lens 边界值的效果）|
| T2 | bt_loss 数学：固定输入的闭式值对拍；r_c=r_r 时 loss=log2；对 r_c 的梯度为负（升 reward）|
| T3 | dpo_loss 数学：固定四个 logp 输入，与手算闭式值一致（beta 两组）；policy==ref 时 loss=log2、margin=0 |
| T4 | **训练不变量**：tiny 模型 + 构造的 toy 偏好对训练 100 步：隐式 reward margin 单调趋势上升（首尾比较 >0）、BT 准确率（margin>0 的比例）上升到 >0.9；**reference model 参数逐位不变** |
| T5 | ref 冻结检查：dpo 训练路径中 ref 的 forward 在 no_grad 下（ref 参数 .grad 全 None）|
| T6 | RM 训练：RewardModel 在同一 toy 偏好上 100 步内 BT 准确率 >0.9；取分位置正确（右 pad 变长不改变取到的 token）|

## 5. 产物

- `minidpo/*.py` 全绿 + toy 训练曲线
- `docs/5.1-dpo/POSTMORTEM.md`（含 DPO 闭式推导手写版）
