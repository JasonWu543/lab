# SPEC — Phase 5.2: GRPO / Online RL（可验证任务）

> 状态：FROZEN（接口已冻结）
> 模式：Foundation —— group advantage、clipped surrogate、KL 正则全部手写
> 基座：官方 transformers tiny 模型；任务用可验证的算术格式任务
> 算力：correctness 本地；真实 GRPO 一次 M 级（W11）
> 工期：约 1 周

## 1. 问题

在可自动判分的任务上实现 GRPO 闭环：rollout（每 prompt 采 G 个）→
组内标准化 advantage（无 critic）→ clipped surrogate + KL 正则 →
策略更新。验收的核心是 reward 上升且能检测 reward hacking。

学完必须能回答（写进 POSTMORTEM）：
- GRPO 用组内均值替代 critic 的 baseline，方差和偏差各付出什么代价？
  全组同分（std=0）时怎么处理、为什么？
- clip 的作用机制？ratio 超界时梯度发生什么？
- KL 正则和 clip 都在限制策略偏移，各限制的是什么？只留一个会怎样？
- rollout freshness：用陈旧 rollout 更新多步，什么信号告诉你该停？

## 2. 范围与非目标

范围：单卡、rollout 与训练交替（不并发）、token-level clipped surrogate、
k3 KL 估计、格式+正确性两级 reward、hacking 检查脚本。
非目标：不做 vLLM 加速 rollout、不做 critic/GAE（PPO 路线）、
不做 process reward。

## 3. 任务（冻结）：两位数加法

prompt：`"Q: {a}+{b}=\nA:"`（a,b ∈ [10,99]）；期望补全：` {a+b}` 后接 EOS。
reward：答案数值正确 = 1.0；格式对（是数字）但值错 = 0.1；其余 = 0.0。
tokenizer 复用 5.0 的 ByteTokenizer（`<|pad|>` 作 EOS 用，id=258）。

## 4. 冻结接口（minigrpo/）

```python
# minigrpo/reward.py
def parse_answer(completion: str) -> int | None:
    """从补全里解析首个整数；解析失败返回 None。"""
def reward_fn(prompt: str, completion: str) -> float:
    """按上面的两级 reward 冻结定义打分。"""

# minigrpo/advantage.py
def group_advantages(rewards: Tensor,        # (n_prompts, G)
                     eps: float = 1e-6) -> Tensor:
    """组内标准化：(r - mean_g) / (std_g + eps)，逐组计算。
    全组同分（std=0）时该组 advantage 全 0。返回 (n_prompts, G)。"""

# minigrpo/loss.py
def grpo_loss(logps: Tensor,                 # (B, T) 当前策略 per-token logp
              old_logps: Tensor,             # (B, T) rollout 时的策略
              ref_logps: Tensor,             # (B, T) 冻结参考策略
              advantages: Tensor,            # (B,)   序列级，广播到 token
              mask: Tensor,                  # (B, T) completion token 为 1
              clip_eps: float = 0.2,
              kl_coef: float = 0.04) -> tuple[Tensor, dict]:
    """token-level clipped surrogate + k3 KL：
    loss = -E[min(ratio*A, clip(ratio,1±eps)*A)] + kl_coef * E[k3]
    k3 = exp(ref-logp) - (ref-logp) - 1（无偏且非负的 KL 估计）。
    按 mask 内 token 平均。返回 (loss, {"pg_loss", "kl", "clip_frac"})。"""

# minigrpo/rollout.py
@torch.no_grad()
def rollout(model, tok, prompts: list[str], G: int, max_new_tokens: int,
            temperature: float, generator=None) -> dict:
    """每个 prompt 采 G 条补全。返回
    {"input_ids": (B,T) 右pad, "prompt_lens": (B,), "completions": list[str],
     "old_logps": (B,T), "mask": (B,T)}，B = len(prompts)*G。"""

# scripts/train_grpo.py —— 完整训练循环脚手架（rollout→adv→若干 epoch 更新），
#   含 reward 曲线 jsonl 与 hacking 检查（格式 reward 占比监控），接线留 TODO
```

## 5. 验收标准（tests/test_grpo.py，CPU）

| 编号 | 通过条件 |
| --- | --- |
| T1 | reward_fn：正确答案/格式对值错/乱码 三级打分逐例对拍；parse 边界（负号、前导空格、多个数字取首个）|
| T2 | group_advantages：手算对拍（含 std=0 组→全 0）；每组均值 ≈0、std ≈1 |
| T3 | grpo_loss 数学：ratio=1 时 pg_loss = -mean(A·mask)/…的闭式对拍；policy==ref 时 kl≈0；clip_frac 在构造的超界 ratio 下 >0 且梯度确实被截（超界 token 的 dloss/dlogp = 0，用 autograd 断言）|
| T4 | k3 KL 非负性：随机 logp 对，k3 逐元素 ≥0；ref==policy 时 =0 |
| T5 | rollout 记账：mask 恰好覆盖 completion（含 EOS 前）；old_logps 与用模型重算的一致（no_grad、同 ids）；G 条/每 prompt |
| T6 | **端到端收敛（bandit 级）**：把词表缩到个位数加法（a,b∈[1,4]、答案 ≤8）+ tiny 模型先 SFT 到会输出数字格式，再 GRPO 训练 ≤60 轮：平均 reward 显著上升（首 10 轮均值 → 末 10 轮均值提升 >0.3）且全程无 NaN |

T6 允许 ≤120s；若 tiny 模型不稳可在测试内先做 20 步格式预热（脚手架提供）。

## 6. 产物

- `minigrpo/*.py` 全绿 + 训练曲线 + hacking 检查记录
- `docs/5.2-grpo/POSTMORTEM.md`（含 k3 估计为什么无偏的推导）
