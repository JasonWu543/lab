# SPEC — Phase 4.1: 投机解码（speculative decoding）

> 状态：FROZEN（接口已冻结）
> 模式：Foundation —— acceptance-rejection 的分布校正是核心手写内容
> 基座：官方 transformers（测试用两个 tiny 随机 Qwen2Config：draft 更小）
> 算力：correctness 本地；加速比 benchmark M 级（真模型对，如 0.5B draft + 3B target 可选）
> 工期：约 0.5 周（W7 主线）

## 1. 问题

实现 Leviathan/Chen 版投机解码：draft 模型先猜 k 个 token，target 模型
一次前向验证，acceptance-rejection 保证**输出分布与 target 单独采样严格一致**
（无损性），并测量 acceptance rate 与加速边界。

学完必须能回答（写进 POSTMORTEM）：
- 为什么接受概率是 min(1, p/q)？拒绝后为什么从 norm(max(0, p−q)) 重采样？
  合起来为什么恰好等于从 p 采样？（写出完整证明）
- acceptance rate 由什么决定？temperature 升高时它怎么变？
- 什么情况下投机反而更慢？（draft 开销、k 的选择、batch 场景）

## 2. 范围与非目标

范围：单序列（batch=1）、固定 k、temperature/top-p 支持、统计记录。
非目标：不做 tree speculation（Medusa/EAGLE 记 backlog）、不做动态 k、
不做与 4.0 engine 的集成（独立函数）。

## 3. 冻结接口（minivllm/speculative.py）

```python
@dataclass
class SpecStats:
    proposed: int = 0        # draft 提出的 token 总数
    accepted: int = 0        # 被接受的 token 总数
    target_forwards: int = 0
    draft_forwards: int = 0
    @property
    def acceptance_rate(self) -> float: ...

@torch.no_grad()
def speculative_generate(
    target, draft,                      # 两个 HF CausalLM
    input_ids: Tensor,                  # (1, T)
    max_new_tokens: int,
    k: int = 4,                         # 每轮 draft 猜的 token 数
    temperature: float = 1.0,
    top_p: float = 1.0,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, SpecStats]:
    """返回 (1, T+新token) 与统计。
    分布约定：p/q 都是各自 logits 经同样的 temperature+top-p 处理后的分布
    （top-p 在校正之前施加，保证与 target 单独采样时的分布定义一致）。
    temperature=0 退化为 greedy：输出必须与 target 单独 greedy 完全一致。
    每轮：draft 自回归猜 k 个 → target 一次前向算 k+1 个位置的分布 →
    逐位置 accept/reject → 全接受时额外白得 1 个 bonus token。"""
```

## 4. 验收标准（tests/test_speculative.py，tiny 模型 CPU）

| 编号 | 通过条件 |
| --- | --- |
| T1 | **无损性（greedy）**：temperature=0 时输出与 target 单独 greedy 逐 token 一致（多组 prompt/k）|
| T2 | **无损性（采样）**：vocab≤64 的 tiny 模型上，spec 采样第一个新 token 的经验分布与 target 直接采样的经验分布 TV 距离 < 0.05（各 ≥5000 样本，固定种子）|
| T3 | 校正数学（无模型单元测）：给定固定 p/q 向量模拟 accept/reject 流程 50k 次，接受率 ≈ Σ min(p,q)，拒绝后样本的经验分布 ≈ norm(max(0, p−q))（各容差 0.02）|
| T4 | 统计记账：accepted ≤ proposed；全等模型（draft=target）时 acceptance_rate > 0.99；bonus token 计数正确（生成数 = accepted + 拒绝轮数 + bonus 数）|
| T5 | 边界：max_new_tokens 精确截断（不多生成）；k=1 与 k=8 输出都正确 |

benchmark（不进测试）：`benchmarks/bench_spec.py` —— 不同 temperature 下
acceptance rate 曲线、不同 k 的实际加速比、找出「更慢」的配置。

## 5. 产物

- `minivllm/speculative.py` 全绿
- `docs/4.1-speculative/POSTMORTEM.md`（含无损性证明手写版）
