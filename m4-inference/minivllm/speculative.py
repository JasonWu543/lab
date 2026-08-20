"""Phase 4.1 — 学生实现文件：投机解码（speculative decoding）

任务：实现 speculative_generate，通过 tests/test_speculative.py 全部 16 关。

核心（也是唯一的）难点是分布校正：draft 从 q 里猜了 token x，
target 认为分布应该是 p——你需要一个「接受判据」和一个「拒绝后的重采样分布」，
使得**合成的输出分布严格等于 p**（无损性）。

从这个约束出发自己推导两者（这是 POSTMORTEM 的证明作业）：
  设你以某个概率 a(x) 接受 x~q，拒绝时从某个分布 r 重采样，
  则输出分布 = q(x)·a(x) + (1 - Σ_y q(y)a(y))·r(x)，令它恒等于 p(x)，
  解出 a 和 r。自查：p == q 时应恒接受；p 与 q 完全不重叠时应恒拒绝。
  推完对照 Chen et al. 2023 (arXiv 2302.01318) 附录验证。

每轮流程（结构给定）：
  1. draft 自回归猜 k 个 token，记录每步的分布 q_i
  2. target 对「context + k 个猜测」一次前向，得到 k+1 个位置的分布 p_i
     （注意 off-by-one：哪个位置的 logits 预测第一个猜测 token？）
  3. 从 i=0 开始逐位置 accept/reject；一旦拒绝，重采样一个 token 并
     丢弃后续所有猜测，本轮结束
  4. k 个全接受 → 从第 k+1 个分布额外白得一个 bonus token
  5. 重复直到 max_new_tokens（精确截断：一个不多，一个不少）

约定（与测试一致）：
  - p/q 都是各自 logits 经同样 temperature + top-p 处理后的分布
    （工具函数已给，直接用）
  - temperature=0 为 greedy：必须与 target 单独 greedy 完全一致
  - stats.proposed 只计「被检验过」的 draft token（拒绝后丢弃的不计）
  - 所有随机性走传入的 generator（可复现性测试会抓）

运行测试：
    cd m4-inference && python3 -m pytest tests/test_speculative.py -x -q

卡住了再看：reference/4.1-speculative/speculative_solution.py（30 分钟规则）
"""
from __future__ import annotations

import torch
from dataclasses import dataclass
from torch import Tensor
from typing import Optional


@dataclass
class SpecStats:
    proposed: int = 0          # 被检验的 draft token 总数
    accepted: int = 0          # 被接受的 token 总数
    target_forwards: int = 0
    draft_forwards: int = 0

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.proposed if self.proposed > 0 else 0.0


# ──────────────────── 采样工具（给定脚手架，不需要修改）────────────────────

def _apply_temperature(logits: Tensor, temperature: float) -> Tensor:
    """temperature scaling；temperature=0 返回 one-hot（argmax）。"""
    if temperature == 0.0:
        idx = logits.argmax(dim=-1, keepdim=True)
        result = torch.zeros_like(logits)
        result.scatter_(-1, idx, 1.0)
        return result
    return torch.softmax(logits / temperature, dim=-1)


def _apply_top_p(probs: Tensor, top_p: float) -> Tensor:
    """nucleus 截断 + 重新归一化；top_p>=1 原样返回。"""
    if top_p >= 1.0:
        return probs
    sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
    cumsum = sorted_probs.cumsum(dim=-1)
    mask = (cumsum - sorted_probs) >= top_p
    sorted_probs = sorted_probs.masked_fill(mask, 0.0)
    result = torch.zeros_like(probs)
    result.scatter_(-1, sorted_idx, sorted_probs)
    return result / result.sum(dim=-1, keepdim=True).clamp(min=1e-12)


def _get_probs(logits: Tensor, temperature: float, top_p: float) -> Tensor:
    """logits → SPEC 约定的分布（先 temperature 后 top-p）。"""
    return _apply_top_p(_apply_temperature(logits, temperature), top_p)


def _sample(probs: Tensor, generator: Optional[torch.Generator] = None) -> Tensor:
    """从 (1, V) 概率向量采一个 token，返回 (1,)。"""
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)


# ─────────────────────────── 主函数（学生实现）───────────────────────────

@torch.no_grad()
def speculative_generate(
    target,
    draft,
    input_ids: Tensor,              # (1, T)
    max_new_tokens: int,
    k: int = 4,
    temperature: float = 1.0,
    top_p: float = 1.0,
    generator: Optional[torch.Generator] = None,
) -> tuple[Tensor, SpecStats]:
    """返回 (生成后的完整序列 (1, T+max_new_tokens), SpecStats)。

    提示：
      - draft/target 都是 HF CausalLM：`model(ids).logits` → (1, T, V)
      - 每轮 draft 的步数要用 max_new_tokens 剩余量截断（min(k, 剩余)）
      - 拒绝重采样时注意残差分布可能数值上全零（p 被 q 完全覆盖的极端），
        想想此时理论上意味着什么、代码上怎么兜底
      - stats 的四个计数各自在什么时机 +1，对照 T4 的断言想清楚
    """
    raise NotImplementedError("Phase 4.1 —— 学生实现")
