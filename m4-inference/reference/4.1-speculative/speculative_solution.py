"""
Phase 4.1 参考答案 — 投机解码（Speculative Decoding）

自包含实现，不依赖 minivllm 其他文件。
仅供卡住 30 分钟后参考；看完请能指出自己版本差在哪。

核心数学（Leviathan et al., 2023 / Chen et al., 2023）：
  接受概率：  α_i = min(1, p(x) / q(x))
  拒绝后重采  从 norm(max(0, p − q)) 重采样
  联合保证输出分布恰好等于 target 分布 p。
"""

from __future__ import annotations
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from torch import Tensor
from typing import Optional


@dataclass
class SpecStats:
    proposed: int = 0          # draft 提出的 token 总数
    accepted: int = 0          # 被接受的 token 总数
    target_forwards: int = 0
    draft_forwards: int = 0

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.proposed if self.proposed > 0 else 0.0


# ──────────────────────── 采样工具 ────────────────────────

def _apply_temperature(logits: Tensor, temperature: float) -> Tensor:
    """temperature scaling；temperature=0 返回 one-hot (argmax)。"""
    if temperature == 0.0:
        # greedy：返回尖峰分布（one-hot），后续 multinomial 会拿到 argmax
        idx = logits.argmax(dim=-1, keepdim=True)
        result = torch.zeros_like(logits)
        result.scatter_(-1, idx, 1.0)
        return result
    return torch.softmax(logits / temperature, dim=-1)


def _apply_top_p(probs: Tensor, top_p: float) -> Tensor:
    """nucleus filtering；top_p=1.0 直接返回原始 probs（不重新归一化）。"""
    if top_p >= 1.0:
        return probs
    # 降序排列，计算累积概率
    sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
    cumsum = sorted_probs.cumsum(dim=-1)
    # 从累积概率中去掉刚好超出 top_p 的部分
    # 保留满足 cumsum - prob <= top_p 的位置（保证至少保留概率最大的那个）
    mask = (cumsum - sorted_probs) >= top_p
    sorted_probs = sorted_probs.masked_fill(mask, 0.0)
    # 还原到原来的顺序
    result = torch.zeros_like(probs)
    result.scatter_(-1, sorted_idx, sorted_probs)
    # 重新归一化
    result = result / result.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    return result


def _get_probs(logits: Tensor, temperature: float, top_p: float) -> Tensor:
    """logits -> 处理后的概率分布（与 SPEC 分布约定一致）。"""
    probs = _apply_temperature(logits, temperature)
    probs = _apply_top_p(probs, top_p)
    return probs


def _sample(probs: Tensor, generator: Optional[torch.Generator] = None) -> Tensor:
    """从概率向量采样单个 token，返回标量 Tensor。"""
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)


# ──────────────────────── 主函数 ────────────────────────

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
    """
    投机解码主循环。

    每轮流程：
      1. draft 自回归跑 k 步，收集 k 个候选 token 及其分布 q
      2. target 一次前向验证 k+1 个位置的分布 p
      3. 逐位置 accept/reject：
         - 以 min(1, p/q) 的概率接受
         - 拒绝时从 norm(max(0, p−q)) 重采样，截断后续 draft token
      4. 若前 k 个全部接受，额外白得第 k+1 个 bonus token（来自 target 分布）
    """
    stats = SpecStats()
    generated = input_ids.clone()   # (1, current_len)
    device = input_ids.device
    new_tokens_count = 0

    while new_tokens_count < max_new_tokens:
        # 本轮实际能 draft 的步数（不超出 max_new_tokens）
        k_this = min(k, max_new_tokens - new_tokens_count)

        # ── 阶段 1：draft 自回归采 k 个候选 token ──────────────────
        draft_tokens = []     # list of scalar Tensor, length k_this
        draft_probs = []      # list of (1, V) Tensor, length k_this

        draft_context = generated.clone()  # (1, cur_len)
        for _ in range(k_this):
            d_out = draft(draft_context)
            d_logits = d_out.logits[:, -1, :]          # (1, V)
            d_probs = _get_probs(d_logits, temperature, top_p)  # (1, V)
            d_tok = _sample(d_probs, generator)         # (1,)
            draft_tokens.append(d_tok)
            draft_probs.append(d_probs)
            draft_context = torch.cat([draft_context, d_tok.unsqueeze(-1)], dim=1)
            stats.draft_forwards += 1

        # ── 阶段 2：target 一次前向验 k_this+1 个位置 ──────────────
        # 把 draft 猜的 token 拼到 context 后面送给 target
        target_input = draft_context                   # (1, cur_len + k_this)
        t_out = target(target_input)
        # 取从原序列末尾开始的 k_this+1 个位置的 logits
        cur_len = generated.shape[1]
        # 位置 cur_len-1 对应"下一个 token"（即 draft_tokens[0] 的预测位置）
        t_logits_all = t_out.logits[:, cur_len - 1: cur_len + k_this, :]  # (1, k_this+1, V)
        stats.target_forwards += 1

        # ── 阶段 3：逐位置 accept/reject ────────────────────────────
        num_accepted = 0
        bonus_tok: Optional[Tensor] = None

        for i in range(k_this):
            x_i = draft_tokens[i]                      # (1,) token id
            q_i = draft_probs[i]                       # (1, V)
            p_logits_i = t_logits_all[:, i, :]         # (1, V)
            p_i = _get_probs(p_logits_i, temperature, top_p)  # (1, V)

            # 接受概率 α = min(1, p(x) / q(x))
            p_xi = p_i[0, x_i[0]].item()
            q_xi = q_i[0, x_i[0]].item()
            # x_i was sampled from q_i, so q_xi is positive.  Adding epsilon
            # here would bias acceptance for small but representable masses.
            alpha = min(1.0, p_xi / q_xi) if q_xi > 0.0 else 1.0

            # 采样接受/拒绝
            u = torch.rand(1, generator=generator).item()
            stats.proposed += 1

            if u < alpha:
                # 接受该 token
                generated = torch.cat([generated, x_i.unsqueeze(-1)], dim=1)
                new_tokens_count += 1
                stats.accepted += 1
                num_accepted += 1

                if new_tokens_count >= max_new_tokens:
                    break
            else:
                # 拒绝：从残差分布 norm(max(0, p−q)) 重采样
                residual = torch.clamp(p_i - q_i, min=0.0)           # (1, V)
                residual_sum = residual.sum(dim=-1, keepdim=True)
                if residual_sum.item() <= 0.0:
                    # 极端情况：p 完全被 q 覆盖，从 p 重采
                    residual = p_i
                else:
                    residual = residual / residual_sum
                new_tok = _sample(residual, generator)
                generated = torch.cat([generated, new_tok.unsqueeze(-1)], dim=1)
                new_tokens_count += 1
                # 拒绝后截断：退出本轮
                break
        else:
            # for-else：k_this 个全接受 → 拿 bonus token
            if new_tokens_count < max_new_tokens:
                p_bonus_logits = t_logits_all[:, k_this, :]          # (1, V)
                p_bonus = _get_probs(p_bonus_logits, temperature, top_p)
                bonus_tok = _sample(p_bonus, generator)
                generated = torch.cat([generated, bonus_tok.unsqueeze(-1)], dim=1)
                new_tokens_count += 1

    return generated, stats
