"""
Phase 1.2：自回归生成
"""
import torch
from typing import Optional
from .model import KVCache, MiniLM


def sample_next(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: float = 1.0,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    从 logits 采样下一个 token。

    logits: (B, vocab_size)
    返回: (B,) token ids

    实现步骤（按顺序）：
      1. temperature == 0 → 直接 argmax（贪心）
      2. logits /= temperature，softmax 得到 probs
      3. top_p < 1.0 时做 nucleus sampling：
         a. 对 probs 降序排列，得 sorted_probs, sorted_idx
         b. 计算 cumsum；将 (cumsum - sorted_probs) > top_p 的位置置 0
         c. 重新归一化后 multinomial 采样；用 gather 恢复到原始词表 idx
      4. top_p == 1.0 → 直接 multinomial(probs, 1)
    """
    raise NotImplementedError


def generate(
    model: MiniLM,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    use_cache: bool = True,
    eos_token_id: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    自回归生成。

    input_ids: (B, T_in)
    返回: (B, T_in + T_new)，T_new <= max_new_tokens

    提示：
      - use_cache=True：创建 KVCache，先 prefill，再逐步解码（每次只喂 1 token）
      - use_cache=False：每步把全部序列喂入模型（较慢，用于对比验证）
      - 遇到 eos_token_id 且全 batch 都输出时提前退出
    """
    raise NotImplementedError
