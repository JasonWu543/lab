"""
Phase 1.2 参考答案：generate
学生文件：minilm/model/generate.py
"""
import torch
from typing import Optional
from .model_solution import KVCache, MiniLM


def sample_next(logits: torch.Tensor, temperature: float = 1.0, top_p: float = 1.0, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """
    从 logits 采样下一个 token。
    logits: (B, vocab_size)
    返回: (B,) token ids
    """
    if temperature == 0.0:
        return logits.argmax(dim=-1)

    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)

    if top_p < 1.0:
        sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
        cumsum = sorted_probs.cumsum(dim=-1)
        # 移除累积概率超出 top_p 的 token（但保留第一个让累积达到 top_p 的）
        remove = cumsum - sorted_probs > top_p
        sorted_probs = sorted_probs.masked_fill(remove, 0.0)
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
        next_idx = torch.multinomial(sorted_probs, 1, generator=generator)
        next_token = sorted_idx.gather(-1, next_idx).squeeze(-1)
    else:
        next_token = torch.multinomial(probs, 1, generator=generator).squeeze(-1)

    return next_token


def generate(model: MiniLM, input_ids: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_p: float = 1.0, use_cache: bool = True, eos_token_id: Optional[int] = None, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """
    自回归生成。
    input_ids: (B, T_in)
    返回: (B, T_in + T_new)
    """
    model.eval()
    result = input_ids.clone()
    if max_new_tokens <= 0:
        return result

    with torch.no_grad():
        if use_cache:
            cfg = model.cfg
            dtype = next(model.parameters()).dtype
            device = input_ids.device
            cache = KVCache(cfg, result.shape[0], cfg.max_seq_len, device, dtype)

            # Prefill
            logits = model(result, kv_cache=cache)
            next_token = sample_next(logits[:, -1, :], temperature, top_p, generator)
            result = torch.cat([result, next_token.unsqueeze(1)], dim=1)

            for _ in range(max_new_tokens - 1):
                logits = model(result[:, -1:], kv_cache=cache)
                next_token = sample_next(logits[:, -1, :], temperature, top_p, generator)
                result = torch.cat([result, next_token.unsqueeze(1)], dim=1)
                if eos_token_id is not None and (next_token == eos_token_id).all():
                    break
        else:
            for _ in range(max_new_tokens):
                logits = model(result)
                next_token = sample_next(logits[:, -1, :], temperature, top_p, generator)
                result = torch.cat([result, next_token.unsqueeze(1)], dim=1)
                if eos_token_id is not None and (next_token == eos_token_id).all():
                    break

    return result
