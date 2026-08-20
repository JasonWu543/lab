"""参考答案 5.2 — rollout（每 prompt 采 G 条）。"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


@torch.no_grad()
def rollout(model, tok, prompts, G: int, max_new_tokens: int,
            temperature: float, generator=None) -> dict:
    """每个 prompt 采 G 条补全。返回
    {"input_ids": (B,T) 右pad, "prompt_lens": (B,), "completions": list[str],
     "old_logps": (B,T), "mask": (B,T)}，B = len(prompts)*G。
    """
    model.eval()
    device = next(model.parameters()).device
    eos_id = tok.eos_id
    pad_id = tok.pad_id

    # 展开成 B 条：每个 prompt 复制 G 次
    prompt_ids = [tok.encode(p) for p in prompts]
    flat_prompts = []
    flat_prompt_lens = []
    for pid in prompt_ids:
        for _ in range(G):
            flat_prompts.append(list(pid))
            flat_prompt_lens.append(len(pid))
    B = len(flat_prompts)

    # 逐条自回归采样（tiny 模型，简单实现即可）
    seqs = [list(p) for p in flat_prompts]
    gen_tokens = [[] for _ in range(B)]
    done = [False] * B

    for _ in range(max_new_tokens):
        # 找最长当前序列做一个 batch step（这里逐条以简化 mask 处理）
        for b in range(B):
            if done[b]:
                continue
            ids = torch.tensor(seqs[b], device=device).unsqueeze(0)
            logits = model(input_ids=ids).logits[0, -1, :]  # (V,)
            if temperature > 0:
                probs = F.softmax(logits / temperature, dim=-1)
                nxt = torch.multinomial(probs, 1, generator=generator).item()
            else:
                nxt = int(logits.argmax().item())
            seqs[b].append(nxt)
            gen_tokens[b].append(nxt)
            if nxt == eos_id:
                done[b] = True

    # 右 pad 到统一长度
    max_len = max(len(s) for s in seqs)
    input_ids = torch.full((B, max_len), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros((B, max_len), dtype=torch.long, device=device)
    completions = []
    prompt_lens = torch.tensor(flat_prompt_lens, dtype=torch.long, device=device)

    for b in range(B):
        s = seqs[b]
        input_ids[b, :len(s)] = torch.tensor(s, device=device)
        plen = flat_prompt_lens[b]
        # completion token（含 EOS）位置：[plen, len(s))
        mask[b, plen:len(s)] = 1
        completions.append(tok.decode(gen_tokens[b]))

    # old_logps：no_grad 下用同一模型重算 per-token logp（对齐 shift）
    old_logps = _per_token_logp(model, input_ids)  # (B, T)

    return {
        "input_ids": input_ids,
        "prompt_lens": prompt_lens,
        "completions": completions,
        "old_logps": old_logps,
        "mask": mask,
    }


@torch.no_grad()
def _per_token_logp(model, input_ids: Tensor) -> Tensor:
    """返回 (B, T) 的 per-token logp，位置 t 处存 token t 的 logp（由 t-1 预测）。
    位置 0 无预测者，置 0。"""
    logits = model(input_ids=input_ids).logits          # (B, T, V)
    logp = F.log_softmax(logits[:, :-1, :], dim=-1)      # (B, T-1, V)
    tok_logp = logp.gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)  # (B, T-1)
    B = input_ids.shape[0]
    out = torch.zeros_like(input_ids, dtype=tok_logp.dtype)
    out[:, 1:] = tok_logp
    return out
