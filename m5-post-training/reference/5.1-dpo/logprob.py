"""参考答案 5.1 — sequence_logprob。"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def sequence_logprob(model, input_ids: Tensor, prompt_lens: Tensor) -> Tensor:
    """每条序列 response 部分（位置 >= prompt_len）的 log p 之和。

    off-by-one：位置 t 的 logits 预测 token t+1。因此对于第 t 个 target token
    （t 从 1 开始，即 input_ids[:, 1:]），其 log-prob 来自 logits[:, t-1]。
    只对 response 区域（原始位置 >= prompt_len）求和；prompt 与 pad 不计入。

    本 phase 统一右 pad；pad id 取自 model.config.pad_token_id。
    """
    pad_id = getattr(model.config, "pad_token_id", None)
    attention_mask = input_ids.ne(pad_id) if pad_id is not None else None
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits  # (B, T, V)
    # shift: logits[:, :-1] 预测 input_ids[:, 1:]
    shift_logits = logits[:, :-1, :]          # (B, T-1, V)
    shift_labels = input_ids[:, 1:]           # (B, T-1)
    logp = F.log_softmax(shift_logits, dim=-1)
    token_logp = logp.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)  # (B, T-1)

    B, Tm1 = token_logp.shape
    # position index of each target token in the *original* sequence == j+1
    pos = torch.arange(1, Tm1 + 1, device=input_ids.device).unsqueeze(0)  # (1, T-1)
    resp_mask = pos >= prompt_lens.unsqueeze(1)  # response token 位置 >= prompt_len
    if attention_mask is not None:
        resp_mask = resp_mask & attention_mask[:, 1:]
    return (token_logp * resp_mask).sum(dim=-1)  # (B,)
