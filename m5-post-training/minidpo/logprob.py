"""minidpo/logprob.py — 学生实现：sequence log-prob（带 prompt mask）。

接口（已冻结）：
    sequence_logprob(model, input_ids, prompt_lens) -> Tensor  # (B,)

实现要点（思考问题，不是直接给答案）：
    1. 调用 model(input_ids=input_ids).logits 得到 (B, T, V)。
    2. 「位置 t 的 logits 预测 token t+1」——shift 后，shift_logits[:,:-1,:]
       对应的 target 是 input_ids[:,1:]。
    3. API 链：F.log_softmax → .gather(-1, target.unsqueeze(-1)).squeeze(-1)
       → 得到 (B, T-1) 的 per-token log-prob。
    4. 问：shift 后的位置 j 对应原始序列的哪个位置？如何用 prompt_lens
       构造一个 (B, T-1) 的 boolean mask 仅选出 response token？
    5. 最后对 mask 内的 token log-prob 求和，返回 (B,)。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def sequence_logprob(model, input_ids: Tensor,   # (B, T)
                     prompt_lens: Tensor          # (B,)
                     ) -> Tensor:                 # (B,)
    """每条序列 response 部分（位置 >= prompt_len）的 log p 之和。

    TODO: 学生实现。
    提示：off-by-one 是关键——位置 t 的 logits 预测 token t+1。
    """
    raise NotImplementedError("sequence_logprob 尚未实现，请完成 TODO")
