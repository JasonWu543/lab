"""minigrpo/rollout.py — 学生实现：每 prompt 采 G 条补全的 rollout。

接口（已冻结）：
    @torch.no_grad()
    rollout(model, tok, prompts, G, max_new_tokens, temperature, generator=None)
    -> dict with keys: "input_ids", "prompt_lens", "completions", "old_logps", "mask"
    B = len(prompts) * G

实现框架（脚手架，非核心算法）：

步骤：
    1. 把每个 prompt 编码并复制 G 次，得到 B = len(prompts)*G 条序列。
    2. 自回归采样循环：每步对每条序列取最后位置的 logits，
       按 temperature softmax 后用 torch.multinomial 采样（传入 generator）；
       生成 eos_id 则标记为 done。
    3. 右 pad 到统一长度，构造 mask（只在 completion 区域，即 [plen, seq_end) 为 1）。
    4. 记录 old_logps：思考——应在 rollout 采样完成后（模型参数不变时）计算，
       还是在训练更新后？（时机不对会导致 off-policy 偏差）

辅助函数 _per_token_logp(model, input_ids) -> (B, T)：
    位置 t 存 token_t 的 logp（由位置 t-1 的 logits 预测），位置 0 置 0。
    注意 shift：logits[:,:-1,:] 对应 target input_ids[:,1:]。

tok（ByteTokenizer）接口：
    tok.encode(str) -> list[int]
    tok.decode(list[int]) -> str
    tok.eos_id = 258
    tok.pad_id = 258
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


@torch.no_grad()
def rollout(model, tok, prompts: list[str], G: int, max_new_tokens: int,
            temperature: float, generator=None) -> dict:
    """每个 prompt 采 G 条补全，返回 rollout 记账字典。

    TODO: 学生实现。
    返回格式：
    {
        "input_ids":   (B, T) 右pad,
        "prompt_lens": (B,),
        "completions": list[str]，长度 B,
        "old_logps":   (B, T) per-token logp（rollout 时策略，no_grad 计算）,
        "mask":        (B, T) completion token 位置为 1,
    }
    """
    raise NotImplementedError("rollout 尚未实现，请完成 TODO")


@torch.no_grad()
def _per_token_logp(model, input_ids: Tensor) -> Tensor:
    """返回 (B, T) per-token logp，位置 0 置 0。

    TODO: 学生实现。
    提示：位置 t 处存 token_t 的 logp（由 logits[:,t-1,:] 预测），
    用 log_softmax + gather 计算。
    """
    raise NotImplementedError("_per_token_logp 尚未实现，请完成 TODO")
