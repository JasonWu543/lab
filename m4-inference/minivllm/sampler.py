"""sampler.py — 温度 / top-p 采样（Phase 4.0，学生实现文件）

U3 任务（与 engine.py 同步）：
  实现 sample()，让 T6 全绿。

────────────────────────────────────────────────────────────────────────────────
接口约定：

  sample(logits, seqs, generator=None) → list[int]
    logits   : Tensor [len(seqs), vocab_size]，各 seq 对应的最后一步 logit
    seqs     : list[Sequence]，每个 seq 携带自己的 temperature 和 top_p
    generator: torch.Generator | None，用于可复现采样
    返回     : list[int]，长度 = len(seqs)，每个元素是采样的 token id

────────────────────────────────────────────────────────────────────────────────
算法步骤（每条 seq 独立）：

  1. temperature == 0.0  →  argmax（greedy），不用 generator
  2. temperature > 0.0   →  logit / temperature → softmax → probs
  3. top_p < 1.0         →  nucleus sampling：
       a. 按概率降序排列
       b. 累积概率超过 top_p 的 token 概率置 0
          （保留至少 1 个 token：用 cumsum - sorted_probs > top_p 作 mask）
       c. 重新归一化
       d. scatter 回原始词汇表顺序
  4. torch.multinomial(probs, 1, generator=generator)

────────────────────────────────────────────────────────────────────────────────
引导问题：

  Q1. top-p 截断用 "cumsum - sorted_probs > top_p" 而非 "cumsum > top_p"，
      多减一个 sorted_probs 的目的是什么？

  Q2. 多次调用 sample 时，若 generator 是同一个对象，结果会相同吗？
      测试的"固定 generator 可复现"要求如何满足？

────────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import torch
from torch import Tensor

from .request import Sequence


def sample(
    logits: Tensor,
    seqs: list[Sequence],
    generator: torch.Generator | None = None,
) -> list[int]:
    """按每个 seq 的 temperature/top_p 独立采样。返回长度 = len(seqs) 的 token id 列表。"""
    # TODO（U3）
    raise NotImplementedError
