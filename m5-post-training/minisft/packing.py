"""minisft/packing.py — Greedy bin-packing for SFT.

学生任务：实现 pack_examples。
"""
from __future__ import annotations

import torch


def pack_examples(
    examples: list[tuple[list[int], list[int]]],
    seq_len: int,
    pad_id: int,
) -> list[dict]:
    """
    将 tokenized 样本贪心地打包成定长 bin，返回模型可直接消费的 batch dict 列表。

    打包规则：
    - 不拆分单个样本（要么整个放入当前 bin，要么开新 bin）
    - 若单个样本长度超过 seq_len，直接截断到 seq_len
    - 每个 bin 用 pad_id 补齐到 seq_len，pad 位置 labels=-100

    返回列表，每个元素是一个 dict：
        "input_ids":      LongTensor  shape (S,)
        "labels":         LongTensor  shape (S,)
        "attention_mask": BoolTensor  shape (1, S, S)
        "doc_spans":      list[(start, end)]  — end 是 exclusive

    attention_mask 语义：
        mask[0, i, j] = True  iff  i 和 j 属于同一文档 AND j <= i
        （文档内因果注意力；跨文档位置相互屏蔽；pad 区域全 False）

    提示：
    ① 贪心装箱：维护一个当前 bin，尝试放入下一个样本；放不下就关闭当前 bin 开新的。
    ② attention_mask 的形状是 (1, S, S)，可以先建 zeros(bool)，再逐文档填块。
    ③ 如何知道每个位置属于哪个文档？doc_spans 告诉你每个文档的 [start, end)，
      可以给每个 token 分配一个 doc_id，然后向量化判断 same_doc 和 causal。

    Args:
        examples: list of (input_ids, labels)，两者均为 list[int]
        seq_len:  目标序列长度 S
        pad_id:   padding token id

    Returns:
        list of dict（见上）
    """
    # TODO: 实现贪心装箱 + attention_mask 构建
    #
    # 骨架（伪代码）：
    # ─────────────────────────────────────────
    # 1. 截断超长样本
    # 2. 贪心装箱：
    #    bins = []
    #    cur_bin, cur_len = [], 0
    #    for each example:
    #        if cur_bin and cur_len + len(example) > seq_len:
    #            bins.append(cur_bin); cur_bin = []; cur_len = 0
    #        cur_bin.append(example); cur_len += len(example)
    #    if cur_bin: bins.append(cur_bin)
    # 3. 对每个 bin：
    #    a. 拼接 input_ids / labels，记录 doc_spans
    #    b. 补齐 pad
    #    c. 构建 (1, S, S) bool attention_mask
    # ─────────────────────────────────────────
    raise NotImplementedError("pack_examples: 你来实现")
