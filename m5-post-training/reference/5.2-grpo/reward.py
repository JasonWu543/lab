"""参考答案 5.2 — reward 函数。"""
from __future__ import annotations

import re


def parse_answer(completion: str) -> int | None:
    """从补全里解析首个整数（含可选负号）；解析失败返回 None。"""
    m = re.search(r"-?\d+", completion)
    if m is None:
        return None
    try:
        return int(m.group())
    except ValueError:
        return None


def reward_fn(prompt: str, completion: str) -> float:
    """两级 reward（冻结定义）：
    - 答案数值正确 = 1.0
    - 格式对（是数字）但值错 = 0.1
    - 其余（解析不出数字）= 0.0
    prompt 形如 "Q: {a}+{b}=\nA:"。
    """
    m = re.search(r"(-?\d+)\s*\+\s*(-?\d+)", prompt)
    if m is None:
        return 0.0
    target = int(m.group(1)) + int(m.group(2))
    pred = parse_answer(completion)
    if pred is None:
        return 0.0
    if pred == target:
        return 1.0
    return 0.1
