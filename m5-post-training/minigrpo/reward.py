"""minigrpo/reward.py — 学生实现：reward 函数（可验证算术任务）。

接口（已冻结）：
    parse_answer(completion: str) -> int | None
    reward_fn(prompt: str, completion: str) -> float

reward_fn 两级规则（冻结）：
    - 答案数值正确 = 1.0
    - 格式对（能解析出整数）但值错 = 0.1
    - 其余（解析失败）= 0.0

prompt 格式：`"Q:{a}+{b}=\\nA:"` 或 `"Q: {a}+{b}=\\nA:"`。
需要先从 prompt 中解析出 a 和 b，再计算正确答案 target = a + b。

思考问题：
    1. parse_answer 要取「首个整数」，支持负号、前导空格、多个数字时取第一个。
       用什么正则？re.search 还是 re.findall？
    2. reward_fn 需要先从 prompt 里找到 a+b 模式，再和 completion 的解析结果比较。
       prompt 解析失败时返回什么？
"""
from __future__ import annotations

import re


def parse_answer(completion: str) -> int | None:
    """从补全里解析首个整数（含可选负号）；解析失败返回 None。

    TODO: 学生实现。
    """
    raise NotImplementedError("parse_answer 尚未实现，请完成 TODO")


def reward_fn(prompt: str, completion: str) -> float:
    """两级 reward（冻结定义）：正确=1.0，格式对值错=0.1，其余=0.0。

    TODO: 学生实现。
    """
    raise NotImplementedError("reward_fn 尚未实现，请完成 TODO")
