"""请求数据结构（给定脚手架）。"""

from dataclasses import dataclass


@dataclass
class PDRequest:
    """一条请求的静态参数与模拟运行时状态。"""

    req_id: int
    arrival_time: int
    prompt_len: int
    max_new_tokens: int
    prefilled_tokens: int = 0
    generated_tokens: int = 0
    first_token_time: int | None = None
    finish_time: int | None = None
