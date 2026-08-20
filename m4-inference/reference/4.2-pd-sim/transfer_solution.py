"""Phase 4.2 transfer 参考答案。"""

from dataclasses import dataclass


@dataclass
class TransferModel:
    base_ticks: int
    bytes_per_token: int
    bandwidth_bytes_per_tick: int


def transfer_ticks(model: TransferModel, num_tokens: int) -> int:
    """返回固定成本加带宽成本；整数式实现 ceil，避免浮点误差。"""
    if num_tokens < 0 or model.base_ticks < 0 or model.bytes_per_token < 0:
        raise ValueError("costs and token count must be non-negative")
    if model.bandwidth_bytes_per_tick <= 0:
        raise ValueError("bandwidth_bytes_per_tick must be positive")
    payload = num_tokens * model.bytes_per_token
    return model.base_ticks + (payload + model.bandwidth_bytes_per_tick - 1) // model.bandwidth_bytes_per_tick
