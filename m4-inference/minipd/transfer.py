"""KV cache 迁移成本模型。"""

from dataclasses import dataclass


@dataclass
class TransferModel:
    """固定启动成本、每 token 字节量和每 tick 带宽。"""

    base_ticks: int
    bytes_per_token: int
    bandwidth_bytes_per_tick: int


def transfer_ticks(model: TransferModel, num_tokens: int) -> int:
    """计算迁移 ``num_tokens`` 个 token 所需的整数 tick。

    想一想：不能整除的最后一小段数据是否仍要占用完整 tick？零 token
    时，固定启动成本还在吗？
    """
    raise NotImplementedError("请实现 KV cache 迁移耗时的向上取整计算")
