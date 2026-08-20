"""FP8 格式、缩放与 amax 历史。"""

import torch


def fp8_finfo(fmt: str) -> tuple[float, float]:
    """返回给定 FP8 格式的最大有限值与最小正规数。

    想一想：指数偏置如何决定最小正规数？E4M3FN 的最高指数编码与
    IEEE 风格 E5M2 有什么不同？
    """
    raise NotImplementedError("请从 exponent/mantissa 位宽与偏置推导两个边界")


def quantize_fp8(t: torch.Tensor, fmt: str, scale: float) -> torch.Tensor:
    """按给定 scale 将张量饱和量化为 FP8 存储张量。

    想一想：为什么必须在 cast 前处理格式边界？scale 应满足哪些条件？
    """
    raise NotImplementedError("请实现 scale、饱和与 FP8 cast")


def dequantize_fp8(q: torch.Tensor, scale: float) -> torch.Tensor:
    """将 FP8 存储张量反量化为 fp32。"""
    raise NotImplementedError("请撤销量化时的缩放")


def compute_scale(amax: float, fmt: str, margin: int = 0) -> float:
    """用 amax 计算二次幂缩放因子。

    想一想：哪些 amax 没有可用的幅值信息？向下取整保证了什么？
    """
    raise NotImplementedError("请按冻结公式计算 scale 并处理异常幅值")


class AmaxHistory:
    """维护 delayed scaling 所需的有限窗口幅值历史。"""

    def __init__(self, window: int = 16):
        raise NotImplementedError("请建立固定长度窗口并验证 window")

    def update(self, t: torch.Tensor) -> None:
        """记录当前张量的绝对值最大值。"""
        raise NotImplementedError("请计算并写入当前 amax")

    def scale(self, fmt: str, margin: int = 0) -> float:
        """根据窗口内最大 amax 返回 delayed scale。"""
        raise NotImplementedError("请区分空历史并复用 scale 契约")
