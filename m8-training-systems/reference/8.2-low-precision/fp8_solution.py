"""FP8 格式与 delayed scaling 参考答案。"""

from collections import deque
import math

import torch


def _layout(fmt):
    if fmt == "e4m3":
        return 4, 3, 7, torch.float8_e4m3fn
    if fmt == "e5m2":
        return 5, 2, 15, torch.float8_e5m2
    raise ValueError("fmt must be 'e4m3' or 'e5m2'")


def fp8_finfo(fmt: str) -> tuple[float, float]:
    """从位布局闭式推导；不依赖 torch.finfo。

    smallest normal = 2^(1-bias)。E5M2 保留全 1 指数给 inf/nan，最大
    有限指数为 2^e-2；E4M3FN 无 inf，最高指数仍表示有限数，但最高尾数
    编码留给 NaN，因此 significand 最大为 1+6/8。
    """
    exponent_bits, mantissa_bits, bias, _ = _layout(fmt)
    smallest_normal = 2.0 ** (1 - bias)
    if fmt == "e4m3":
        max_exponent = (2**exponent_bits - 1) - bias
        significand = 1.0 + (2**mantissa_bits - 2) / 2**mantissa_bits
    else:
        max_exponent = (2**exponent_bits - 2) - bias
        significand = 2.0 - 2.0**(-mantissa_bits)
    return math.ldexp(significand, max_exponent), smallest_normal


def quantize_fp8(t: torch.Tensor, fmt: str, scale: float) -> torch.Tensor:
    maximum, _ = fp8_finfo(fmt)
    _, _, _, dtype = _layout(fmt)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")
    # E4M3FN 超界 cast 为 NaN、E5M2 超界 cast 为 inf，所以必须先饱和。
    return (t * scale).clamp(-maximum, maximum).to(dtype)


def dequantize_fp8(q: torch.Tensor, scale: float) -> torch.Tensor:
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")
    return q.float() / scale


def compute_scale(amax: float, fmt: str, margin: int = 0) -> float:
    maximum, _ = fp8_finfo(fmt)
    if isinstance(margin, bool) or not isinstance(margin, int) or margin < 0:
        raise ValueError("margin must be a non-negative integer")
    if not math.isfinite(amax) or amax <= 0:
        return 1.0
    exponent = math.floor(math.log2(maximum / amax))
    return math.ldexp(1.0, exponent - margin)


class AmaxHistory:
    def __init__(self, window: int = 16):
        if not isinstance(window, int) or isinstance(window, bool) or window <= 0:
            raise ValueError("window must be a positive integer")
        self.window = window
        self._values = deque(maxlen=window)

    def update(self, t: torch.Tensor) -> None:
        if not isinstance(t, torch.Tensor):
            raise TypeError("t must be a tensor")
        self._values.append(float(t.detach().abs().max()) if t.numel() else 0.0)

    def scale(self, fmt: str, margin: int = 0) -> float:
        if not self._values:
            return 1.0
        return compute_scale(max(self._values), fmt, margin)
