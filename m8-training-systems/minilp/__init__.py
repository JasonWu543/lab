"""Phase 8.2：低精度训练的三个最小机制。"""

from .fp8 import AmaxHistory, compute_scale, dequantize_fp8, fp8_finfo, quantize_fp8
from .scaler import SimpleGradScaler
from .train import fp8_linear_forward, master_weight_sgd_step

__all__ = [
    "AmaxHistory", "SimpleGradScaler", "compute_scale", "dequantize_fp8",
    "fp8_finfo", "fp8_linear_forward", "master_weight_sgd_step", "quantize_fp8",
]
