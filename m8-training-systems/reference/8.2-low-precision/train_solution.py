"""低精度训练参考答案。"""

import torch

from minilp.fp8 import AmaxHistory, dequantize_fp8, quantize_fp8


@torch.no_grad()
def master_weight_sgd_step(params_lp, master_fp32, grads_lp, lr) -> None:
    """在 fp32 中累积低于 fp16 ULP 的更新，再同步存储副本。"""
    if not (len(params_lp) == len(master_fp32) == len(grads_lp)):
        raise ValueError("parameter, master, and gradient collections must align")
    for parameter, master, gradient in zip(params_lp, master_fp32, grads_lp):
        master.add_(gradient.float(), alpha=-lr)
        parameter.copy_(master.to(parameter.dtype))


def fp8_linear_forward(x, w, x_hist: AmaxHistory, w_hist: AmaxHistory):
    """先读过去的 scale，后写当前 amax，使突增在下一步得到适配。"""
    x_scale = x_hist.scale("e4m3")
    w_scale = w_hist.scale("e4m3")
    x_hist.update(x)
    w_hist.update(w)
    x_hat = dequantize_fp8(quantize_fp8(x, "e4m3", x_scale), x_scale)
    w_hat = dequantize_fp8(quantize_fp8(w, "e4m3", w_scale), w_scale)
    return x_hat @ w_hat
