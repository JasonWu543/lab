"""低精度参数更新与 FP8 线性层模拟。"""

from .fp8 import AmaxHistory


def master_weight_sgd_step(params_lp, master_fp32, grads_lp, lr) -> None:
    """用 fp32 master 权重完成一步 SGD，再同步低精度参数。

    想一想：小于低精度 ULP 的单步更新怎样在 master 副本中累积？
    """
    raise NotImplementedError("请在无梯度跟踪下更新 master 并回写参数")


def fp8_linear_forward(x, w, x_hist: AmaxHistory, w_hist: AmaxHistory):
    """以 delayed E4M3 量化/反量化模拟 fp32 线性乘法。

    想一想：当前张量的 amax 应在本次 scale 读取之前还是之后写入历史？
    """
    raise NotImplementedError("请分别处理激活与权重，再执行 fp32 matmul")
