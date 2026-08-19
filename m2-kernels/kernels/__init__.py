"""kernels — M2 Phase 2.0 Triton fused kernel 对外接口

对外暴露三个函数，签名由 SPEC 冻结：
    rmsnorm(x, weight, eps=1e-6)
    swiglu_mul(gate, up)
    fused_cross_entropy(logits, targets, ignore_index=-100)

注意：本包依赖 triton，triton 仅在 Linux + CUDA 环境下可用。
在 Mac / CPU 机器上 import 会因缺少 triton 而 ImportError；
测试文件用 pytest.importorskip("triton") 在 module 级 skip 整个文件。
"""
from kernels.rmsnorm import rmsnorm
from kernels.swiglu import swiglu_mul
from kernels.cross_entropy import fused_cross_entropy

__all__ = ["rmsnorm", "swiglu_mul", "fused_cross_entropy"]
