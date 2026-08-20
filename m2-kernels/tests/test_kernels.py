"""
Phase 2.0: Triton kernel 三件套 测试套件 (T1–T5)
所有测试仅在 NVIDIA GPU 上运行。

运行方式（需要 CUDA GPU）：
    cd m2-kernels && python3 -m pytest tests/test_kernels.py -x -q

在无 GPU 环境（Mac / CPU 机器）：
    python3 -m pytest tests/test_kernels.py -q
    → 所有测试 SKIP（collection 不报错）
"""
import gc

import pytest
import torch
import torch.nn.functional as F

# ─── Module 级 skip：无 GPU 或无 triton 时跳过整个文件 ───────────────────────
# 必须在所有 import 和测试定义之前，这样 pytest collection 阶段就直接 skip。
if not torch.cuda.is_available():
    pytest.skip("CUDA GPU 不可用，跳过全部 Triton kernel 测试", allow_module_level=True)

triton = pytest.importorskip(
    "triton",
    reason="triton 未安装（仅 Linux+CUDA 支持），跳过全部 kernel 测试",
)

# triton 可用后才 import kernels
from kernels.rmsnorm import rmsnorm
from kernels.swiglu import swiglu_mul
from kernels.cross_entropy import fused_cross_entropy


# ─── 工具函数 ─────────────────────────────────────────────────────────────────

def _ref_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """PyTorch 参考实现（fp32 内部计算）。"""
    x_f32 = x.float()
    rms = torch.sqrt(x_f32.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x_f32 / rms * weight.float()).to(x.dtype)


def _ref_swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """PyTorch 参考实现。"""
    return F.silu(gate) * up


# ─── 固定随机种子 ─────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def fix_seed():
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)


# ─── T1: RMSNorm fwd + bwd ───────────────────────────────────────────────────

class TestRMSNorm:
    """T1：rmsnorm fwd/bwd 与 torch 参考实现数值对齐。"""

    @pytest.mark.parametrize("shape,dtype,rtol", [
        # (B, T, H) 多维前缀
        ((4, 128, 512),  torch.float32, 1e-5),
        ((4, 128, 512),  torch.bfloat16, 2e-2),
        ((2, 64,  768),  torch.float32, 1e-5),
        ((2, 64,  768),  torch.bfloat16, 2e-2),
        # 非 2 幂 H
        ((8, 32,  1000), torch.float32, 1e-5),
        ((8, 32,  1000), torch.bfloat16, 2e-2),
        # 大 H
        ((2, 16,  4096), torch.float32, 1e-5),
        ((2, 16,  4096), torch.bfloat16, 2e-2),
        # 2D 输入
        ((256, 512),     torch.float32, 1e-5),
        ((256, 768),     torch.bfloat16, 2e-2),
    ])
    def test_fwd(self, shape, dtype, rtol):
        H = shape[-1]
        x = torch.randn(*shape, dtype=dtype, device="cuda")
        w = torch.randn(H, dtype=dtype, device="cuda")

        y_ref = _ref_rmsnorm(x, w)
        y_tri = rmsnorm(x, w)

        assert y_tri.shape == y_ref.shape, "输出 shape 不匹配"
        assert y_tri.dtype == dtype, f"输出 dtype 应为 {dtype}"
        torch.testing.assert_close(y_tri.float(), y_ref.float(), rtol=rtol, atol=0)

    @pytest.mark.parametrize("shape,dtype,rtol", [
        ((4, 128, 512),  torch.float32, 1e-5),
        ((4, 128, 512),  torch.bfloat16, 2e-2),
        ((8, 32,  1000), torch.float32, 1e-5),
        ((2, 16,  4096), torch.float32, 1e-5),
        ((256, 768),     torch.bfloat16, 2e-2),
    ])
    def test_bwd(self, shape, dtype, rtol):
        H = shape[-1]
        x_tri = torch.randn(*shape, dtype=dtype, device="cuda", requires_grad=True)
        w_tri = torch.randn(H, dtype=dtype, device="cuda", requires_grad=True)

        x_ref = x_tri.detach().clone().requires_grad_(True)
        w_ref = w_tri.detach().clone().requires_grad_(True)

        # Forward
        y_tri = rmsnorm(x_tri, w_tri)
        y_ref = _ref_rmsnorm(x_ref, w_ref)

        # Upstream gradient
        dy = torch.randn_like(y_tri)

        y_tri.backward(dy)
        y_ref.backward(dy)

        assert x_tri.grad is not None, "x.grad 为 None（bwd 未实现？）"
        assert w_tri.grad is not None, "w.grad 为 None（bwd 未实现？）"

        torch.testing.assert_close(
            x_tri.grad.float(), x_ref.grad.float(), rtol=rtol, atol=0,
            msg="dx 与参考不一致",
        )
        torch.testing.assert_close(
            w_tri.grad.float(), w_ref.grad.float(), rtol=rtol, atol=0,
            msg="dw 与参考不一致",
        )

    def test_noncontiguous_weight(self):
        """W_ptr 是元素指针：wrapper 必须先处理带 stride 的 weight view。"""
        N, H = 8, 768
        x_tri = torch.randn(N, H, device="cuda", requires_grad=True)
        w_storage_tri = torch.randn(2 * H, device="cuda", requires_grad=True)
        w_tri = w_storage_tri[::2]
        x_ref = x_tri.detach().clone().requires_grad_(True)
        w_storage_ref = w_storage_tri.detach().clone().requires_grad_(True)
        w_ref = w_storage_ref[::2]
        assert not w_tri.is_contiguous()

        dy = torch.randn_like(x_tri)
        y_tri = rmsnorm(x_tri, w_tri)
        y_ref = _ref_rmsnorm(x_ref, w_ref)
        torch.testing.assert_close(y_tri, y_ref, rtol=1e-5, atol=0)
        y_tri.backward(dy)
        y_ref.backward(dy)
        torch.testing.assert_close(x_tri.grad, x_ref.grad, rtol=1e-5, atol=0)
        torch.testing.assert_close(
            w_storage_tri.grad, w_storage_ref.grad, rtol=1e-5, atol=0,
        )


# 外部可调用的简短名称（与 SPEC 一致）
def test_rmsnorm():
    """T1 入口——实例化并运行 TestRMSNorm 所有用例。"""
    t = TestRMSNorm()
    for shape, dtype, rtol in [
        ((4, 128, 512), torch.float32, 1e-5),
        ((4, 128, 512), torch.bfloat16, 2e-2),
        ((8, 32, 1000), torch.float32, 1e-5),
        ((2, 16, 4096), torch.float32, 1e-5),
    ]:
        t.test_fwd(shape, dtype, rtol)
        t.test_bwd(shape, dtype, rtol)


# ─── T2: SwiGLU fwd + bwd ────────────────────────────────────────────────────

class TestSwiGLU:
    """T2：swiglu_mul fwd/bwd 与 torch 参考实现数值对齐。"""

    @pytest.mark.parametrize("shape,dtype,rtol", [
        ((4, 128, 512),  torch.float32, 1e-5),
        ((4, 128, 512),  torch.bfloat16, 2e-2),
        ((2, 64,  2048), torch.float32, 1e-5),
        ((8, 32,  1001), torch.float32, 1e-5),   # 非 2 幂
        ((256, 4096),    torch.bfloat16, 2e-2),
    ])
    def test_fwd(self, shape, dtype, rtol):
        gate = torch.randn(*shape, dtype=dtype, device="cuda")
        up   = torch.randn(*shape, dtype=dtype, device="cuda")

        y_ref = _ref_swiglu(gate, up)
        y_tri = swiglu_mul(gate, up)

        assert y_tri.shape == shape
        assert y_tri.dtype == dtype
        torch.testing.assert_close(y_tri.float(), y_ref.float(), rtol=rtol, atol=0)

    @pytest.mark.parametrize("shape,dtype,rtol", [
        ((4, 128, 512),  torch.float32, 1e-5),
        ((4, 128, 512),  torch.bfloat16, 2e-2),
        ((8, 32,  1001), torch.float32, 1e-5),
        ((256, 4096),    torch.float32, 1e-5),
    ])
    def test_bwd(self, shape, dtype, rtol):
        g_tri = torch.randn(*shape, dtype=dtype, device="cuda", requires_grad=True)
        u_tri = torch.randn(*shape, dtype=dtype, device="cuda", requires_grad=True)
        g_ref = g_tri.detach().clone().requires_grad_(True)
        u_ref = u_tri.detach().clone().requires_grad_(True)

        dy = torch.randn(*shape, dtype=dtype, device="cuda")

        swiglu_mul(g_tri, u_tri).backward(dy)
        _ref_swiglu(g_ref, u_ref).backward(dy)

        torch.testing.assert_close(
            g_tri.grad.float(), g_ref.grad.float(), rtol=rtol, atol=0, msg="d_gate 不一致",
        )
        torch.testing.assert_close(
            u_tri.grad.float(), u_ref.grad.float(), rtol=rtol, atol=0, msg="d_up 不一致",
        )

    def test_shape_mismatch_rejected(self):
        gate = torch.randn(2, 8, device="cuda")
        up = torch.randn(2, 7, device="cuda")
        with pytest.raises(AssertionError, match="形状必须相同"):
            swiglu_mul(gate, up)


def test_swiglu():
    """T2 入口。"""
    t = TestSwiGLU()
    for shape, dtype, rtol in [
        ((4, 128, 512), torch.float32, 1e-5),
        ((4, 128, 512), torch.bfloat16, 2e-2),
        ((8, 32, 1001), torch.float32, 1e-5),
    ]:
        t.test_fwd(shape, dtype, rtol)
        t.test_bwd(shape, dtype, rtol)


# ─── T3: fused_cross_entropy 数值对齐 ────────────────────────────────────────

@pytest.mark.parametrize("N,V,ignore_index,dtype,loss_atol,grad_atol", [
    (128,  1024, -100, torch.float32,  1e-5, 1e-5),
    (256,  4096, -100, torch.float32,  1e-5, 1e-5),
    (64,   1024,    0, torch.float32,  1e-5, 1e-5),
    (128,  1024, -100, torch.bfloat16, 1e-5, 2e-4),
])
def test_cross_entropy(N, V, ignore_index, dtype, loss_atol, grad_atol):
    """T3：loss 与 dlogits 均与 F.cross_entropy 对齐；含 ignore_index 用例。"""
    logits = torch.randn(N, V, dtype=dtype, device="cuda")
    targets = torch.randint(0, V, (N,), device="cuda")

    # 每种 ignore_index（包括默认 -100）都实际覆盖约 20% 的忽略行。
    mask = torch.rand(N, device="cuda") < 0.2
    targets[mask] = ignore_index

    # --- loss 对比 ---
    # Triton 路径保留参数化 dtype；fp32 oracle 从同一份（可能已量化的）
    # 输入构造，避免 bf16 用例在调用被测实现之前被悄悄升级为 fp32。
    logits_tri = logits.clone().requires_grad_(True)
    logits_ref = logits.float().requires_grad_(True)

    loss_tri = fused_cross_entropy(logits_tri, targets, ignore_index=ignore_index)
    loss_ref = F.cross_entropy(logits_ref, targets, ignore_index=ignore_index)

    assert loss_tri.dtype == torch.float32, "fused CE loss 必须以 fp32 返回"
    torch.testing.assert_close(
        loss_tri, loss_ref, atol=loss_atol, rtol=0, msg="loss 不一致",
    )

    # --- dlogits 对比 ---
    loss_tri.backward()
    loss_ref.backward()

    assert logits_tri.grad is not None
    torch.testing.assert_close(
        logits_tri.grad.float(), logits_ref.grad, atol=grad_atol, rtol=0,
        msg="dlogits 不一致",
    )


def test_cross_entropy_all_ignored():
    """mean reduction 全忽略时：与 PyTorch 一样 loss=NaN、梯度全零。"""
    N, V = 8, 1024
    logits_tri = torch.randn(N, V, device="cuda", requires_grad=True)
    logits_ref = logits_tri.detach().clone().requires_grad_(True)
    targets = torch.full((N,), -100, device="cuda", dtype=torch.long)

    loss_tri = fused_cross_entropy(logits_tri, targets)
    loss_ref = F.cross_entropy(logits_ref, targets)
    assert torch.isnan(loss_tri) and torch.isnan(loss_ref)

    loss_tri.backward()
    loss_ref.backward()
    torch.testing.assert_close(logits_tri.grad, logits_ref.grad, atol=0, rtol=0)


# ─── T4: 显存占用 < torch 版的 60% ───────────────────────────────────────────

def test_memory_fused_ce():
    """T4：V=32k、N=4096 时 fused CE 增量峰值显存 < torch 版的 60%。"""
    N, V = 4096, 32768

    def measure_incremental_peak(fn):
        # 输入是两种实现共同的必要常驻量，先分配再记录基线；每轮结束显式
        # 释放整张 autograd 图，防止上一轮的 logits/grad 污染下一轮峰值。
        logits = torch.randn(N, V, device="cuda", dtype=torch.float32,
                             requires_grad=True)
        targets = torch.randint(0, V, (N,), device="cuda")
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()

        loss = fn(logits, targets)
        loss.backward()
        torch.cuda.synchronize()
        peak_delta = torch.cuda.max_memory_allocated() - baseline

        del loss, logits, targets
        gc.collect()
        torch.cuda.empty_cache()
        return peak_delta

    mem_torch = measure_incremental_peak(
        lambda logits, targets: F.cross_entropy(logits, targets),
    )
    mem_fused = measure_incremental_peak(
        lambda logits, targets: fused_cross_entropy(logits, targets),
    )

    ratio = mem_fused / mem_torch
    assert ratio < 0.60, (
        f"fused CE 增量峰值 = {mem_fused/1e6:.1f} MB，"
        f"torch = {mem_torch/1e6:.1f} MB，"
        f"比例 = {ratio:.2%}（需 < 60%）"
    )


# ─── T5: 数值稳定性（极大/极小 logits）────────────────────────────────────────

def test_numerical_stability():
    """T5：logits 含 ±1e4 极值时 loss 有限，且与 fp64 参考一致（rtol 1e-3）。"""
    N, V = 128, 1024
    # 生成含极值的 logits
    logits = torch.randn(N, V, dtype=torch.float32, device="cuda") * 100
    # 注入几个极端值
    logits[0, 0] = 1e4
    logits[1, 100] = -1e4
    targets = torch.randint(0, V, (N,), device="cuda")

    loss_tri = fused_cross_entropy(logits, targets)

    # 检查有限性
    assert torch.isfinite(loss_tri), f"loss 为 inf 或 nan：{loss_tri}"

    # 与 fp64 参考对比
    loss_ref64 = F.cross_entropy(logits.double(), targets.long())
    torch.testing.assert_close(
        loss_tri.double(), loss_ref64,
        rtol=1e-3, atol=0,
        msg="极值 logits 下 loss 与 fp64 参考偏差过大（online softmax 实现可能有误）",
    )
