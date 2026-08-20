"""Phase 8.2 验收：每个测试对应冻结 SPEC 的 T 编号。"""

import ast
import inspect
import math
import textwrap

import pytest
import torch

from minilp.fp8 import (
    AmaxHistory, compute_scale, dequantize_fp8, fp8_finfo, quantize_fp8,
)
from minilp.scaler import SimpleGradScaler
from minilp.train import fp8_linear_forward, master_weight_sgd_step


_FORMATS = {
    "e4m3": (torch.float8_e4m3fn, 3),
    "e5m2": (torch.float8_e5m2, 2),
}


@pytest.mark.parametrize("fmt", ["e4m3", "e5m2"])
def test_t1_fp8_finfo_matches_torch(fmt):
    """T1：学生闭式边界与 torch 元数据逐位对拍。"""
    dtype, _ = _FORMATS[fmt]
    maximum, smallest_normal = fp8_finfo(fmt)
    oracle = torch.finfo(dtype)
    assert maximum == oracle.max
    assert smallest_normal == oracle.smallest_normal
    source = textwrap.dedent(inspect.getsource(fp8_finfo))
    tree = ast.parse(source)
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "torch"
        and node.func.attr == "finfo"
        for node in ast.walk(tree)
    ), "fp8_finfo 必须从位布局推导，不能调用 torch.finfo"


@pytest.mark.parametrize("fmt", ["e4m3", "e5m2"])
def test_t2_saturating_roundtrip(fmt):
    """T2：范围内误差满足尾数界，范围外严格饱和且有限。"""
    dtype, mantissa_bits = _FORMATS[fmt]
    info = torch.finfo(dtype)
    generator = torch.Generator().manual_seed(820 + mantissa_bits)
    exponents = torch.empty(256).uniform_(
        math.log2(info.smallest_normal), math.log2(info.max), generator=generator,
    )
    magnitudes = torch.pow(2.0, exponents)
    signs = torch.where(torch.rand(256, generator=generator) < .5, -1.0, 1.0)
    values = magnitudes * signs
    restored = dequantize_fp8(quantize_fp8(values, fmt, 1.0), 1.0)
    relative = (restored - values).abs() / values.abs()
    assert relative.max().item() <= 2.0 ** (-mantissa_bits)

    extreme = torch.tensor([-1e30, 1e30])
    saturated = quantize_fp8(extreme, fmt, 1.0).float()
    assert torch.isfinite(saturated).all()
    assert torch.equal(saturated, torch.tensor([-info.max, info.max]))


@pytest.mark.parametrize("fmt", ["e4m3", "e5m2"])
def test_t3_compute_scale_power_of_two_range(fmt):
    """T3：独立 math oracle 验证二次幂 scale 与异常 amax。"""
    maximum = 448.0 if fmt == "e4m3" else 57344.0
    for amax in [0.03, 1.0, 17.5, 999.0, 80000.0]:
        actual = compute_scale(amax, fmt)
        expected = 2.0 ** math.floor(math.log2(maximum / amax))
        assert actual == expected
        assert maximum / 2 < amax * actual <= maximum
        margin_scale = compute_scale(amax, fmt, margin=3)
        assert margin_scale == actual / 8
        assert maximum / 16 < amax * margin_scale <= maximum / 8
    for invalid in [0.0, -1.0, math.inf, -math.inf, math.nan]:
        assert compute_scale(invalid, fmt) == 1.0
    for invalid_margin in [-1, True, 1.5, "1"]:
        with pytest.raises(ValueError, match="margin"):
            compute_scale(1.0, fmt, invalid_margin)


def test_t4_amax_history_window_and_empty():
    """T4：逐步手算窗口最大值，覆盖淘汰与空历史。"""
    history = AmaxHistory(window=3)
    assert history.scale("e4m3") == 1.0
    seen = []
    for value in [1.0, 8.0, 2.0, 0.5, 4.0, 32.0]:
        history.update(torch.tensor([-value, value / 2]))
        seen.append(value)
        window_max = max(seen[-3:])
        expected = 2.0 ** math.floor(math.log2(448.0 / window_max))
        assert history.scale("e4m3") == expected


def _scaled_value(scaler):
    return float(scaler.scale(torch.tensor(1.0)))


def test_t5_grad_scaler_state_machine():
    """T5：非有限跳步、回退、增长、精确 unscale 与重复调用。"""
    parameter = torch.nn.Parameter(torch.tensor([3.0]))
    optimizer = torch.optim.SGD([parameter], lr=0.25)
    scaler = SimpleGradScaler(init_scale=8.0, growth_interval=2)

    before = parameter.detach().clone()
    parameter.grad = torch.tensor([math.inf])
    scaler.unscale_(optimizer)
    with pytest.raises(RuntimeError):
        scaler.unscale_(optimizer)
    scaler.step(optimizer)
    assert torch.equal(parameter, before)
    scaler.update()
    assert _scaled_value(scaler) == 4.0

    for expected_scale in [4.0, 8.0]:
        parameter.grad = torch.tensor([3.0 * _scaled_value(scaler)])
        scaler.unscale_(optimizer)
        torch.testing.assert_close(parameter.grad, torch.tensor([3.0]), atol=1e-7, rtol=0)
        scaler.step(optimizer)
        scaler.update()
        assert _scaled_value(scaler) == expected_scale

    # found_inf 必须按 optimizer 隔离；一个 optimizer 跳步不能阻止另一个更新。
    bad = torch.nn.Parameter(torch.tensor([2.0]))
    good = torch.nn.Parameter(torch.tensor([2.0]))
    bad_optimizer = torch.optim.SGD([bad], lr=0.5)
    good_optimizer = torch.optim.SGD([good], lr=0.5)
    multi = SimpleGradScaler(init_scale=8.0, growth_interval=10)
    bad.grad = torch.tensor([math.inf])
    good.grad = torch.tensor([8.0])
    multi.unscale_(bad_optimizer)
    multi.unscale_(good_optimizer)
    multi.step(bad_optimizer)
    multi.step(good_optimizer)
    assert bad.item() == 2.0
    assert good.item() == 1.5
    with pytest.raises(RuntimeError):
        multi.step(good_optimizer)
    multi.update()
    assert _scaled_value(multi) == 4.0


def test_t6_master_weights_accumulate_sub_ulp_updates():
    """T6：fp16 直接更新停滞，而 fp32 master 累积微小更新。"""
    direct = torch.tensor([1.0], dtype=torch.float16)
    low_precision = direct.clone()
    master = low_precision.float().clone()
    gradient = torch.tensor([1e-4], dtype=torch.float16)
    lr = 0.1
    for _ in range(200):
        direct.add_(gradient, alpha=-lr)
        master_weight_sgd_step([low_precision], [master], [gradient], lr)
    assert direct.item() == 1.0
    assert master.item() < 0.999
    assert low_precision.item() < direct.item()


def _train_linear(use_fp8):
    generator = torch.Generator().manual_seed(827)
    x = torch.randn(96, 4, generator=generator)
    truth = torch.randn(4, 2, generator=generator)
    target = x @ truth
    weight = torch.zeros(4, 2, requires_grad=True)
    optimizer = torch.optim.SGD([weight], lr=0.12)
    x_history, w_history = AmaxHistory(8), AmaxHistory(8)
    losses = []
    # 五步处于两条曲线都明显下降、又尚未被 FP8 量化地板主导的区间；若把
    # fp32 训练到近零再做相对误差，分母会让“<30%”口径失去数值意义。
    for _ in range(5):
        optimizer.zero_grad(set_to_none=True)
        prediction = (
            fp8_linear_forward(x, weight, x_history, w_history)
            if use_fp8 else x @ weight
        )
        loss = torch.nn.functional.mse_loss(prediction, target)
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
        losses.append(float(loss))
    return losses


def test_t7_fp8_toy_converges_near_fp32():
    """T7：相同固定数据与优化器下，独立 fp32 路径作为基线。"""
    fp32_losses = _train_linear(False)
    fp8_losses = _train_linear(True)
    assert all(math.isfinite(value) for value in fp8_losses)
    assert fp32_losses[-1] < fp32_losses[0] * 0.6
    assert fp8_losses[-1] < fp8_losses[0] * 0.6
    assert abs(fp8_losses[-1] - fp32_losses[-1]) / fp32_losses[-1] < 0.30

    # 1.125 在 E4M3 中精确可表示，而 E5M2 会舍入到 1.0；这是独立格式契约门。
    probe = fp8_linear_forward(
        torch.tensor([[1.125]]), torch.tensor([[1.0]]), AmaxHistory(), AmaxHistory()
    )
    assert probe.item() == 1.125


def test_t8_delayed_scaling_handles_amax_spike():
    """T8：突增本步只饱和，下一步使用已记录的大 amax。"""
    x_history, w_history = AmaxHistory(4), AmaxHistory(4)
    fp8_linear_forward(torch.tensor([[1.0]]), torch.tensor([[1.0]]), x_history, w_history)
    old_scale = x_history.scale("e4m3")
    spike = torch.tensor([[1000.0]])
    current = fp8_linear_forward(spike, torch.tensor([[1.0]]), x_history, w_history)
    assert torch.isfinite(current).all()
    assert current.item() == pytest.approx(448.0 / old_scale)

    adapted_scale = x_history.scale("e4m3")
    expected = 2.0 ** math.floor(math.log2(448.0 / 1000.0))
    assert adapted_scale == expected
    following = fp8_linear_forward(spike, torch.tensor([[1.0]]), x_history, w_history)
    assert torch.isfinite(following).all()
    assert following.abs().item() > current.abs().item()
