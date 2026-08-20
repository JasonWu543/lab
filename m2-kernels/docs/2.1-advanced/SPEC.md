# SPEC — Phase 2.1: 进阶 kernel（Fused RoPE / Blockwise Attention）

> 状态：FROZEN（接口已冻结）
> 模式：同 2.0 —— kernel 本体学生手写；wrapper / autograd.Function 骨架给定
> ⚠️ 本地无 CUDA：与 2.0 相同约束——测试全部 CUDA-skip，参考答案只能静态审，
>    首次租卡先跑 `scripts/validate_reference.sh`（本 phase 必须把新 kernel 加进去）

## 1. 问题

- **RoPE**：训练中每层每步都要旋转 Q/K，eager 实现要 4 次读写 + 中间 tensor；
  fused kernel 一次 load/store 完成，且 backward 有优雅闭式（旋转 −θ）。
- **Blockwise attention**：FlashAttention 的核心思想——O(T²) 的 score 矩阵
  从不落显存，K/V 分块流过 SRAM，用 **online softmax** 维护 running max 与
  分母。本 phase 实现简化版 forward（backward 用重算 + PyTorch 兜底，
  真 bwd kernel 列为 bonus）。

学完必须能回答（写进 POSTMORTEM）：
- RoPE bwd 为什么恰好是「旋转 −θ」？从旋转矩阵正交性推一遍。
- online softmax 每步 rescale 修正的是什么？数值上为什么必须减 running max？
- blockwise 省的是显存带宽还是容量？什么 (T, D) 下相对 SDPA 反而更慢？

## 2. 冻结接口（kernels/）

```python
# kernels/rope.py
def rope_fwd(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor,
             sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """q,k: (B, H, T, D)，cos/sin: (T, D//2)。rotate_half 约定与 m1 的
    Qwen 实现完全一致（前半/后半配对，非相邻交错）。fused 单 kernel，
    fp32 中间计算，输出 dtype 同输入。"""

class RoPEFunction(torch.autograd.Function):
    """骨架给定，设计已定：bwd 通过 sign 参数翻转复用同一个 kernel
    （kernel 带 sign 形参，fwd 传 +1、bwd 传 −1）。学生任务是写出这个
    带 sign 的 kernel 本体，并在 POSTMORTEM 推导为什么 bwd 恰好是旋转 −θ。"""

def apply_rope(q, k, cos, sin): ...   # 用户入口，走 autograd.Function

# kernels/block_attn.py
def block_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                    causal: bool = True, sm_scale: float | None = None,
                    BLOCK_M: int = 64, BLOCK_N: int = 64) -> torch.Tensor:
    """(B, H, T, D)，D ∈ {32, 64, 128}。forward-only Triton kernel：
    program 处理一个 (batch·head, M 块)；内层循环 N 块流式做 online softmax；
    score 矩阵不得整体物化（T5 用显存断言验证）。
    sm_scale 默认 1/sqrt(D)。返回 fp32 累加后 cast 回输入 dtype。"""

class BlockAttentionFunction(torch.autograd.Function):
    """骨架给定：fwd 调 kernel 并保存 (q,k,v)；bwd 用 PyTorch 重算标准
    attention 求梯度（正确性兜底）。真 bwd kernel 为 bonus，不进验收。"""
```

## 3. 验收标准（tests/test_kernels_21.py，无 CUDA 全 skip）

| 编号 | 通过条件 |
| --- | --- |
| T1 | rope_fwd 对拍 m1 的 `apply_rope` 参考语义（多 shape、fp32/bf16，bf16 atol 按 2.0 的 F-07 口径从量化噪声推导并写明依据）|
| T2 | RoPE bwd：`gradcheck` 级对拍 eager 实现的 autograd 梯度（fp64 不可用则 fp32 + 放宽到推导依据内）；旋转正交性→‖grad_out‖=‖grad_in‖ 逐行验证 |
| T3 | block_attention ≡ `F.scaled_dot_product_attention`（causal/非 causal、T 非 BLOCK 整数倍、D∈{32,64,128}、fp32/bf16）|
| T4 | 数值稳定：logits 加 +80 偏移（softmax 溢出区），fused 输出仍与 fp32 SDPA 一致——验证 running-max 有效 |
| T5 | **不物化 score**：T=4096 下增量峰值显存（2.0 F-06 的口径）< 物化 (T×T) fp32 score 矩阵的一半 |
| T6 | benchmark 脚手架：warmup 后 vs SDPA / eager RoPE 的加速表打印（不做硬性加速断言，真卡填 BASELINE）|

## 4. 产物

- `kernels/{rope,block_attn}.py` 静态自查通过 + 真卡全绿
- `scripts/validate_reference.sh` 扩展含 2.1
- `docs/2.1-advanced/POSTMORTEM.md`（含 online softmax 推导）
