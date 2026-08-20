# SPEC — Phase 3.1: DeepSeek-V4 方向机制（sparse attention / Hyper-Connections / Muon）

> 状态：FROZEN（接口已冻结）
> 模式：Foundation —— 三个机制的数学核心由学生手写；脚手架/toy 模型给定
> 算力：机制正确性全 CPU；三个真实消融（显存/吞吐/收敛对照）留待租卡（M 级），
>       本 phase 只交付机制实现 + 消融脚本
> 三个单元相互独立，可任选顺序

## 1. 问题

DeepSeek-V4 方向的三个代表性机制，各做一个最小正确实现：

- **U1 Block Top-K Sparse Attention**：每个 query 只 attend 到「按块池化打分
  选出的 top-k 个 KV 块 + 本地窗口 + attention sink（首块）」，NSA/MoBA 家族的
  共同骨架。
- **U2 Hyper-Connections（HC）**：把单条 residual stream 扩成 n 条，
  层间用可学习矩阵混合（Zhu et al. 2024, ByteDance）。核心可测性质：
  **特定初始化下 HC 精确退化为标准 Pre-Norm residual**。
- **U3 Muon**：对 2D 参数用 Newton–Schulz 迭代做矩阵符号函数
  msign(G) ≈ U Vᵀ 的正交化更新（Jordan et al. 2024），非 2D 参数回退 AdamW。

学完必须能回答（写进 POSTMORTEM）：
- sparse attention 省的是什么（FLOPs 还是 KV 显存还是都省）？decode 和 prefill
  阶段的收益结构有何不同？
- HC 为什么能在不加宽模型主干的情况下改善深层梯度传播？初始化如何保证
  「起点不劣于 residual」？
- msign 更新与 SGD/Adam 的几何区别是什么？为什么只对 2D 矩阵参数做？

## 2. 冻结接口（minidsv4/）

```python
# minidsv4/sparse_attn.py —— 学生实现
@dataclass
class SparseAttnConfig:
    block_size: int      # KV 分块大小
    top_k: int           # 每 query 选中的远程块数
    local_blocks: int    # 无条件保留的最近块数（含当前块）
    sink_blocks: int = 1 # 无条件保留的开头块数

def build_block_mask(scores_qk_block: torch.Tensor, cfg: SparseAttnConfig,
                     q_pos: torch.Tensor) -> torch.Tensor:
    """scores_qk_block: (B, H, T_q, N_blocks) 块级打分（已给定池化方式：
    K 在块内 mean-pool 后与 q 点积）。返回 bool (B, H, T_q, N_blocks)。
    因果约束：不许选中含未来位置的块（当前块内部由 token 级 causal mask 处理）。"""

def sparse_attention(q, k, v, cfg: SparseAttnConfig) -> torch.Tensor:
    """(B, H, T, D) 因果 sparse attention。被 mask 的块不参与 softmax。
    参考实现允许 dense 计算 + 块 mask（正确性优先，不要求真稀疏算力）。"""

def sparse_attn_flops(T: int, D: int, cfg: SparseAttnConfig) -> int:
    """闭式：decode 单 token 的 QK^T+AV FLOPs（按选中块数上界，2*选中token数*D*2）。"""

# minidsv4/hyper_conn.py —— 学生实现
class HyperConnection(nn.Module):
    def __init__(self, dim: int, expand_n: int, layer_id: int): ...
    """静态 HC。持有 n 条 stream 的混合参数：
    beta ∈ R^n（层输出写回各 stream 的深度权重），
    alpha ∈ R^{n×n}（stream 间宽度混合）+ 层输入读出权重 ∈ R^n。
    构造器默认即恒等初始化：必须精确等价于标准 residual（见 T4）。"""
    def width_mix(self, h: torch.Tensor) -> torch.Tensor: ...   # (B,T,n,C)->(B,T,n,C)
    def read(self, h: torch.Tensor) -> torch.Tensor: ...        # (B,T,n,C)->(B,T,C)
    def write(self, h: torch.Tensor, out: torch.Tensor) -> torch.Tensor: ...

def expand_stream(x: torch.Tensor, n: int) -> torch.Tensor:      # (B,T,C)->(B,T,n,C) 复制
def collapse_stream(h: torch.Tensor) -> torch.Tensor:            # 均值聚合回 (B,T,C)

# minidsv4/muon.py —— 学生实现
def newton_schulz_msign(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """五步 NS 迭代，系数冻结 (a,b,c)=(3.4445, -4.7750, 2.0315)：
    X ← a·X + b·(XXᵀ)X + c·(XXᵀ)²X，X₀ = G/‖G‖_F（bf16/fp32 均按 fp32 算）。
    行>列时先转置算完再转回。"""

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True): ...
    """仅接受 2D 参数；步骤：momentum buffer → (nesterov) → msign → 
    按 sqrt(max(1, rows/cols)) 缩放 → p ← p − lr·update。"""

# minidsv4/toy_model.py —— 给定脚手架：可插拔 residual/HC 的 N 层 MLP toy 模型
# scripts/ablate_dsv4.py —— 给定：三个消融的租卡脚本（--dry-run 本地可跑）
```

## 3. 验收标准（tests/test_dsv4.py，CPU）

| 编号 | 通过条件 |
| --- | --- |
| T1 | **sparse≡dense 退化**：top_k + local + sink 覆盖全部块时，sparse_attention 与 dense causal SDPA 数值一致（atol 1e-5）|
| T2 | 因果性：build_block_mask 永不选中含未来 token 的块（随机 cfg × 20 组枚举断言）；输出对未来 v **和未来 k** 的扰动零敏感（数值验证。未来 k 的路径更隐蔽：当前块 pooled 分数含未来 K，参与 top-k 竞争会把过去块挤出选择集——当前块必须排除出 gating 并无条件保留）|
| T3 | 选块正确性：手工构造块打分，断言恰好选中 top-k + local + sink 的并集；flops 闭式与 mask 实际选中 token 数对拍 |
| T4 | **HC 恒等初始化**：identity_init 下，n∈{2,4} 的 HC toy 模型与标准 residual toy 模型 forward 输出逐位一致（同权重注入）|
| T5 | HC 梯度流：随机初始化、深度 16 的 toy 模型，HC 各层梯度范数的 max/min 比值不超过 residual 版的比值（deep supervision 的最低要求，固定 seed）|
| T6 | msign 近正交性（修订 2026-08：Muon 的五步 NS 系数为速度而调，不收敛到精确极分解，原 0.05/0.1 门槛物理不可达——实测正交误差 ~0.36、UVᵀ 相对误差 ~0.21）：随机高斯 G(64×32) 多 seed 下 (a) NS 输出奇异值全部 ∈ [0.6, 1.2]（实测 [0.682, 1.134]）；(b) 与 SVD 精确 UVᵀ 的 Frobenius 相对误差 < 0.3（实测 ≤ 0.232）；(c) 正交输入近不动点 max\|NS(Q)−Q\| < 0.05（实测 0.0085）。三条合起来排除清零/单纯归一化/方向错误的实现 |
| T7 | Muon 语义：momentum buffer 更新可手算对拍（2 步）；非 2D 参数传入必须 raise；rows/cols 缩放因子对拍 |
| T8 | Muon toy 收敛：固定 seed 的小回归任务，Muon 在等步数内 loss 低于同 lr 网格下最优 SGD-momentum（弱断言，只防实现完全错误）|

## 4. 产物

- `minidsv4/*.py` 全绿
- `scripts/ablate_dsv4.py --dry-run` 本地可跑通（真实消融租卡执行）
- `docs/3.1-dsv4/POSTMORTEM.md`
