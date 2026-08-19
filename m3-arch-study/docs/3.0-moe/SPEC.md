# SPEC — Phase 3.0: MoE / DeepSeek-V3 机制研究

> 状态：FROZEN（接口已冻结）
> 模式：Copilot —— 四个机制的核心（router、bias 均衡、MLA、MTP）学生手写；
>       dispatch 循环、模型组装、实验脚本脚手架提示给足
> 算力：correctness 全部本地 CPU；消融实验 M 级（W8–9 租卡）
> 命名纪律：这是 mechanism study，不宣称「复现 DeepSeek-V3」
> 依赖：独立模块，不 import M1 代码；PyTorch 实现

## 1. 问题

在 10–30M 规模上把 DeepSeek-V3 的四个核心机制各实现一遍，
correctness 由 CPU 测试定义，然后带着可信的实现上卡回答消融问题：

- 等 activated FLOPs 下 MoE 是否优于 Dense？
- bias 更新速度（gamma）如何影响 expert 均衡与 collapse？
- MLA 省多少 KV cache、损失多少质量？
- MTP 改善的是 next-token 还是更远的预测？

学完必须能回答（写进 POSTMORTEM）：
- aux-loss-free balancing 里 bias 为什么只加在「选择」上、
  不进 gating 权重？如果进了会怎样？
- 细粒度切分 + shared expert 各自解决什么问题？
- MLA 的低秩压缩为什么配合 RoPE 时需要 decoupled 的 rope 分支？
- MTP 的梯度流回主干的路径是什么？

## 2. 范围与非目标

范围：MoE 层（fine-grained + shared，dropless）、aux-loss-free 负载均衡、
MLA（含 decoupled RoPE 与压缩 KV cache）、MTP（深度 1，V3 风格顺序模块）、
等算力 config 生成器、消融实验脚手架。
非目标：不做 expert parallelism / 容量丢弃 / 序列级辅助 loss、
不做 MLA 的矩阵吸收推理优化（记入 backlog）、不做 >100M 训练。

## 3. 冻结接口（minimoe/）

```python
# minimoe/config.py
@dataclass
class MoEConfig:
    hidden_size: int = 256
    # dense FFN（对照组）
    intermediate_size: int = 1024
    # MoE
    n_routed_experts: int = 16
    n_shared_experts: int = 1
    top_k: int = 2
    moe_intermediate_size: int = 128     # 细粒度：单个 expert 很小
    bias_update_speed: float = 0.001     # gamma
    # MLA
    q_lora_rank: int = 96
    kv_lora_rank: int = 64
    qk_nope_head_dim: int = 32
    qk_rope_head_dim: int = 16
    v_head_dim: int = 32
    num_heads: int = 8
    # 通用
    vocab_size: int = 8192
    num_layers: int = 4
    max_seq_len: int = 512
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0

# minimoe/moe.py
class Router(nn.Module):
    def __init__(self, cfg: MoEConfig): ...
    def forward(self, x: Tensor  # (N, H)
                ) -> tuple[Tensor, Tensor]:
        """返回 (topk_idx (N, top_k) long, topk_weight (N, top_k) float)。
        选择：score = sigmoid(logits)，选 top-k 用 score + bias（bias 不参与梯度）；
        gating 权重：用**不加 bias** 的 score 归一化（和为 1）。"""
    @torch.no_grad()
    def update_bias(self, topk_idx: Tensor) -> None:
        """aux-loss-free 均衡：统计本 batch 各 expert 负载，
        高于均值的 bias -= gamma，低于均值的 bias += gamma。"""

class MoELayer(nn.Module):
    """fine-grained routed experts + always-on shared experts，dropless。"""
    def __init__(self, cfg: MoEConfig): ...
    def forward(self, x: Tensor) -> Tensor:   # (B, T, H) -> (B, T, H)
    # 属性：self.router；self.experts: ModuleList[SwiGLU]；self.shared: SwiGLU

# minimoe/mla.py
class MLA(nn.Module):
    """Multi-head Latent Attention（V2/V3 风格，训练形态，不做矩阵吸收）。
    q：低秩压缩 (H→q_lora_rank→heads*(nope+rope))；
    kv：压缩到 kv_lora_rank 的 latent + 独立的 shared rope key（每 token 一份）；
    cache 只存 (latent, rope_key)。"""
    def __init__(self, cfg: MoEConfig): ...
    def forward(self, x, cos, sin, positions,
                kv_cache: MLACache | None = None) -> Tensor: ...

class MLACache:
    def __init__(self, cfg, batch_size, max_seq_len, device, dtype): ...
    def update(self, latent: Tensor, k_rope: Tensor) -> tuple[Tensor, Tensor]: ...
    @property
    def seq_len(self) -> int: ...
    def memory_bytes(self) -> int:
        """当前缓存实际占用字节数（用于与 MHA cache 对比）。"""

# minimoe/mtp.py
class MTPHead(nn.Module):
    """深度 1 的 multi-token prediction（V3 风格顺序模块）：
    h'_t = TransformerBlock( RMSNorm(h_t) ⊕ RMSNorm(emb(t+1 token)) 经线性合并 )，
    用共享的 embedding/输出头预测 t+2。"""
    def __init__(self, cfg: MoEConfig, embed: nn.Embedding, head: nn.Linear): ...
    def forward(self, h: Tensor, input_ids: Tensor) -> Tensor:
        """h: 主干最后隐层 (B, T, H)；返回对位置 t+2 的 logits (B, T-1, V)。"""

# minimoe/model.py
class MiniMoELM(nn.Module):
    """组装：embedding → [MLA + MoELayer] × L → norm → head（tied）。
    __init__(cfg, use_moe: bool = True, use_mtp: bool = False)
    forward(input_ids, kv_caches=None) -> logits；use_mtp 时额外返回 mtp_logits。"""

# minimoe/parity.py
def dense_config_matching_flops(cfg: MoEConfig) -> MoEConfig:
    """给定 MoE config，返回 activated FLOPs 相等的 Dense config
    （调 intermediate_size），误差 < 2%。"""
def activated_ffn_params(cfg: MoEConfig, moe: bool) -> int:
    """每 token 实际参与计算的 FFN 参数量（闭式手算）。"""
```

## 4. 验收标准（tests/test_moe.py，全部 CPU）

| 编号 | 单元 | 通过条件 |
| --- | --- | --- |
| T1 | Router | top-k 选择与朴素 for 循环参考一致；gating 权重和为 1 且**不受 bias 影响**（改 bias 后权重数值不变，只有选择变）|
| T2 | MoELayer | dropless 等价性：层输出 == 逐 token 朴素计算 Σ w_i·expert_i(x) + shared(x)（atol 1e-6）；空 expert（本 batch 无 token 命中）不产生 NaN |
| T3 | 均衡动力学 | 构造偏斜输入训练若干步：不开 bias 更新时负载不均（max/min > 4）；开 bias 更新后负载比值显著收敛（max/min < 2）；bias 无梯度（.grad is None）|
| T4 | MLA 对齐 | cached vs non-cached 逐位置输出 allclose（atol 1e-5）；greedy 生成序列一致 |
| T5 | MLA 显存 | memory_bytes 与手算闭式一致，且 < 等 heads 标准 MHA cache 的 40%（本 config 下）|
| T6 | MTP | 输出 shape (B, T-1, V)；toy 数据上 main+MTP 联合 loss 300 步内下降 >50%；MTP 梯度确实流回主干（主干某参数 .grad 非零）|
| T7 | 等算力 | dense_config_matching_flops 的 activated FLOPs 误差 < 2%（用 activated_ffn_params 闭式互验）|

## 5. 消融实验（不进测试，W8–9 租卡跑，脚手架先备好）

`scripts/ablate.py --exp {moe_vs_dense, gamma_sweep, mla_vs_mha, mtp}`：
统一 TinyStories tokenized 数据、等 token 预算、固定 seed、
结果写 `results/*.json`。四组实验各 M 级以内，公平性要求：
等 activated FLOPs 或等 wall-clock 二选一并在报告里声明。

## 6. 产物

- `minimoe/*.py`（学生实现）+ 全绿测试
- 消融报告 `docs/3.0-moe/ABLATION.md`（四个问题各一节，图/表 + 结论）
- `docs/3.0-moe/POSTMORTEM.md`（含第 1 节四问）
