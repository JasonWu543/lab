# SPEC — Phase 1.2: Qwen-like Transformer from scratch

> 状态：FROZEN（接口已冻结）
> 模式：Foundation（RoPE / GQA attention / KV cache 手搓）
>       + Copilot（config、生成采样循环、权重转换脚手架给较密提示）
> 算力：本地为主（correctness 全部 CPU/MPS）；U2.6 冒烟 S 级
> 工期：约 1.5 周（W2–W3）
> 从本 phase 起切换到 PyTorch；Phase 1.1 的 Tensor 系统封存不再 import。

## 1. 问题

用 PyTorch 从零实现一个 Qwen2.5 风格的 decoder-only Transformer：
RMSNorm、RoPE、GQA、SwiGLU、KV cache、weight tying、采样生成。
两个硬门槛证明你真的写对了：

- **U2.4**：带 KV cache 与不带 cache 的解码，逐位置 logits allclose；
- **U2.5**：把官方 Qwen2.5-0.5B 权重加载进你的实现，
  与 transformers 官方实现在同一 prompt 下 logits 对齐。

学完必须能回答（写进 POSTMORTEM）：
- RoPE 为什么等价于在复平面上旋转？为什么它编码的是相对位置？
- GQA 省的是什么内存？推理时省在哪一步？
- KV cache 解码时，第 t 步的 query 只有 1 个 token，
  causal mask 和 position 各要怎么处理？
- weight tying 为什么可行？对 embedding 学习有什么影响？

## 2. 范围与非目标

范围：单卡、eager 模式、训练+推理都支持的模型代码；温度/top-p 采样。
非目标：不做 FlashAttention/compile 优化（M2 的事）、不做 batch 化
serving（M4 的事）、不做 MoE（M3 的事）、不做 dropout（小模型预训练不用）。

## 3. 冻结接口（minilm/model/）

```python
# minilm/model/config.py
@dataclass
class ModelConfig:
    vocab_size: int = 8192
    hidden_size: int = 256
    intermediate_size: int = 1024
    num_layers: int = 4
    num_heads: int = 8
    num_kv_heads: int = 4          # GQA；== num_heads 时退化为 MHA
    head_dim: int | None = None    # None → hidden_size // num_heads
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    attention_bias: bool = True    # Qwen2.5 风格：QKV 有 bias，o_proj 无
    tie_word_embeddings: bool = True

# minilm/model/model.py
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6): ...
    def forward(self, x): ...      # 计算用 float32，返回原 dtype

def precompute_rope(head_dim: int, max_seq_len: int, theta: float,
                    device=None) -> tuple[Tensor, Tensor]:
    """返回 (cos, sin)，shape 均为 (max_seq_len, head_dim)。"""

def apply_rope(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor,
               positions: Tensor) -> tuple[Tensor, Tensor]:
    """q: (B, n_heads, T, head_dim), k: (B, n_kv_heads, T, head_dim),
    positions: (T,) 或 (B, T)。旋转采用「前后对半」布局
    （rotate_half，与 HF Qwen2 一致），不是奇偶交错布局。"""

class KVCache:
    """预分配的逐层 cache。"""
    def __init__(self, cfg: ModelConfig, batch_size: int,
                 max_seq_len: int, device, dtype): ...
    def update(self, layer_idx: int, k: Tensor, v: Tensor
               ) -> tuple[Tensor, Tensor]:
        """写入新 k/v 并返回该层截至当前的全部 k/v。"""
    @property
    def seq_len(self) -> int: ...  # 已缓存的 token 数

class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig): ...
    def forward(self, x, cos, sin, positions,
                kv_cache: KVCache | None = None,
                layer_idx: int = 0): ...

class MLP(nn.Module):              # SwiGLU: down(silu(gate(x)) * up(x))
    def __init__(self, cfg: ModelConfig): ...

class DecoderLayer(nn.Module):     # pre-norm ×2 + 残差 ×2
    def __init__(self, cfg: ModelConfig, layer_idx: int): ...

class MiniLM(nn.Module):
    def __init__(self, cfg: ModelConfig): ...
    def forward(self, input_ids: Tensor,          # (B, T)
                kv_cache: KVCache | None = None,
                positions: Tensor | None = None   # None → 自动推断
                ) -> Tensor:                       # logits (B, T, vocab)

# minilm/model/generate.py
@torch.no_grad()
def sample_next(logits: Tensor, temperature: float = 1.0,
                top_p: float = 1.0,
                generator: torch.Generator | None = None) -> Tensor:
    """logits: (B, vocab) → next ids (B,)。temperature=0 视为 greedy。"""

@torch.no_grad()
def generate(model: MiniLM, input_ids: Tensor, max_new_tokens: int,
             temperature: float = 1.0, top_p: float = 1.0,
             use_cache: bool = True, eos_token_id: int | None = None,
             generator: torch.Generator | None = None) -> Tensor:
    """返回 (B, T_in + T_new)。use_cache=False 时每步全量前向（慢但作对照）。"""

# minilm/model/counting.py
def count_params(cfg: ModelConfig) -> int:
    """闭式手算参数量（不实例化模型），必须 == sum(p.numel())。"""
def estimate_flops_per_token(cfg: ModelConfig, seq_len: int) -> int:
    """前向 FLOPs/token 估算（矩阵乘 2mnk 记法，含 attention 两次 matmul）。"""

# minilm/model/convert_qwen.py
def convert_qwen_config(hf_config: dict) -> ModelConfig: ...
def load_qwen(model_dir: str | Path, device="cpu",
              dtype=torch.float32) -> MiniLM:
    """读官方 Qwen2.5 safetensors，重命名映射进 MiniLM（config 在 model.cfg）。"""
```

约定：

- RoPE 用 rotate_half（前后对半）布局，cos/sin 以 float32 预计算；
- attention 用显式 `q @ k^T / sqrt(head_dim)` + causal mask 实现
  （不许调 `F.scaled_dot_product_attention`——它是测试里的对照物）；
- GQA 在计算时把 k/v 沿 head 维 repeat 到 num_heads（repeat_kv）；
- 带 cache 解码时 mask 的规则：新 token 能看见 cache 里全部位置 +
  自身及之前的新位置；
- 初始化：Linear 权重 `normal_(0, 0.02)`，embedding 同；o_proj/down_proj
  按 `0.02/sqrt(2*num_layers)` 缩放（GPT-2 残差流惯例）。

## 4. 数据与外部依赖

- U2.5 需要官方权重：`Qwen/Qwen2.5-0.5B`（~1GB，safetensors）。
  下载脚本 `scripts/download_qwen.py` → `data/qwen2.5-0.5b/`。
  权重不存在时该组测试自动 skip（其余测试不受影响）。
- U2.6 冒烟不依赖 Phase 1.0 的 tokenizer（用随机 token 序列即可）。

## 5. 验收标准（tests/test_transformer.py）

| 编号 | 单元 | 通过条件 |
| --- | --- | --- |
| T1 | U2.1 | RMSNorm/MLP/Embedding 与 PyTorch 官方等价实现数值对齐（fp32, rtol 1e-5）|
| T2 | U2.2 | attention 输出与 `F.scaled_dot_product_attention`（repeat_kv 后）对齐；`num_kv_heads=num_heads` 时与 MHA 等价 |
| T3 | U2.2 | RoPE：与 HF rotate_half 参考数值一致；attention 分数只依赖相对位置（整体平移 positions 分数不变）|
| T4 | U2.3 | `count_params(cfg)` == `sum(p.numel())`（多组 config，含 tie/untie、GQA/MHA）；weight tying 后 lm_head 与 embedding 是同一块存储 |
| T5 | U2.4 | **硬门槛**：cached vs non-cached 逐位置 logits allclose（atol 1e-5）；greedy 下输出序列完全一致；prefill+逐 token 与一次性前向一致 |
| T6 | U2.4 | sampler：top-p 截断正确（给定 logits 只可能采到核内 token）；temperature=0 == argmax；固定 generator 可复现 |
| T7 | U2.5 | **硬门槛**：官方 Qwen2.5-0.5B 权重加载后，与 transformers 在同一 prompt 的 logits 对齐（fp32, atol 5e-4）；无权重时 skip |
| T8 | U2.6 | 10M 级小模型在 32 条随机序列上 ≤300 步过拟合到 loss < 0.2，无 NaN |

FLOPs 估算 vs profiler 实测（差距 <2x 并解释）放 `benchmarks/bench_model.py`，
不进测试（profiler 数值因机器而异）。

## 6. 产物

- `minilm/model/*.py`（学生实现）
- `docs/1.2-transformer/POSTMORTEM.md`（含第 1 节四问 + FLOPs 对比表）
- 通关后：用 Phase 1.0 的 tokenizer 走一次端到端 encode → generate → decode
