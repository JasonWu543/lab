# SPEC — Phase 3.2: 长上下文机制（Delta Rule / 线性注意力状态压缩）

> 状态：FROZEN（接口已冻结）
> 模式：Foundation —— 递推与 chunk 并行化由学生手写（chunk 化是本 phase 的核心难点）
> 算力：正确性全 CPU；长序列显存/召回消融留待租卡
> 参考文献：DeltaNet (Schlag et al. 2021)、Gated DeltaNet (Yang et al. 2024)、
>           Kimi 线性注意力方向的公开资料

## 1. 问题

标准 attention 的 KV cache 随序列长度线性增长；线性注意力把上下文压进
**固定大小的矩阵状态 S ∈ R^{d_v×d_k}**。朴素外积累加（Linear Attention）
只会往 S 里加东西，检索会互相污染；**delta rule** 在写入前先「擦除旧关联」：

```
S_t = S_{t-1} (I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ        # DeltaNet
S_t = α_t · S_{t-1} (I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ   # Gated：标量遗忘门 α_t
o_t = S_t q_t
```

其中 k_t 已 L2 归一化，β_t ∈ (0,1) 为写入强度，α_t ∈ (0,1] 为衰减门。
逐 token 递推 O(T) 串行不可训练加速，**chunkwise 并行**（块内并行、块间递推）
是让它实用的关键，也是学生的核心任务。

学完必须能回答（写进 POSTMORTEM）：
- delta rule 相比朴素线性注意力多做了什么？为什么它能精确「覆写」一个 key 的关联？
- chunk 化的数学：块间只需要传什么？为什么复杂度从 O(T·d²) 串行变成
  O(T/C · C²·d) 并行友好？
- 固定大小状态的信息论上限在哪里？什么任务长度下必然开始丢东西？

## 2. 冻结接口（minikda/）

```python
# minikda/delta.py —— 学生实现
def delta_rule_recurrent(q, k, v, beta, alpha=None) -> torch.Tensor:
    """逐 token 递推参考实现（正确性 oracle 的地位，但它本身也是学生任务）。
    q,k,v: (B, H, T, D)，k 需内部 L2 归一化；beta: (B, H, T)；
    alpha: (B, H, T) 或 None（None = DeltaNet 无衰减）。
    返回 o: (B, H, T, D)。fp32 累加。"""

def delta_rule_chunked(q, k, v, beta, alpha=None, chunk_size: int = 16)
        -> torch.Tensor:
    """chunkwise 并行版：块内用矩阵运算一次算完，块间递推状态。
    数值上必须与 recurrent 版一致（T1 的 atol 门槛）。"""

# minikda/layer.py —— 学生实现
class DeltaAttention(nn.Module):
    """完整层：x -> q,k,v,beta,alpha 投影（beta,alpha 过 sigmoid）->
    delta_rule_chunked -> 输出投影。支持 incremental decode：
    step(x_t, state) -> (o_t, state)，state 即 (B,H,D,D) 矩阵 + 无其他缓存。"""
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
    def step(self, x_t: torch.Tensor, state: torch.Tensor | None)
            -> tuple[torch.Tensor, torch.Tensor]: ...

def state_bytes(cfg) -> int:
    """闭式：单请求 decode 状态字节数（与 T 无关！）——对照 m3 3.0 的
    mha_cache_bytes 同口径，供 4.0 风格的容量讨论。
    口径：状态随递推以 fp32 持有（与输入 dtype 无关），按 4 字节/元素计。"""

# minikda/tasks.py —— 给定脚手架：copy / associative-recall 合成任务生成器
# scripts/ablate_longctx.py —— 给定：长序列显存曲线 + 召回率消融（--dry-run 可本地）
```

## 3. 验收标准（tests/test_longctx.py，CPU）

| 编号 | 通过条件 |
| --- | --- |
| T1 | **chunked ≡ recurrent**：随机输入，chunk_size ∈ {1, 4, 16, T}、T 非 chunk 整数倍、含/不含 alpha，两实现 atol 1e-4 一致（fp32）|
| T2 | 精确覆写语义：构造 k 相同、v 不同的两次写入（β=1, α=1），第二次写入后用该 k 检索必须得到第二个 v（atol 1e-5）——delta rule 区别于朴素累加的判据 |
| T3 | 退化对拍：β 恒 1、k 集合两两正交、无 α 时，输出等于「正交槽位查表」手算结果；α 恒 1 时 Gated 退化为 DeltaNet（数值一致）|
| T4 | 衰减门方向：α 恒 0.5 时，很早写入的关联的检索范数严格小于刚写入的（构造性断言）|
| T5 | **forward ≡ step 流式一致**：DeltaAttention 整段 forward 与逐 token step 的输出逐位置 atol 1e-4 一致；state shape 恒为 (B,H,D,D) 与 T 无关 |
| T6 | state_bytes 闭式对拍 + 与实际 state tensor `element_size()*nelement()` 相等 |
| T7 | 因果性：扰动未来 token，前缀输出零变化（数值验证）|
| T8 | toy 关联召回：给定脚手架任务上训练 200 步（固定 seed，双层小模型），召回准确率 > 0.9（弱门槛，防机制性写错；CPU < 60s）|

## 4. 产物

- `minikda/*.py` 全绿
- `scripts/ablate_longctx.py --dry-run` 可跑通
- `docs/3.2-longctx/POSTMORTEM.md`（含 chunk 化推导）
