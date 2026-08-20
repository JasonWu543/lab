# SPEC — Phase 8.2: 低精度训练（混合精度全家桶 / FP8 数值机制）

> 状态：FROZEN
> 模式：Foundation —— scaling 机制与数值分析手写；toy 模型脚手架给定
> 算力：全 CPU。FP8 用 torch 的 float8_e4m3fn / float8_e5m2 **存储** dtype 做
>       quantize-dequantize 模拟（计算仍在 fp32）；真 FP8 tensor core（H100 级）
>       的端到端加速与收敛 trick 记入 backlog，本 phase 只交付机制正确性
> 前置事实（冻结口径）：E4M3 max=448、E5M2 max=57344；fp16 max=65504

## 1. 问题

低精度训练 = 三层机制叠加：
1. **fp16/bf16 混合精度**：master weights fp32 + loss scaling + inf/nan 跳步
   （GradScaler 协议——你在 7.2b 刚 review 过写错的版本，现在自己从零写对）；
2. **FP8 格式取舍**：E4M3（精度多）放 forward 激活/权重、E5M2（范围大）放
   梯度——为什么；
3. **per-tensor scaling + amax 历史（delayed scaling）**：Transformer Engine
   的核心机制，scale 用过去 window 的 amax 推算，而不是当前值。

学完必须能回答（写进 POSTMORTEM）：
- loss scaling 解决的是什么（哪个 tensor 在哪一步下溢）？为什么 bf16 通常不需要？
- E4M3/E5M2 的 (exponent, mantissa) 分配如何决定「该放哪类 tensor」？
- delayed scaling 为什么可行（amax 的时间相关性）？溢出时协议怎么兜底？

## 2. 冻结接口（minilp/）

```python
# minilp/fp8.py —— 学生实现
def fp8_finfo(fmt: str) -> tuple[float, float]:
    """fmt ∈ {"e4m3","e5m2"}。返回 (max_representable, smallest_normal)，
    必须**手推闭式**计算（禁止调 torch.finfo——测试会拿 torch.finfo 对拍你）。"""

def quantize_fp8(t: torch.Tensor, fmt: str, scale: float) -> torch.Tensor:
    """(t * scale) 先 clamp 到 ±max 再 cast 到对应 float8 dtype 返回
    （存储模拟；saturate 语义）。"""

def dequantize_fp8(q: torch.Tensor, scale: float) -> torch.Tensor:
    """q.float() / scale。"""

def compute_scale(amax: float, fmt: str, margin: int = 0) -> float:
    """scale = 2^floor(log2(max_representable / amax)) / 2^margin；
    amax<=0 或非有限时返回 1.0。margin 必须是非负整数（bool 不算整数）。"""

class AmaxHistory:
    def __init__(self, window: int = 16): ...
    def update(self, t: torch.Tensor) -> None: ...   # 记录当前 amax
    def scale(self, fmt: str, margin: int = 0) -> float:
        """用窗口内 amax 最大值计算 scale（delayed scaling）。空历史返回 1.0。"""

# minilp/scaler.py —— 学生实现
class SimpleGradScaler:
    def __init__(self, init_scale=2.**16, growth_factor=2.0,
                 backoff_factor=0.5, growth_interval=2000): ...
    def scale(self, loss): ...            # loss * scale
    def unscale_(self, optimizer): ...    # 各参数 grad /= scale，记录 inf/nan
    def step(self, optimizer): ...        # 有 inf/nan 则跳过 step
    def update(self): ...                 # 跳步→scale*=backoff；连续 growth_interval
                                          # 个成功步→scale*=growth
    # 协议顺序与 torch.cuda.amp 相同：scale→backward→unscale_→clip→step→update

# minilp/train.py —— 学生实现
def master_weight_sgd_step(params_lp, master_fp32, grads_lp, lr) -> None:
    """低精度参数 + fp32 master 副本：master -= lr*grad(fp32)；
    params_lp ← master cast 回低精度。"""

def fp8_linear_forward(x, w, x_hist: AmaxHistory, w_hist: AmaxHistory):
    """模拟 FP8 matmul：x/w 各自 delayed scaling → quantize(e4m3) →
    dequantize → fp32 matmul。返回输出。"""
```

## 3. 验收标准（tests/test_lowprec.py，CPU）

| 编号 | 通过条件 |
| --- | --- |
| T1 | fp8_finfo 闭式 vs `torch.finfo(torch.float8_e4m3fn/e5m2)` 对拍（max 与 smallest_normal）|
| T2 | quantize/dequantize roundtrip：scale=1 时在表示范围内的相对误差 ≤ 2^-3（e4m3）/ 2^-2（e5m2）；超范围值饱和到 ±max 不产生 inf |
| T3 | compute_scale：margin=0 时断言 `amax*scale ∈ (max/2, max]`；一般 margin=m 时断言 `amax*scale ∈ (max/2^(m+1), max/2^m]`；amax=0/inf/nan → 1.0；负数/bool/非整数 margin 必须 raise |
| T4 | AmaxHistory：窗口滚动淘汰、取窗口最大、空历史 1.0——构造序列逐步对拍 |
| T5 | **SimpleGradScaler 协议**：注入 inf 梯度 → step 被跳过（参数逐位不变）且 update 后 scale 减半；连续 growth_interval 个干净步 → scale 翻倍；unscale_ 后梯度等于真实梯度（1e-7）；对同一 optimizer 重复 unscale_ 必须 raise |
| T6 | master weights：fp16 参数直接 SGD 200 步 vs master-fp32 版，构造小梯度（更新量 < fp16 ulp）场景，前者停滞、后者持续下降——方向性断言 |
| T7 | **端到端 toy 收敛**：固定 seed 小回归任务，fp32 基线 vs fp8_linear_forward 模拟训练，最终 loss 差 < 30%（机制正确性弱门槛）；训练全程无 inf/nan |
| T8 | delayed scaling 兜底：构造 amax 突增 10^3 的 batch，当前步饱和不崩（无 inf），下一步 scale 已自适应 |

## 4. 产物

- `minilp/*.py` 全绿；`docs/8.2-low-precision/POSTMORTEM.md`
- 真 H100 FP8（Transformer Engine 对照、per-tensor 粒度实验）记 backlog
