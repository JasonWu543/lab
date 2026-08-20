# SPEC — Phase 6.1: Scaling Law（拟合 / compute-optimal / 外推验证）

> 状态：FROZEN（接口已冻结）
> 模式：Foundation —— 拟合与最优配比的闭式推导全部手写
> 算力：拟合数学全本地（合成数据 + 公开数据点）；
>       isoFLOP 真实实验 L 级（W12 可选项，本 phase 唯一的 GPU 部分）
> 工期：数学部分约 2 天；实验部分 W12 视预算

## 1. 问题

复刻 Chinchilla 方法论的最小版：拟合参数化损失面
**L(N, D) = E + A/N^α + B/D^β**，推导固定算力 C ≈ 6ND 下的
compute-optimal 配比 (N*, D*)，并外推到没训过的算力点。
correctness 全部在合成数据上验证（已知真参数能否回收），
真实 isoFLOP 实验是可选的 capstone。

学完必须能回答（写进 POSTMORTEM）：
- 从 L(N,D) 和约束 C=6ND 出发，手推 N* ∝ C^a 的闭式（a 用 α/β 表示）；
- 为什么拟合要在 log 空间做、用 Huber loss？普通 MSE 会被什么带偏？
- Kaplan 和 Chinchilla 的结论差异来自方法的哪一步？
- 你的外推在什么范围内可信？什么时候必然失效？

## 2. 范围与非目标

范围：损失面拟合（多起点 L-BFGS）、闭式最优配比、isoFLOP 谷底提取、
外推 + 置信讨论、实验脚手架。
非目标：不做 μP / 学习率 scaling、不做数据受限重复 token 的
scaling（Muennighoff 2023，backlog）、不做超过 4 组规模的真实训练。

## 3. 冻结接口（minilaw/）

```python
# minilaw/fit.py
@dataclass
class ScalingParams:
    E: float; A: float; alpha: float; B: float; beta: float

def predict_loss(N, D, p: ScalingParams):
    """L = E + A/N^α + B/D^β。N/D 可为标量或 np.ndarray。"""

def fit_scaling_law(N: np.ndarray, D: np.ndarray, L: np.ndarray,
                    n_starts: int = 16, seed: int = 0) -> ScalingParams:
    """Chinchilla Approach 3 风格：
    在 log 空间参数化（拟合 log A、log B、log E 保证正性），
    目标 = Huber(delta=1e-3) on (log L_pred − log L_obs)，
    多起点（对 α,β ∈ [0,2.5]、logA/logB 网格随机）取最优。
    确定性：同 seed 同输入必须同输出。"""

# minilaw/optimal.py
def compute_optimal(C: float, p: ScalingParams) -> tuple[float, float]:
    """min_{N,D} L s.t. 6ND = C 的闭式解 (N*, D*)。
    要求解析推导（不许数值搜索）——测试会拿数值网格搜索对拍你的闭式。"""

def optimal_exponents(p: ScalingParams) -> tuple[float, float]:
    """返回 (a, b)：N* ∝ C^a, D* ∝ C^b。验证 a + b = 1。"""

# minilaw/isoflop.py
def isoflop_minima(runs: list[dict]) -> list[dict]:
    """runs: [{"C": float, "N": float, "L": float}, ...]（同一 C 多个 N）。
    每个 C 组内对 (log N, L) 做抛物线拟合取谷底，
    返回 [{"C", "N_opt", "L_min"}, ...]。组内点数 < 3 时跳过该组。"""

# scripts/run_isoflop.py —— 实验脚手架（完整）：4 个 FLOPs 预算 × 每档 4 个
#   模型尺寸的 isoFLOP 网格（10–80M，HF tiny 架构 + TinyStories），
#   产出 runs.json 喂给上面的纯数学部分；W12 租卡再跑
```

## 4. 验收标准（tests/test_scaling.py，CPU，全合成数据）

| 编号 | 通过条件 |
| --- | --- |
| T1 | predict_loss 闭式对拍（标量与数组广播）|
| T2 | **参数回收**：用已知 ScalingParams（取 Chinchilla 论文量级：E≈1.7, α≈0.34, β≈0.28）生成 40 个 (N,D,L) 点 + 2% 乘性噪声，拟合回收 α/β 误差 < 0.05、E 误差 < 0.2（E 与 A/B 存在强交叉相关，40 点收不到更紧——这本身是个思考点）|
| T3 | 拟合确定性：同 seed 两次拟合结果逐位一致；抗离群点：污染 2 个 10× 离群 L 后 α/β 仍在容差内（Huber 的意义）|
| T4 | **闭式 vs 数值**：compute_optimal 与网格搜索（每个 C 上 10^4 点）的最优 N 相对误差 < 1%（3 个不同数量级的 C）|
| T5 | optimal_exponents：a+b = 1（1e-9 容差）；α=β 时 a=b=0.5 |
| T6 | isoflop_minima：合成抛物线谷底回收误差 < 5%；点数不足的组被跳过不崩 |
| T7 | **端到端外推**：合成数据只用小算力点（C ≤ 1e18）拟合，外推预测 C=1e19 的 (N*, L)，与真参数生成值误差 < 10% |

## 5. 产物

- `minilaw/*.py` 全绿
- `docs/6.1-scaling/POSTMORTEM.md`（含 N* ∝ C^a 的完整手推）
- （可选 W12）真实 isoFLOP 报告：外推的下一档配置 vs 实际训练验证
