# SPEC — Phase 8.1: 模型并行（Tensor Parallel / Pipeline Parallel）

> 状态：FROZEN
> 模式：Foundation —— 「哪一步需要哪种通信」是核心推导，学生手写；脚手架给定
> 算力：全 CPU。TP 用 gloo 多进程（复用 8.0 的 run_distributed）；
>       PP 用**单进程调度模拟**（重点是 micro-batch 调度与梯度等价，不是真通信）

## 1. 问题

- **TP（Megatron 风格）**：把 Linear 按列/行切到多个 rank。列切 + 行切串联
  组成 MLP 时，中间激活不需要通信、只在行切输出后 all-reduce 一次——
  为什么？forward/backward 各在哪里需要 all_gather / all_reduce / identity，
  是本 phase 的核心推导。
- **PP（GPipe 风格）**：层按 stage 切分，batch 切成 micro-batch 流水。
  bubble 占比闭式 (p−1)/(m+p−1)，与调度模拟对拍；micro-batch 累积梯度
  必须与全量 batch 一致。

学完必须能回答（写进 POSTMORTEM）：
- 列切 Linear 的 backward 里，对输入的梯度为什么需要 all-reduce？手推。
- TP 和 PP 各自的通信量随什么增长？什么形状的模型该用哪个？
- 1F1B 相对 GPipe 省的是什么（bubble 还是峰值激活显存）？

## 2. 冻结接口

```python
# minitp/layers.py —— 学生实现（gloo，多进程）
class ColumnParallelLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, world_size, rank, seed):
        """完整权重 (out, in) 用 seed 确定性生成后按行切块（对应输出列切分），
        本 rank 持有第 rank 块。冻结初始化：CPU Generator manual_seed(seed)、
        torch.randn * 0.02、无 bias。gather_output=False：forward 返回本地分片。"""
    def forward(self, x): ...

class RowParallelLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, world_size, rank, seed):
        """完整权重按列切块（对应输入维切分），初始化规则同 Column；输入已是分片；
        forward 内部 all_reduce 本地部分积后返回完整输出。TPMLP 中 Row 使用
        seed+1，Column 使用 seed。"""
    def forward(self, x): ...

class TPMLP(torch.nn.Module):
    """Column(d→4d, GELU) → Row(4d→d)。backward 对输入梯度的 all_reduce
    由学生用 autograd.Function 实现（_CopyToTP / _ReduceFromTP 两个原语）。"""

# minipp/schedule.py —— 学生实现（单进程）
def gpipe_schedule(num_stages: int, num_microbatches: int) -> list[list[tuple]]:
    """返回逐 tick 的执行表：每 tick 是 [(stage, mb, 'F'|'B'), ...]。
    约束：mb 的 F 在 stage s 依赖其在 s-1 的 F；B 反向依赖；每 stage 每 tick
    至多做一件事；B 全部排在该 mb 的最后一个 F 之后（GPipe：flush 式）。"""

def bubble_fraction(num_stages: int, num_microbatches: int) -> float:
    """闭式 (p-1)/(m+p-1)。"""

# minipp/runner.py —— 学生实现（单进程）
class PipelineRunner:
    def __init__(self, stages: list[torch.nn.Module]): ...
    def train_step(self, x, y, num_microbatches: int,
                   loss_fn) -> torch.Tensor:
        """把 batch 均分为 micro-batch 逐个 fwd/bwd，梯度累积；
        返回全 batch 平均 loss。等价性要求见 T4。"""
```

## 3. 验收标准（tests/test_tp_pp.py）

| 编号 | 通过条件 |
| --- | --- |
| T1 | **TP ≡ 单进程**（world_size=2，gloo）：TPMLP forward 输出与同 seed 重建的完整 MLP 逐位一致（fp32 求和顺序差异允许 atol 1e-6）|
| T2 | TP backward：对权重分片的梯度 = 完整 MLP 对应块的梯度；**对输入的梯度**与完整 MLP 一致（漏掉输入梯度 all_reduce 必挂此项）|
| T3 | gpipe_schedule：依赖约束逐 tick 机器检查（枚举断言）；总 tick 数 = 闭式 2(m+p−1)（F、B 等耗时假设）；bubble_fraction 与调度表实际空闲比例对拍 |
| T4 | **PP 梯度等价**：PipelineRunner 以 m∈{1,2,4} 训练一步；m=1 与整批 backward `torch.equal`，m>1 以 `rtol=1e-6, atol=1e-7` 对齐（loss 取均值口径）。测试须记录 loss_fn 的调用形状，确认确实执行 m 个 B/m micro-batch，并核对返回的全 batch mean loss |
| T5 | 边界：m=1（无流水收益）、partition 中出现空 `nn.Sequential` 必须 raise、batch 不能整除 m 必须 raise |
| T6 | 确定性：同 seed 两次 TP/PP 运行结果逐位一致 |

多进程 spawn ≤ 3 次，全套 CPU < 90s。

## 4. 产物

- `minitp/*.py`、`minipp/*.py` 全绿；`docs/8.1-tp-pp/POSTMORTEM.md`（含两条手推）
- 1F1B 调度、真实多卡吞吐记 backlog
