# SPEC — Phase 8.0: 数据并行（梯度桶 DDP / ZeRO-1 / ZeRO-2 语义）

> 状态：FROZEN
> 模式：Foundation —— 通信下的等价性推导与分片逻辑全部手写；进程管理脚手架给定
> 算力：全 CPU（torch.distributed gloo backend，多进程模拟多卡）
> 环境约定：mp spawn 启动、`init_method="file://<tmpdir>/rdv"`、world_size ≤ 4、
>           每个多进程测试设 60s 超时

## 1. 问题

数据并行的正确性核心是一条不变量：**k 个 rank 各算 1/k 数据的平均梯度，
all-reduce 求均值后，必须与单进程算全量数据的梯度数值等价**（loss 为 batch 均值、
各 rank 等量数据时成立）。由于整批 reduction 与“分片 reduction 后再 SUM”的 fp32
结合顺序不同，对单进程 oracle 使用 `rtol=1e-6, atol=1e-7`；相同分布式路径的 rank
之间、不同 bucket cap 之间仍要求 `torch.equal`。ZeRO 在此之上把 optimizer state
（ZeRO-1）和梯度（ZeRO-2）分片，用通信换显存。

学完必须能回答（写进 POSTMORTEM）：
- 为什么 DDP 梯度要按参数**逆序**分桶？桶太大/太小各损失什么？
- ZeRO-1/2/3 各分片了什么？各自增加了哪次通信？
- 上述数值等价在什么条件下会被打破（数据不均分、loss 用 sum、bf16）？为什么
  正确 fp32 实现也不一定与整批 oracle 逐位一致？

## 2. 冻结接口（minidist/）

```python
# minidist/comm.py —— 给定脚手架
def run_distributed(fn, world_size: int, *args) -> list:
    """mp.spawn + gloo + file init 启动 fn(rank, world_size, *args)，
    收集各 rank 返回值（经 mp.Queue）。测试统一用它。"""

# minidist/bucket.py —— 学生实现
def partition_buckets(params: list[torch.nn.Parameter],
                      bucket_cap_bytes: int) -> list[list[int]]:
    """按参数列表**逆序**贪心装桶：当前桶字节数超过 cap 即封桶开新桶。
    返回每桶的参数下标列表（下标为原列表顺序）。单参数超 cap 时独占一桶。"""

def allreduce_gradients(model: torch.nn.Module,
                        bucket_cap_bytes: int) -> None:
    """对 model 所有 requires_grad 参数的 .grad 按 partition_buckets 分桶，
    每桶 flatten 成单一 tensor 做一次 dist.all_reduce(SUM)，除以 world_size
    后写回各 .grad。grad 为 None 的参数视为零参与。"""

# minidist/zero.py —— 学生实现
def shard_params(params: list[torch.nn.Parameter],
                 world_size: int) -> list[list[int]]:
    """确定性贪心负载均衡：按 numel 降序遍历，每个参数分给当前总 numel
    最小的 rank（并列取 rank 小者）。返回每 rank 的参数下标列表。"""

class Zero1Optimizer:
    def __init__(self, params: list[torch.nn.Parameter], lr: float,
                 betas=(0.9, 0.999), eps=1e-8):
        """AdamW 语义（weight_decay=0），但只为本 rank 拥有（shard_params）的
        参数维护 exp_avg/exp_avg_sq。"""
    def step(self) -> None:
        """本 rank 对 owned 参数做 AdamW update，然后逐参数从 owner rank
        broadcast 更新后的参数数据到所有 rank。"""
    def state_bytes(self) -> int:
        """本 rank optimizer state 实际字节数（闭式可对拍）。"""

def zero2_reduce_gradients(model, shards: list[list[int]]) -> None:
    """ZeRO-2 语义版：all_reduce 求平均后，非 owned 参数的 .grad 置 None
    （模拟 reduce-scatter 的显存效果；gloo 无高效 reduce_scatter）。"""
```

## 3. 验收标准（tests/test_dist.py）

| 编号 | 通过条件 |
| --- | --- |
| T1 | partition_buckets 闭式：手工参数尺寸下桶划分/顺序/超大参数独占逐一对拍；cap 边界（恰好等于/差 1 字节）|
| T2 | **DDP ≡ 单进程**：固定 seed 小 MLP，batch 8 均分 2 rank；allreduce 后各 rank 梯度与单进程全量 batch 梯度以 `rtol=1e-6, atol=1e-7` 对齐；bucket_cap 取 {1 参数/桶, 全参数 1 桶, 中间值}，三档之间及各 rank 之间仍须 `torch.equal` |
| T3 | **ZeRO-1 轨迹 ≡ 单进程 AdamW**：20 步训练（world_size=2，等量数据）后所有 rank 的模型参数与单进程 `torch.optim.AdamW`（同超参、weight_decay=0）以 `rtol=1e-6, atol=1e-7` 对齐，且两 rank 间参数逐位一致 |
| T4 | shard_params：构造尺寸下分配结果确定且各 rank numel 差 ≤ 最大单参数 numel；world_size=1 退化为全量 |
| T5 | ZeRO-1 state_bytes：与闭式（owned numel × 2 状态 × 4 字节 + step 计数）对拍；两 rank 之和 = 单进程 AdamW 全量 state |
| T6 | ZeRO-2：调用后 owned 参数 grad 等于全量平均梯度、非 owned 为 None |
| T7 | world_size=1 全链路退化：DDP/ZeRO-1 与朴素单进程完全一致 |
| T8 | 确定性：同 seed 跑两次 run_distributed，各 rank 返回值逐位一致 |

多进程测试控制在 ≤6 个 spawn（每次 spawn 数秒），全套 CPU < 120s。

## 4. 产物

- `minidist/*.py` 全绿；`docs/8.0-ddp-zero/POSTMORTEM.md`
- 真实多卡带宽/加速比测量记入 backlog（租多卡时补 BASELINE）
