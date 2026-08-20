# SPEC — Phase 4.2: PD 分离（功能模拟版）

> 状态：FROZEN（接口已冻结）
> 模式：Lead 倾向（同 4.0：系统管线，骨架提示可以给密；调度核心逻辑仍由学生实现）
> 算力：纯 CPU 离散时间模拟，无真实模型
> 性能版（双卡真实 KV transfer）为可选扩展，不进主线

## 1. 问题

Prefill 与 Decode 的算力特征完全不同（compute-bound vs memory-bound），
合并在一个 worker 上互相干扰：长 prompt 的 prefill 会阻塞正在 decode 的请求，
拉高 TPOT 尾延迟。PD 分离（DistServe / Mooncake 思路）把两阶段拆到独立 worker，
代价是 KV cache 要跨 worker 迁移。本 phase 用**离散时间模拟**建模这套调度，
不跑真模型，只回答调度问题。

学完必须能回答（写进 POSTMORTEM）：
- PD 分离改善的是 TTFT 还是 TPOT？为什么合并部署下 TPOT 尾部会被 prefill 打爆？
- chunked prefill 与 PD 分离各自解决什么？什么时候二者互补/互替？
- KV transfer 的延迟什么时候会吃掉分离的收益（推一个盈亏平衡条件）？

## 2. 模拟模型（冻结语义）

- 时间为整数 tick。每 tick：先 transfer 推进，再 decode step，再 prefill step，
  再调度器做 admission/迁移决策（顺序冻结，保证确定性）。
- PrefillWorker：每 tick 最多处理 `prefill_tokens_per_tick` 个 prompt token，
  同一 tick 可跨请求分配（按 FIFO），单请求单 tick 最多消耗 `chunk_size` token
  （chunked prefill）。
- DecodeWorker：最多 `decode_slots` 个并发请求，每 tick 每个在座请求产 1 token。
- Transfer：prefill 完成 → 迁移占 `transfer_ticks(model, prompt_len)` 个 tick →
  到达 decode 侧等待 slot。迁移通道并发不限（简化）。
- TTFT(req) = 第一个 output token 产出的 tick − arrival_time
  （decode 第一步产出即首 token；prefill 不产 token）。
- TPOT(req) = (finish_time − first_token_time) / max(1, max_new_tokens − 1)。

## 3. 冻结接口（minipd/）

```python
# minipd/request.py —— 给定
@dataclass
class PDRequest:
    req_id: int
    arrival_time: int
    prompt_len: int
    max_new_tokens: int
    # 运行时字段（给定默认值）：prefilled_tokens=0, generated_tokens=0,
    # first_token_time=None, finish_time=None

# minipd/transfer.py
@dataclass
class TransferModel:
    base_ticks: int
    bytes_per_token: int
    bandwidth_bytes_per_tick: int

def transfer_ticks(model: TransferModel, num_tokens: int) -> int:
    """base_ticks + ceil(num_tokens * bytes_per_token / bandwidth_bytes_per_tick)"""

# minipd/workers.py —— 学生实现
class PrefillWorker:
    def __init__(self, prefill_tokens_per_tick: int, chunk_size: int): ...
    def add(self, req: PDRequest) -> None: ...
    def step(self, t: int) -> list[PDRequest]:
        """推进一个 tick，返回本 tick prefill 恰好完成的请求（FIFO 分配算力）。"""

class DecodeWorker:
    def __init__(self, decode_slots: int): ...
    def add(self, req: PDRequest) -> None:
        """slot 满时进入内部 FIFO 等待队列。"""
    def step(self, t: int) -> list[PDRequest]:
        """在座请求各产 1 token；记录 first_token_time；返回本 tick 完成的请求，
        完成即释放 slot 并按 FIFO 补位（补位请求下一 tick 才开始产 token）。"""

# minipd/scheduler.py —— 学生实现（核心）
class PDScheduler:
    def __init__(self, prefill: PrefillWorker, decode: DecodeWorker,
                 transfer: TransferModel): ...
    def submit(self, req: PDRequest) -> None: ...
    def tick(self, t: int) -> None:
        """按 §2 冻结顺序推进一个 tick，管理 transfer 在途队列。"""
    def all_done(self) -> bool: ...

# minipd/simulate.py
@dataclass
class SimResult:
    ttft: dict[int, int]        # req_id -> ticks
    tpot: dict[int, float]
    finish: dict[int, int]
    total_ticks: int
    throughput: float           # 总产出 token / total_ticks

def run_sim(requests: list[PDRequest], prefill_tokens_per_tick: int,
            chunk_size: int, decode_slots: int,
            transfer: TransferModel) -> SimResult: ...

def run_colocated_baseline(requests: list[PDRequest], tokens_per_tick: int,
                           decode_slots: int) -> SimResult:
    """合并部署 baseline（给定/脚手架）：同一算力池，prefill 与 decode 抢
    tokens_per_tick，prefill 优先（这正是打爆 TPOT 的策略）。无 transfer。"""

def make_workload(n: int, seed: int, arrival_rate: float,
                  prompt_len_range: tuple[int, int],
                  gen_len_range: tuple[int, int]) -> list[PDRequest]:
    """给定：确定性合成 workload（numpy RNG，泊松到达取整）。"""
```

## 4. 验收标准（tests/test_pd.py，CPU）

| 编号 | 通过条件 |
| --- | --- |
| T1 | transfer_ticks 闭式对拍（含整除/非整除、num_tokens=0 → base_ticks）|
| T2 | chunked prefill：prompt_len=10, chunk_size=4, 算力充足 → 恰好 3 tick 完成；两请求并发时 FIFO 分配可手算验证 |
| T3 | **单请求端到端手算对拍**：给定全部参数，TTFT / TPOT / finish 与手推逐 tick 时间线精确相等（表格写进测试注释）|
| T4 | decode_slots=1、两请求：第二个在等待队列，first_token_time 严格晚于第一个 finish；补位请求下一 tick 才产 token |
| T5 | 确定性：同 workload 两次 run_sim 结果逐字段相等；make_workload 同 seed 复现 |
| T6 | chunked prefill 的意义：长 prompt 先到 + 短请求后到，chunk_size 小的配置下短请求 TTFT 严格更低 |
| T7 | **PD 分离 vs 合并**：构造 decode 在座时持续来长 prefill 的 workload，分离配置下最大 TPOT 严格低于 colocated（decode 不再被 prefill 抢占）|
| T8 | transfer 盈亏：把 bandwidth 调到极低，分离的 TTFT 反超 colocated（收益被迁移吃掉）——断言方向翻转 |
| T9 | 边界：max_new_tokens=1（首 token 即完成）、prompt_len=1、空 workload 不崩 |

## 5. 产物

- `minipd/*.py` 全绿
- `benchmarks/bench_pd.py`：扫 arrival_rate × chunk_size，输出 TTFT/TPOT 分位数表
- `docs/4.2-pd-sim/POSTMORTEM.md`（含盈亏平衡条件的推导）
