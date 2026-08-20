# SPEC — Phase 4.0: mini-vLLM（naive → continuous batching → paged KV → prefix cache）

> 状态：FROZEN（接口已冻结）
> 模式：Lead 倾向 —— 本 phase 是 M7 Phase 7.1 的主战场：接口冻结后可以由你
>       指挥多个 Agent 并发实现（BlockManager / Scheduler / Sampler / Engine），
>       你负责接口、review、集成；也可以全部手写。两种走法都以同一套测试验收。
> 基座：官方 transformers 实现（测试用 tiny 随机初始化 Qwen2Config 模型，
>       CPU 秒级；正式 benchmark 换 Qwen2.5-0.5B 真权重）。**不重写模型结构。**
> 算力：correctness 全本地；正式 benchmark M 级
> 工期：约 2 周（W5–6 主线）

## 1. 问题

从零搭一个单卡推理引擎，走完 vLLM 的核心思想链：
KV cache 复用 → iteration-level continuous batching → 逻辑分页的
block 管理 → 跨请求 prefix cache。全程记录 TTFT/TPOT/tokens/s。

学完必须能回答（写进 POSTMORTEM）：
- continuous batching 相对 static batching 的收益来自哪里？什么 workload 下没收益？
- paged KV 解决的是什么问题？block_size 的权衡是什么？
- prefix cache 的 refcount 什么时候必须为 0 才能 free？搞错会发生什么？
- 你的实现和真 vLLM 的最大差距在哪一层？（提示：本实现每步 gather 物化 KV）

## 2. 范围与非目标

范围：FCFS 调度、prefill/decode 混合 step、按 token 预算的准入、
逻辑分页 + hash 前缀复用、温度/top-p 采样、per-request 指标。
非目标：不做抢占/swap-out、不做 chunked prefill（4.2 backlog）、
不做真 paged attention kernel（物理布局用 gather 物化模拟，
差距在 POSTMORTEM 讨论）、不做 streaming API/HTTP 层。

## 3. 冻结接口（minivllm/）

```python
# minivllm/request.py
class SeqStatus(Enum): WAITING; RUNNING; FINISHED

@dataclass
class Request:
    req_id: int
    prompt_ids: list[int]
    max_new_tokens: int
    temperature: float = 0.0          # 0 = greedy
    top_p: float = 1.0
    arrival_step: int = 0             # 到达时刻（用引擎 step 数模拟时钟）

class Sequence:
    """一个请求的运行态。属性：request, status, output_ids: list[int],
    block_table: list[int], num_cached_tokens: int（prefix cache 命中数）。
    方法：num_tokens（prompt+output）、last_token、is_finished。"""

# minivllm/block_manager.py
class BlockManager:
    def __init__(self, num_blocks: int, block_size: int): ...
    @property
    def num_free_blocks(self) -> int: ...
    def can_allocate(self, seq: Sequence) -> bool:
        """prefill 准入检查（考虑 prefix cache 命中后实际需要的新块数）。"""
    def allocate(self, seq: Sequence) -> None:
        """为 prefill 分配 block_table；命中前缀的块直接复用（refcount+1）。"""
    def can_append(self, seq: Sequence) -> bool: ...
    def append_slot(self, seq: Sequence) -> None:
        """decode 一个新 token；跨块边界时分配新块。"""
    def free(self, seq: Sequence) -> None:
        """refcount-1，归零的块回收；同时更新前缀哈希表。"""
    # prefix cache（块粒度 hash：key = 该块及其之前所有 token 的元组哈希）
    def match_prefix(self, token_ids: list[int]) -> tuple[list[int], int]:
        """返回 (可复用的 block ids, 命中的 token 数（block_size 的整数倍）)。"""

# minivllm/scheduler.py
class Scheduler:
    def __init__(self, block_manager: BlockManager, max_batch_tokens: int): ...
    def add(self, seq: Sequence) -> None:            # 进 waiting 队列（FCFS）
    def schedule(self) -> tuple[list[Sequence], list[Sequence]]:
        """返回 (本步 prefill 的序列, 本步 decode 的序列)。
        规则：先保证 RUNNING 的都能 decode（不能 append 的本实现直接报错，
        不做抢占）；剩余 token 预算内按 FCFS 准入 waiting 的 prefill。"""
    def finish(self, seq: Sequence) -> None:         # 释放块、移出 running

# minivllm/sampler.py
def sample(logits: Tensor, seqs: list[Sequence],
           generator: torch.Generator | None = None) -> list[int]:
    """按每个 seq 自己的 temperature/top_p 采样；temperature=0 为 argmax。"""

# minivllm/engine.py
class Engine:
    def __init__(self, model, num_blocks: int = 256, block_size: int = 16,
                 max_batch_tokens: int = 2048,
                 generator: torch.Generator | None = None): ...
    def add_request(self, req: Request) -> None: ...
    def step(self) -> list[Sequence]:
        """一个 iteration：schedule → 拼 batch → 模型前向 → 采样 →
        更新各 seq → 返回本步完成的序列。混合 prefill+decode 允许
        分两次前向（简化），但必须同一个 step 内完成。"""
    def run(self) -> dict[int, list[int]]:
        """跑到全部完成，返回 {req_id: output_ids}。"""
    def metrics(self) -> dict[int, dict]:
        """{req_id: {"ttft_steps": int, "tpot_steps": float, "n_out": int}}
        用 step 数当时钟：ttft = 首 token 产出步 - arrival_step。"""

# minivllm/cache_adapter.py（唯一碰模型输入的地方，Copilot 给足提示）
def gather_past_kv(paged_k, paged_v, seqs) -> "DynamicCache":
    """从分页存储按 block_table gather 出各 seq 的连续 KV，
    组装成 transformers 的 past_key_values。物理上是拷贝——
    真 vLLM 靠 paged attention kernel 原地读，差距写进 POSTMORTEM。"""
```

约定：engine 内部的分页 KV 存储 shape 为
`(num_layers, 2, num_blocks, block_size, n_kv_heads, head_dim)`；
模型前向后把新 KV 写回对应 block（scatter）；
正确性的锚点是「与 HF `model.generate` 逐请求串行的结果完全一致」。

## 4. 验收标准（tests/test_minivllm.py，tiny Qwen2Config CPU）

| 编号 | 通过条件 |
| --- | --- |
| T1 | BlockManager 不变量：分配不重叠、free 后可复用、refcount 正确；块不足时 can_allocate=False 而不是崩溃 |
| T2 | **正确性锚点**：8 个不同长度请求（greedy），continuous batching 的输出与 HF 逐请求 `generate` 完全一致 |
| T3 | 动态性：错峰到达（arrival_step 不同）的请求在中途被准入；存在同一 step 既有 prefill 又有 decode |
| T4 | 分页正确性：构造 block_table 非连续（先分配干扰序列再 free）的场景，输出仍与串行一致 |
| T5 | prefix cache：共享长前缀的第二个请求 num_cached_tokens > 0、输出与不开缓存一致；free 后 refcount 归零、块可回收 |
| T6 | sampler：top-p 截断正确、固定 generator 可复现、逐 seq 参数生效 |
| T7 | metrics：错峰请求的 ttft 单调合理（后到的 ttft 不小于先到的同长请求）、tpot > 0 |

正式 benchmark（`benchmarks/bench_engine.py`，不进测试）：
Qwen2.5-0.5B 真权重 + 泊松到达 workload，报告 tokens/s、TTFT/TPOT p50/p95、
并发退化曲线；终点是与 vLLM 部署同模型对比并解释差距。

## 5. 产物

- `minivllm/*.py` 全绿 + benchmark 报告
- `docs/4.0-minivllm/POSTMORTEM.md`（第 1 节四问 + 若走 M7 并发开发路线，
  附 7.1 的度量记录：并行加速比 / 接口返工次数 / 逃逸 bug）
