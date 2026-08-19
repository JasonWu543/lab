# SPEC — Phase 1.3: 训练框架

> 状态：FROZEN（接口已冻结）
> 模式：Copilot —— 核心路径（checkpoint/resume、grad accum 等价性、
>       failure 处理）学生手写；dataloader/日志/配置 boilerplate 提示给足
> 算力：测试全部本地 CPU；U3.5 云端 L 级 capstone
> 工期：约 1 周（W4）

## 1. 问题

在 Phase 1.2 的模型之上生长出一个可信赖的单卡 Trainer。
验收标准不是「能跑」，是「出问题时能解释、能恢复、能复现」：

- **U3.3 硬门槛**：训练中断 → 从 checkpoint 恢复后，loss 轨迹与
  从未中断的基线**逐步 bit 级一致**（CPU 确定性模式下）。
- **U3.4 灵魂**：五种故障注入，每种都要能检测 → 报告 → 恢复或干净失败。

学完必须能回答（写进 POSTMORTEM）：
- 恢复一致性到底需要保存哪些状态？漏掉 RNG / dataloader 位置各会怎样？
- grad accumulation 什么情况下与大 batch 不等价？（提示：loss 归一化、
  BN 类统计、梯度裁剪的时机）
- 为什么 checkpoint 必须原子写入？非原子写会出什么事故？
- cosine + warmup 各自解决什么问题？

## 2. 范围与非目标

范围：memmap 数据管线、sequence packing、grad accumulation、clipping、
cosine scheduler、原子 checkpoint/resume（含全部 RNG）、异常检测、
纯文本日志（jsonl）。
非目标：不做多卡/DDP、不做 wandb 集成（jsonl 足够）、不做
BF16 autocast 的数值测试（本地 CPU 上意义不大，云端 capstone 再开）、
不做 evaluation harness（loss 即指标）。

## 3. 与其他 phase 的关系

- Trainer 对模型的要求只有「是 `nn.Module`、forward 返回 logits」——
  **测试用文件内定义的 tiny 模型**，不 import `minilm.model`，
  这样本 phase 的测试红绿只反映 Trainer 本身的对错。
- U3.5 正式训练时才把 Phase 1.2 的 MiniLM 和 Phase 1.0 的 tokenizer 接进来。

## 4. 冻结接口（minilm/training/）

```python
# minilm/training/data.py
def write_memmap(ids: Sequence[int], out_prefix: str | Path) -> None:
    """token ids → <prefix>.bin (uint16 little-endian) + <prefix>.meta.json
    （记录 dtype 与 token 数）。"""

class PackedDataset:
    """把长 token 流切成定长训练样本（相邻打包，不跨样本 shuffle token）。"""
    def __init__(self, bin_prefix: str | Path, seq_len: int): ...
    def __len__(self) -> int: ...          # floor((N - 1) / seq_len)
    def __getitem__(self, i) -> tuple[Tensor, Tensor]:
        """返回 (x, y)，y 是 x 右移一位；均为 int64，shape (seq_len,)。"""

def make_dataloader(dataset, batch_size: int, shuffle: bool,
                    seed: int, drop_last: bool = True) -> DataLoader:
    """shuffle 用 torch.Generator(seed) —— 同 seed 顺序必须可复现。"""

# minilm/training/scheduler.py
def lr_at(step: int, *, max_lr: float, min_lr: float,
          warmup_steps: int, total_steps: int) -> float:
    """[0, warmup) 线性升温从 0 到 max_lr；
    [warmup, total) 余弦从 max_lr 降到 min_lr；>= total 恒为 min_lr。"""

# minilm/training/checkpoint.py
def save_checkpoint(path: str | Path, *, model, optimizer,
                    step: int, extra: dict | None = None) -> None:
    """原子写入：先写 <path>.tmp，fsync 后 os.replace 到 path。
    内容含 model/optimizer state_dict、step、extra、
    以及全部 RNG 状态（torch / numpy / python random）。"""
def load_checkpoint(path: str | Path, *, model, optimizer) -> dict:
    """恢复所有状态（含 RNG），返回 {"step": ..., "extra": ...}。
    文件损坏（截断/校验失败）时 raise CheckpointCorruptError。"""
class CheckpointCorruptError(RuntimeError): ...

# minilm/training/trainer.py
@dataclass
class TrainConfig:
    max_steps: int
    micro_batch_size: int
    grad_accum_steps: int = 1
    max_lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 10
    grad_clip: float = 1.0
    seed: int = 42
    ckpt_every: int = 0            # 0 = 不自动存
    ckpt_path: str | None = None
    log_path: str | None = None    # jsonl，每 step 一行

class Trainer:
    def __init__(self, model: nn.Module, cfg: TrainConfig,
                 dataloader: DataLoader): ...
    def train_step(self) -> dict:
        """一个 optimizer step（内部循环 grad_accum_steps 个 micro batch）。
        返回 {"step", "loss", "lr", "grad_norm"}。
        loss 是 accum 各 micro batch 的平均（先除 accum 再 backward）。
        顺序：accum 完成 → clip（对整体梯度）→ optimizer.step → scheduler。
        异常检测：loss 非有限 → raise NonFiniteLossError（step 不推进）。"""
    def train(self) -> list[dict]: ...
    def save(self, path) / def resume(self, path): ...
class NonFiniteLossError(RuntimeError): ...
```

约定：

- 优化器固定 `torch.optim.AdamW(betas=(0.9, 0.95), weight_decay=0.1)`；
- 一切随机性来自 `cfg.seed`：模型初始化前 `torch.manual_seed`、
  dataloader 的 generator、（若有）dropout；
- 恢复一致性测试在 `torch.use_deterministic_algorithms(True)` + CPU 下比较；
- dataloader 位置的恢复：允许「按 step 数快进（重放跳过）」的简单实现，
  但必须在 POSTMORTEM 里讨论它在大数据集上的代价与替代方案。

## 5. 验收标准（tests/test_trainer.py）

| 编号 | 单元 | 通过条件 |
| --- | --- | --- |
| T1 | U3.1 | memmap 写读 roundtrip；PackedDataset 的 (x,y) 右移关系、边界样本数正确 |
| T2 | U3.1 | 同 seed 两次构建 dataloader，batch 顺序完全一致；不同 seed 不一致 |
| T3 | U3.2 | `lr_at` 曲线快照：warmup 端点、余弦中点、total 之后恒 min_lr（解析值对拍）|
| T4 | U3.2 | **grad accum 等价性**：accum=4×micro=2 与 accum=1×batch=8 相比，参数更新在 1e-6 内一致（同 seed 同数据）|
| T5 | U3.2 | clipping：构造大梯度，clip 后 global grad norm == grad_clip；小梯度不受影响 |
| T6 | U3.3 | **硬门槛**：train 20 步存 ckpt → 新进程语义下恢复再训 10 步，其 loss 序列与一次性训 30 步的第 21–30 步逐步相等 |
| T7 | U3.3 | 原子性：save 过程中断（注入：写 tmp 后 crash）→ 旧 checkpoint 完好可加载 |
| T8 | U3.4 | 损坏检测：截断的 ckpt 文件 → CheckpointCorruptError，而不是静默半加载 |
| T9 | U3.4 | 数据 NaN / loss 非有限 → NonFiniteLossError，参数未被污染（step 前后参数一致）|
| T10 | U3.4 | optimizer state 丢失（ckpt 里删掉 optimizer 键）→ 干净报错，不允许静默从零初始化 |

## 6. U3.5 云端 capstone（不进测试）

50–100M MiniLM + Phase 1.0 tokenizer + TinyStories train split，
云端单卡训练，产出 HF 格式 checkpoint + 训练报告
（loss 曲线、MFU、显存、一次真实的 kill -9 中断恢复记录）。
脚本 `scripts/train_capstone.py` 骨架由 Agent 提供，本地全绿才允许上云。

## 7. 产物

- `minilm/training/*.py`（学生实现）
- `docs/1.3-trainer/POSTMORTEM.md`（含第 1 节四问）
- U3.5 的训练报告 `docs/1.3-trainer/CAPSTONE_REPORT.md`
