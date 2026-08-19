# SPEC — Phase 1.1: 最小 Tensor 与 Autograd

> 状态：FROZEN（接口已冻结）
> 模式：Foundation —— 全部核心由学生手写；Agent 提供骨架/测试/参考答案
> 算力：纯本地（numpy 底层存储）
> 工期：约 4 天（W1 后半–W2 初）

## 1. 问题

用 numpy 作为存储层，实现一个最小但正确的 reverse-mode autograd 系统，
并在其上搭出 Linear/RMSNorm/SwiGLU 三个模块，用自己的系统把一个
两层 MLP 训到收敛。

学完必须能回答（写进 POSTMORTEM）：
- view/transpose 为什么不复制内存？什么时候必须复制？
- broadcast 的梯度为什么要沿扩张过的维度求和归约？
- backward 时每个 op 到底保存了什么？内存花在哪？
- inplace 修改为什么会破坏计算图？

## 2. 范围

- `Tensor`：包装 `np.ndarray`，`requires_grad`/`grad`/`shape`/`dtype`
- 前向 ops（全部支持 numpy 风格 broadcast）：
  `add, sub, mul, div, neg, pow(标量指数), matmul, exp, log,
   sum(dim, keepdim), mean(dim, keepdim), maximum(标量), sigmoid,
   reshape, transpose(dim0, dim1), getitem(切片)`
- reverse-mode autograd：动态构图 + 拓扑排序 backward + 梯度累积
  （同一 leaf 被多条路径使用时梯度相加）
- `backward()` 从标量输出发起（非标量需传 grad）
- `no_grad()` 上下文管理器
- `nn` 层：`Module` 基类（`parameters()` 递归收集）、
  `Linear`、`RMSNorm`、`SwiGLU`（含门控：`silu(W1 x) * (W3 x)` 再 `W2`）
- `SGD`（lr + 可选 momentum）
- MLP 收敛演示：两层 MLP 在合成二分类数据上 loss 下降到阈值

## 3. 非目标

- 不做二阶导 / retain_graph / GPU / 混合精度
- 不做完整 stride 系统：transpose/reshape 允许用 numpy 的
  view 语义实现，但你要在 POSTMORTEM 里解释 numpy 何时返回 view 何时复制
- 不做 conv / attention（那是 Phase 1.2 用 PyTorch 的事）
- 不追求性能，正确性唯一

## 4. 冻结接口（minilm/tensor/）

```python
# minilm/tensor/tensor.py
class Tensor:
    def __init__(self, data, requires_grad: bool = False): ...
    data: np.ndarray          # 前向值
    grad: np.ndarray | None   # 与 data 同 shape，backward 后填充
    requires_grad: bool
    @property
    def shape(self) -> tuple[int, ...]: ...

    def backward(self, grad: np.ndarray | None = None) -> None: ...
    def zero_grad(self) -> None: ...
    def detach(self) -> "Tensor": ...

    # ops：__add__/__radd__/__sub__/__mul__/__truediv__/__neg__/
    #      __pow__/__matmul__/__getitem__ 及
    #      exp/log/sigmoid/maximum/sum/mean/reshape/transpose

def no_grad(): ...            # 上下文管理器

# minilm/tensor/nn.py
class Module:
    def parameters(self) -> list[Tensor]: ...
    def __call__(self, *args): ...        # 转发到 forward
class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True): ...
class RMSNorm(Module):
    def __init__(self, dim: int, eps: float = 1e-6): ...
class SwiGLU(Module):
    def __init__(self, dim: int, hidden_dim: int): ...

# minilm/tensor/optim.py
class SGD:
    def __init__(self, params: list[Tensor], lr: float, momentum: float = 0.0): ...
    def step(self) -> None: ...
    def zero_grad(self) -> None: ...
```

约定：

- dtype 统一 float64（方便与有限差分对比；不做 dtype 提升系统）
- broadcast 梯度归约：反向时对扩张维度 `sum`，被 keepdim 挤掉的维度 reshape 回去
- `backward()` 只允许从 `requires_grad=True` 的图发起；对非标量不传 grad 要报错
- 拓扑排序保证每个节点的梯度只在全部下游贡献累积完后才向上传播

## 5. 验收标准（tests/test_tensor.py）

| 编号 | 测试 | 通过条件 |
| --- | --- | --- |
| T1 | 前向对齐 | 随机 shape（含 broadcast 组合）下所有 op 与 PyTorch forward 一致（rtol 1e-8）|
| T2 | 反向对齐 | 同上计算图与 PyTorch backward 的梯度一致 |
| T3 | 有限差分 gradcheck | 复合表达式上数值梯度 vs 解析梯度（rtol 1e-4）|
| T4 | broadcast 梯度 | (3,1)+(1,4)、标量+矩阵、keepdim 组合的梯度 shape 与数值正确 |
| T5 | 梯度累积 | 同一 leaf 进入多条路径；两次 backward 前不 zero_grad 则梯度翻倍 |
| T6 | 非连续 | transpose 后参与 matmul/加法，前反向仍正确 |
| T7 | 图与 no_grad | no_grad 下不建图；detach 切断梯度；非标量 backward 不传 grad 报错 |
| T8 | nn 层对齐 | Linear/RMSNorm/SwiGLU 前反向与 PyTorch 等价实现一致 |
| T9 | MLP 收敛 | 两层 MLP + SGD 在合成数据上 500 步内 loss < 0.05 |

## 6. U1.4 找 bug 演练（全绿后进行）

Agent 在你的实现拷贝里注入 3 个隐蔽 bug（broadcast 归约错 / 转置梯度错 /
梯度累积覆盖），你只准看失败的测试输出定位，每个写出机制解释。

## 7. 产物

- `minilm/tensor/{tensor,nn,optim}.py`（学生实现）
- `docs/1.1-tensor/POSTMORTEM.md`（含第 1 节四个问题的回答）
