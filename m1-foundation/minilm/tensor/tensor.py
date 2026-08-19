"""Phase 1.1 — 最小 Tensor 与 Autograd（学生实现文件）

你的任务：实现 Tensor 类与 no_grad，让 tests/test_tensor.py 的 T1–T9 全绿。

闯关顺序建议：
  Step 1  标量/逐元素 ops（add/sub/mul/div/neg/pow/exp/log/sigmoid/maximum）
           → 先过 T1 的基础 forward，再过 T2 的基础 backward
  Step 2  backward 引擎（拓扑排序 + 梯度累积）
           → 过 T2/T3/T5
  Step 3  broadcast 支持（add/sub/mul/div 的梯度归约）
           → 过 T4
  Step 4  matmul / reshape / transpose
           → 过 T6
  Step 5  sum / mean（带 dim/keepdim 参数）
           → 完善 T1/T2
  Step 6  nn 层（Linear / RMSNorm / SwiGLU）+ SGD
           → 过 T8
  Step 7  MLP 端到端训练
           → 过 T9

关键约定（实现前务必读完）：

1. dtype 统一 np.float64
   - __init__ 里：self.data = np.array(data, dtype=np.float64)

2. 每个 op 需要做三件事（op 注册的通用模式）：
   a. 前向：用 numpy 计算结果值 → 存入新 Tensor 的 .data
   b. 记录 parents：result._parents = {self, other} 或 {self}
   c. 注册 backward 闭包：result._backward = lambda: ...
      闭包里对每个 requires_grad=True 的 parent，
      将 upstream_grad（= result.grad）乘以局部偏导，
      累加到 parent.grad（注意：不是赋值，是 +=，
      这样同一 leaf 进入多条路径时梯度自动累积）
   result.requires_grad = any(p.requires_grad for p in parents)

3. broadcast 梯度归约：
   - 当 parent 的 shape 与 result.grad 的 shape 不一致时，
     需要沿「被 broadcast 扩张」的维度对梯度求 sum
   - 工具函数思路：先 sum 掉多余的前缀维度，再把原本为 1 的维度 sum 且保持 keepdim
   - 最后 .reshape(parent.shape) 确保梯度 shape 完全对齐

4. 拓扑排序提示：
   - 对计算图做后序 DFS（先访问 parents，再 append 自己）
   - 得到 topo_order 后 reverse，从输出节点向输入节点传梯度
   - 每个节点的 _backward() 在 topo_order 中只调用一次

5. backward() 发起条件：
   - 只对 requires_grad=True 的输出发起
   - 若输出非标量（shape != ()），必须传入 grad 参数，否则 raise RuntimeError

6. no_grad() 是上下文管理器：
   - 进入时全局禁止建图（_NO_GRAD_MODE = True）
   - 退出时恢复（_NO_GRAD_MODE = False）
   - 在 no_grad 下创建的 op 结果的 requires_grad 强制为 False，且不注册 backward

运行测试：
  cd m1-foundation && python3 -m pytest tests/test_tensor.py -x -q

卡住了再看参考答案：reference/1.1-tensor/tensor_solution.py
（先自己写；对完答案要能说出差在哪）
"""
from __future__ import annotations

from contextlib import contextmanager

import numpy as np

# ---------- 全局 no_grad 标志 ----------

_NO_GRAD_MODE = False


@contextmanager
def no_grad():
    """上下文管理器：进入后所有 op 不建计算图。

    使用示例：
        with no_grad():
            y = model(x)   # y.requires_grad == False，不占图内存
    """
    # 提示：用 global _NO_GRAD_MODE，进入时设 True，退出时恢复
    raise NotImplementedError


# ---------- Tensor ----------

class Tensor:
    """numpy 底层存储的 Tensor，支持 reverse-mode autograd。

    核心属性：
        data          : np.ndarray，前向计算值（float64）
        grad          : np.ndarray | None，backward 后填充，与 data 同 shape
        requires_grad : bool，是否需要梯度
        _parents      : set[Tensor]，直接上游节点（叶节点为空集）
        _backward     : callable，当前节点的反向传播闭包
    """

    def __init__(self, data, requires_grad: bool = False):
        """初始化 Tensor。

        提示：
          - np.array(data, dtype=np.float64) 保证 dtype 统一
          - grad 初始化为 None（backward 后才填充）
          - _parents 和 _backward 是建图用的内部状态
        """
        raise NotImplementedError

    # ------ 基础属性 ------

    @property
    def shape(self) -> tuple[int, ...]:
        """返回 data 的 shape（与 numpy 一致）。"""
        raise NotImplementedError

    @property
    def dtype(self):
        raise NotImplementedError

    # ------ 图操作 ------

    def backward(self, grad: np.ndarray | None = None) -> None:
        """从当前节点发起反向传播。

        提示：
          1. 若 grad 为 None 且 self 是标量（self.data.ndim == 0 或 shape==()），
             则 upstream grad = np.ones((), dtype=np.float64)
          2. 若 grad 为 None 且 self 非标量 → raise RuntimeError
          3. 拓扑排序（后序 DFS）→ reverse → 依次调用各节点 _backward()
          4. 注意：_backward() 的调用顺序必须从输出到输入
        """
        raise NotImplementedError

    def zero_grad(self) -> None:
        """将 grad 清零（置 None 或置全 0 均可）。"""
        raise NotImplementedError

    def detach(self) -> "Tensor":
        """返回与当前 data 共享存储、但脱离计算图的新 Tensor。

        提示：新 Tensor 的 requires_grad=False，_parents=set()
        注意：共享 data（不复制），改 data 会互相影响——这是故意的
        """
        raise NotImplementedError

    # ------ 逐元素 ops ------

    def __add__(self, other) -> "Tensor":
        """前向：self.data + other.data（支持 broadcast + 标量）。

        提示（通用模式）：
          other = other if isinstance(other, Tensor) else Tensor(other)
          result = Tensor(self.data + other.data)
          记录 result._parents = {self, other}
          注册 result._backward = lambda: 对 self/other 累加梯度
          broadcast 梯度归约：若 self.shape != result.shape，
            对多余维度 sum 后 reshape 回 self.shape
        """
        raise NotImplementedError

    def __radd__(self, other) -> "Tensor":
        raise NotImplementedError

    def __sub__(self, other) -> "Tensor":
        raise NotImplementedError

    def __rsub__(self, other) -> "Tensor":
        raise NotImplementedError

    def __mul__(self, other) -> "Tensor":
        raise NotImplementedError

    def __rmul__(self, other) -> "Tensor":
        raise NotImplementedError

    def __truediv__(self, other) -> "Tensor":
        """提示：a/b = a * b^(-1)，或直接 np.divide，梯度分别是 1/b 和 -a/b^2。"""
        raise NotImplementedError

    def __rtruediv__(self, other) -> "Tensor":
        raise NotImplementedError

    def __neg__(self) -> "Tensor":
        raise NotImplementedError

    def __pow__(self, exponent) -> "Tensor":
        """exponent 只需支持标量（int/float）。

        提示：d/dx x^n = n * x^(n-1)
        """
        raise NotImplementedError

    def __matmul__(self, other) -> "Tensor":
        """矩阵乘法。

        这是本关最重要的一次手推：C = A @ B，shape (m,k) @ (k,n) → (m,n)，
        请自己推导 dL/dA 和 dL/dB（都是 upstream_grad 与另一个操作数的
        某种乘积）。两个自查工具：
          1) shape 约束：dL/dA 的 shape 必须等于 A.shape，凑维度只有一种乘法
          2) 标量验证：m=k=n=1 时应退化为普通乘法的求导
        推导过程写进 POSTMORTEM。2D 通过测试即可（batch matmul 不要求）。
        """
        raise NotImplementedError

    def __getitem__(self, idx) -> "Tensor":
        """切片/索引。

        提示：
          前向：result.data = self.data[idx]
          反向：建一个全 0 的 grad_self，grad_self[idx] += upstream_grad
               （用 np.add.at 避免重复索引的覆盖问题）
        """
        raise NotImplementedError

    # ------ 数学函数 ------

    def exp(self) -> "Tensor":
        """提示：d/dx exp(x) = exp(x)，即梯度 = result.data * upstream_grad。"""
        raise NotImplementedError

    def log(self) -> "Tensor":
        """提示：d/dx log(x) = 1/x。"""
        raise NotImplementedError

    def sigmoid(self) -> "Tensor":
        """σ(x) = 1/(1+exp(-x))。

        提示：dσ/dx = σ(x) * (1 - σ(x))，即 result.data * (1 - result.data)
        """
        raise NotImplementedError

    def maximum(self, scalar) -> "Tensor":
        """element-wise max(self, scalar)，scalar 为 Python 数值（如 0）。

        提示：梯度是 mask，self.data > scalar 的位置为 1，否则为 0
             （边界 self.data == scalar 时取 0 或 0.5 均可，测试不考边界）
        """
        raise NotImplementedError

    def sum(self, dim=None, keepdim: bool = False) -> "Tensor":
        """对指定维度（或全部）求和。

        提示：
          前向：np.sum(self.data, axis=dim, keepdims=keepdim)
          反向：upstream_grad 需要 broadcast 回 self.shape
               如果 keepdim=False 且 dim 不为 None，先 expand_dims 再 broadcast
        """
        raise NotImplementedError

    def mean(self, dim=None, keepdim: bool = False) -> "Tensor":
        """对指定维度（或全部）求均值。

        提示：mean = sum / n，梯度 = upstream_grad / n（n 是归约的元素数）
             也可以直接用 np.mean 前向 + 手动算 n
        """
        raise NotImplementedError

    def reshape(self, *shape) -> "Tensor":
        """改变 shape，不复制数据（numpy.reshape 尽量返回 view）。

        提示：反向时 upstream_grad.reshape(self.shape) 即可
        """
        raise NotImplementedError

    def transpose(self, dim0: int, dim1: int) -> "Tensor":
        """交换两个维度（类比 torch.transpose）。

        提示：前向用 np.swapaxes。反向自己想：把 grad 里每个元素
        送回它在原 tensor 中的位置，需要做什么变换？
        （思考题：为什么 transpose 的反向恰好是它自己？）
        """
        raise NotImplementedError

    # ------ 比较/显示 ------

    def __repr__(self) -> str:
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"
