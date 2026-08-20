"""Phase 6.1 — Compute-Optimal 闭式（学生实现文件）

你的任务：实现 compute_optimal 和 optimal_exponents，
让 tests/test_scaling.py 的 T4–T5 全绿。

这是本 phase 的核心手推（POSTMORTEM 必写）。

闯关顺序（U2）：
  Step 1  手推闭式解（纸笔推导，然后编码）
           方法提示：拉格朗日乘子法，或直接代入消元
           ——把约束 6ND=C 代入 D=C/(6N) 后，L 只含 N，
             对 N 求导令 dL/dN = 0，整理出 N* 的显式表达式
           → 过 T4（闭式 vs 数值网格对拍 < 1%）
  Step 2  从 N* 表达式提取指数 a（N* ∝ C^a）
           → 过 T5（a+b=1；α=β 时 a=b=0.5）

泄题红线（下列内容绝对不允许从参考答案抄入）：
  - compute_optimal 的具体闭式表达式
  - optimal_exponents 中 a = β/(α+β) 这个结论

运行测试：
  cd m6-data-scaling && python3 -m pytest tests/test_scaling.py::TestT4ClosedFormVsGrid -x -q
  cd m6-data-scaling && python3 -m pytest tests/test_scaling.py::TestT5Exponents -x -q

卡住 30 分钟以上再看参考答案：reference/6.1-scaling/optimal_solution.py
（看完要能说出你的推导和答案的关键差异）
"""

from __future__ import annotations

from .fit import ScalingParams


def compute_optimal(C: float, p: ScalingParams) -> tuple[float, float]:
    """min_{N,D} L s.t. 6ND = C 的闭式解 (N*, D*)。
    要求解析推导（不许数值搜索）——测试会拿数值网格搜索对拍你的闭式。
    """
    # TODO U2-Step1: 实现闭式解
    # 提示：先在纸上推导。约束 6ND=C → D=C/(6N)，代入 L(N,D) 后
    #       对 N 求导令 dL/dN = 0，整理 N* 的闭式，再算 D* = C/(6N*)
    raise NotImplementedError("TODO: compute_optimal（先推导，再编码）")


def optimal_exponents(p: ScalingParams) -> tuple[float, float]:
    """返回 (a, b)：N* ∝ C^a, D* ∝ C^b。验证 a + b = 1。"""
    # TODO U2-Step2: 从你推导的 N* 表达式中提取指数 a 和 b
    raise NotImplementedError("TODO: optimal_exponents")
