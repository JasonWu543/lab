"""Phase 4.2：Prefill/Decode 分离的离散时间模拟器。"""

from .request import PDRequest
from .simulate import SimResult, make_workload, run_colocated_baseline, run_sim
from .transfer import TransferModel, transfer_ticks
from .workers import DecodeWorker, PrefillWorker

__all__ = [
    "PDRequest",
    "TransferModel",
    "transfer_ticks",
    "PrefillWorker",
    "DecodeWorker",
    "PDScheduler",
    "SimResult",
    "run_sim",
    "run_colocated_baseline",
    "make_workload",
]


def __getattr__(name: str):
    """延迟导入调度器，避免包初始化时制造循环依赖。"""
    if name == "PDScheduler":
        from .scheduler import PDScheduler

        return PDScheduler
    raise AttributeError(name)
