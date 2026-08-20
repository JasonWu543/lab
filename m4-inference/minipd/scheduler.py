"""PD 分离调度器。"""

from .request import PDRequest
from .transfer import TransferModel
from .workers import DecodeWorker, PrefillWorker


class PDScheduler:
    """协调 admission、prefill、KV transfer 与 decode。"""

    def __init__(
        self,
        prefill: PrefillWorker,
        decode: DecodeWorker,
        transfer: TransferModel,
    ):
        """保存三个组件，并准备待 admission、在途及完成状态。"""
        raise NotImplementedError("请初始化 PD 调度器状态")

    def submit(self, req: PDRequest) -> None:
        """登记一条请求，等待 tick 末的 admission 决策。"""
        raise NotImplementedError("请登记待 admission 请求")

    def tick(self, t: int) -> None:
        """严格按 transfer → decode → prefill → 调度决策推进一 tick。

        调度决策需要处理 prefill 完成后的迁移，以及 arrival_time 已到的
        请求；想一想迁移倒计时在哪个阶段递减，怎样避免差一 tick。
        """
        raise NotImplementedError("请实现冻结顺序下的一次调度 tick")

    def all_done(self) -> bool:
        """所有已提交请求均完成且各内部队列均为空时返回 True。"""
        raise NotImplementedError("请判断模拟是否已完成")
