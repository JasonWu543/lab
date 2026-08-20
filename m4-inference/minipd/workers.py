"""Prefill 与 Decode worker。"""

from collections import deque

from .request import PDRequest


class PrefillWorker:
    """按 FIFO 推进、支持 chunked prefill 的 worker。"""

    def __init__(self, prefill_tokens_per_tick: int, chunk_size: int):
        """保存每 tick 总预算和单请求 chunk 上限，并准备 FIFO 队列。"""
        raise NotImplementedError("请初始化 prefill worker 的容量参数与等待队列")

    def add(self, req: PDRequest) -> None:
        """把到达的请求加入 FIFO 队列。"""
        raise NotImplementedError("请将请求加入 prefill 队列")

    def step(self, t: int) -> list[PDRequest]:
        """推进一 tick，返回本 tick 恰好完成 prefill 的请求。

        思考总预算与单请求 chunk 上限如何同时约束分配；一个请求本 tick
        用完 chunk 后，剩余预算应交给队列中的谁？
        """
        raise NotImplementedError("请实现 FIFO chunked prefill 推进")


class DecodeWorker:
    """具有固定并发 slot 和 FIFO 等待队列的 decode worker。"""

    def __init__(self, decode_slots: int):
        """保存 slot 数，并准备在座集合与 FIFO 等待队列。"""
        raise NotImplementedError("请初始化 decode worker")

    def add(self, req: PDRequest) -> None:
        """有空 slot 时入座，否则进入内部 FIFO 等待队列。"""
        raise NotImplementedError("请实现 decode admission")

    def step(self, t: int) -> list[PDRequest]:
        """让 tick 开始时已在座的请求各生成一个 token。

        首 token 和完成时间都取当前 ``t``。完成者释放 slot 后按 FIFO
        补位，但请注意补位者不能在同一 tick 再生成 token。
        """
        raise NotImplementedError("请实现 decode step、完成回收与 FIFO 补位")
