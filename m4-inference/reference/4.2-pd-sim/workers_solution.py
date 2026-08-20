"""Phase 4.2 workers 参考答案。"""

from collections import deque

from .request import PDRequest


class PrefillWorker:
    """FIFO 是跨 tick 的轮转 FIFO：每 tick 每请求至多取一个 chunk。"""

    def __init__(self, prefill_tokens_per_tick: int, chunk_size: int):
        if prefill_tokens_per_tick <= 0 or chunk_size <= 0:
            raise ValueError("prefill capacity and chunk_size must be positive")
        self.prefill_tokens_per_tick = prefill_tokens_per_tick
        self.chunk_size = chunk_size
        self.queue: deque[PDRequest] = deque()

    def add(self, req: PDRequest) -> None:
        self.queue.append(req)

    def step(self, t: int) -> list[PDRequest]:
        del t
        budget = self.prefill_tokens_per_tick
        completed: list[PDRequest] = []
        # 固定本 tick 可服务的队列长度，防止轮转回来后二次超过 chunk 上限。
        turns = len(self.queue)
        for _ in range(turns):
            if budget == 0:
                break
            req = self.queue.popleft()
            remaining = req.prompt_len - req.prefilled_tokens
            used = min(budget, self.chunk_size, remaining)
            req.prefilled_tokens += used
            budget -= used
            if req.prefilled_tokens == req.prompt_len:
                completed.append(req)
            else:
                self.queue.append(req)
        return completed


class DecodeWorker:
    def __init__(self, decode_slots: int):
        if decode_slots <= 0:
            raise ValueError("decode_slots must be positive")
        self.decode_slots = decode_slots
        self.active: list[PDRequest] = []
        self.waiting: deque[PDRequest] = deque()

    def add(self, req: PDRequest) -> None:
        if len(self.active) < self.decode_slots:
            self.active.append(req)
        else:
            self.waiting.append(req)

    def step(self, t: int) -> list[PDRequest]:
        completed: list[PDRequest] = []
        # 快照保证本 tick 结束后补位者要等到下一 tick。
        for req in list(self.active):
            req.generated_tokens += 1
            if req.first_token_time is None:
                req.first_token_time = t
            if req.generated_tokens == req.max_new_tokens:
                req.finish_time = t
                self.active.remove(req)
                completed.append(req)
        while len(self.active) < self.decode_slots and self.waiting:
            self.active.append(self.waiting.popleft())
        return completed
