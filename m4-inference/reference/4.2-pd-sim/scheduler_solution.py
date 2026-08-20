"""Phase 4.2 scheduler 参考答案。"""

from .request import PDRequest
from .transfer import TransferModel, transfer_ticks
from .workers import DecodeWorker, PrefillWorker


class PDScheduler:
    def __init__(self, prefill: PrefillWorker, decode: DecodeWorker, transfer: TransferModel):
        self.prefill = prefill
        self.decode = decode
        self.transfer = transfer
        self.pending: list[PDRequest] = []
        self.in_flight: list[tuple[PDRequest, int]] = []
        self.completed: list[PDRequest] = []
        self.submitted = 0

    def submit(self, req: PDRequest) -> None:
        self.pending.append(req)
        self.submitted += 1

    def tick(self, t: int) -> None:
        # 倒计时表示还要经历多少个 transfer phase；本 phase 减至零即到达。
        still_flying: list[tuple[PDRequest, int]] = []
        for req, remaining in self.in_flight:
            remaining -= 1
            if remaining <= 0:
                self.decode.add(req)
            else:
                still_flying.append((req, remaining))
        self.in_flight = still_flying

        self.completed.extend(self.decode.step(t))
        prefilled = self.prefill.step(t)

        # prefill 在此 tick 结束，迁移要从下一 tick 的 transfer phase 才开始推进。
        for req in prefilled:
            cost = transfer_ticks(self.transfer, req.prompt_len)
            if cost == 0:
                self.decode.add(req)
            else:
                self.in_flight.append((req, cost))
        ready = [req for req in self.pending if req.arrival_time <= t]
        self.pending = [req for req in self.pending if req.arrival_time > t]
        for req in sorted(ready, key=lambda r: (r.arrival_time, r.req_id)):
            self.prefill.add(req)

    def all_done(self) -> bool:
        return (
            self.submitted == len(self.completed)
            and not self.pending
            and not self.in_flight
            and not self.prefill.queue
            and not self.decode.active
            and not self.decode.waiting
        )
