"""SimpleGradScaler 参考答案。"""

import math

import torch


class SimpleGradScaler:
    def __init__(self, init_scale=2.**16, growth_factor=2.0,
                 backoff_factor=0.5, growth_interval=2000):
        if not math.isfinite(init_scale) or init_scale <= 0:
            raise ValueError("init_scale must be finite and positive")
        if growth_factor <= 1 or not 0 < backoff_factor < 1:
            raise ValueError("growth_factor/backoff_factor are invalid")
        if not isinstance(growth_interval, int) or growth_interval <= 0:
            raise ValueError("growth_interval must be a positive integer")
        self.current_scale = float(init_scale)
        self.growth_factor = float(growth_factor)
        self.backoff_factor = float(backoff_factor)
        self.growth_interval = growth_interval
        self._growth_tracker = 0
        # 每个 optimizer 有独立的 found_inf 与阶段；动态 scale/growth tracker
        # 则由本轮所有 optimizer 的结果共同更新。
        self._optimizer_states = {}

    def scale(self, loss):
        return loss * self.current_scale

    def unscale_(self, optimizer):
        key = id(optimizer)
        if key in self._optimizer_states:
            raise RuntimeError("unscale_ called twice for one optimizer in one step")
        inverse = 1.0 / self.current_scale
        found_inf = False
        with torch.no_grad():
            for group in optimizer.param_groups:
                for parameter in group["params"]:
                    if parameter.grad is None:
                        continue
                    parameter.grad.mul_(inverse)
                    if not torch.isfinite(parameter.grad).all():
                        found_inf = True
        self._optimizer_states[key] = {"stage": "unscaled", "found_inf": found_inf}

    def step(self, optimizer):
        state = self._optimizer_states.get(id(optimizer))
        if state is None:
            raise RuntimeError("unscale_ must precede step")
        if state["stage"] != "unscaled":
            raise RuntimeError("step called twice for one optimizer in one iteration")
        result = None if state["found_inf"] else optimizer.step()
        state["stage"] = "stepped"
        return result

    def update(self):
        if not self._optimizer_states:
            raise RuntimeError("unscale_ must precede update")
        # 与 torch.amp 一致：调用方可以只做 inf 检查后直接 update；尚未 step 的
        # optimizer 不更新参数，但其 found_inf 仍参与全局 scale 决策。
        found_inf = any(state["found_inf"] for state in self._optimizer_states.values())
        if found_inf:
            self.current_scale *= self.backoff_factor
            self._growth_tracker = 0
        else:
            self._growth_tracker += 1
            if self._growth_tracker == self.growth_interval:
                self.current_scale *= self.growth_factor
                self._growth_tracker = 0
        self._optimizer_states.clear()
