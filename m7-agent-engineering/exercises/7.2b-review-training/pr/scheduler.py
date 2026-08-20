"""Learning-rate schedules for compact training jobs."""

import math


class CosineSchedule:
    """Linear warmup followed by cosine decay to a configurable floor."""

    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=0.0):
        if total_steps <= 0 or warmup_steps < 0 or warmup_steps >= total_steps:
            raise ValueError("invalid schedule length")
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.step_count = 0

    def _factor(self, step):
        if self.warmup_steps and step <= self.warmup_steps:
            return step / self.warmup_steps
        progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps + 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    def step(self):
        self.step_count += 1
        factor = self._factor(self.step_count)
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = self.min_lr + base_lr * factor

    def state_dict(self):
        return {"step_count": self.step_count, "base_lrs": self.base_lrs}

    def load_state_dict(self, state):
        self.step_count = state["step_count"]
        self.base_lrs = state["base_lrs"]
