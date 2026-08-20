"""Muon 参考答案。"""

import math
import torch


def newton_schulz_msign(G, steps=5):
    """归一化后用冻结的高阶 Newton--Schulz 多项式逼近 UV^T。"""
    if G.ndim != 2:
        raise ValueError("msign expects a 2D tensor")
    X = G.float()
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.T
    X = X / (X.norm() + 1e-7)
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        A = X @ X.T
        X = a * X + b * (A @ X) + c * ((A @ A) @ X)
    if transposed:
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True):
        if lr < 0 or not 0 <= momentum < 1:
            raise ValueError("invalid lr or momentum")
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov))
        for group in self.param_groups:
            for p in group["params"]:
                if p.ndim != 2:
                    raise ValueError("Muon only accepts 2D parameters")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.detach().float()
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(grad)
                buf = state["momentum_buffer"]
                buf.mul_(group["momentum"]).add_(grad)
                candidate = grad.add(buf, alpha=group["momentum"]) \
                    if group["nesterov"] else buf
                update = newton_schulz_msign(candidate)
                rows, cols = p.shape
                update.mul_(math.sqrt(max(1.0, rows / cols)))
                p.add_(update.to(p.dtype), alpha=-group["lr"])
        return loss
