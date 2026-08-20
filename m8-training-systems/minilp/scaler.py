"""教学用动态 loss scaler。"""


class SimpleGradScaler:
    """实现 scale→unscale→step→update 状态机。"""

    def __init__(self, init_scale=2.**16, growth_factor=2.0,
                 backoff_factor=0.5, growth_interval=2000):
        raise NotImplementedError("请初始化缩放参数、增长计数与单步状态")

    def scale(self, loss):
        """返回放大后的 loss。"""
        raise NotImplementedError("请使用当前动态 scale")

    def unscale_(self, optimizer):
        """原位还原梯度并记录非有限值；同一步不得重复调用。"""
        raise NotImplementedError("请遍历参数梯度并维护 optimizer 级状态")

    def step(self, optimizer):
        """仅在梯度有限时执行 optimizer.step。"""
        raise NotImplementedError("请根据本步检测结果决定是否跳步")

    def update(self):
        """根据本步成败回退或增长 scale，并开启下一步。"""
        raise NotImplementedError("请实现连续成功计数和动态缩放状态迁移")
