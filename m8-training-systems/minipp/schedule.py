"""Student exercises for the GPipe flush schedule."""


def gpipe_schedule(num_stages: int, num_microbatches: int) -> list[list[tuple]]:
    """逐 tick 构造 flush GPipe；怎样排列两个方向的 wave？"""
    raise NotImplementedError("请实现 GPipe 调度并处理非法正整数输入")


def bubble_fraction(num_stages: int, num_microbatches: int) -> float:
    """由总 stage slots 与实际操作数推导空闲比例。"""
    raise NotImplementedError("请推导 bubble fraction 并处理非法输入")
