"""学生任务：实现 delta-rule 的递推和 chunkwise 两种计算路径。

冻结公式见本 phase 的 SPEC。这里刻意不提供核心实现或伪代码。
"""

import torch


def delta_rule_recurrent(q, k, v, beta, alpha=None) -> torch.Tensor:
    """逐 token 递推，内部归一化 k，并以 fp32 累加。

    思考：每次写入应先改变状态的哪一部分？输出读取的是更新前还是更新后的
    状态？输入 dtype 与累加 dtype 不同时，返回值应遵守什么约定？
    """
    raise NotImplementedError("请依据冻结递推公式实现逐 token delta rule")


def delta_rule_chunked(q, k, v, beta, alpha=None, chunk_size: int = 16) -> torch.Tensor:
    """块内矩阵化、块间仅传固定大小状态的 delta rule。

    思考：把递推式对一个 chunk 展开，输出和末状态分别由哪些量决定？
    块内先后写入之间的依赖有什么规律？能不能一次性解出、而不是逐 token 算？
    非整块尾部和遗忘门应如何进入同一个推导？
    """
    raise NotImplementedError("请推导并实现 chunkwise delta rule")
