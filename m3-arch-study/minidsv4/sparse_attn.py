"""Block Top-K sparse attention：学生练习。"""

from dataclasses import dataclass
import torch


@dataclass
class SparseAttnConfig:
    block_size: int
    top_k: int
    local_blocks: int
    sink_blocks: int = 1


def build_block_mask(scores_qk_block: torch.Tensor, cfg: SparseAttnConfig,
                     q_pos: torch.Tensor) -> torch.Tensor:
    """返回因果的块选择 mask。

    想一想：哪些块由位置规则无条件保留？在按分数选择前，哪些候选必须排除？
    当前块的 pooled 分数是用哪些 K 算出来的——让它参与 top-k 竞争，
    因果性还成立吗？（T2 会用未来 K 扰动专门查这条路径。）
    """
    raise NotImplementedError("请实现因果的 top-k、local 与 sink 块并集")


def sparse_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                     cfg: SparseAttnConfig) -> torch.Tensor:
    """计算 ``(B,H,T,D)`` 因果 sparse attention。

    如何把块级选择映射回 token，并保证当前块内仍然因果？
    """
    raise NotImplementedError("请实现 block sparse causal attention")


def sparse_attn_flops(T: int, D: int, cfg: SparseAttnConfig) -> int:
    """返回 decode 单 token 的 QK 与 AV FLOPs 上界。

    最后一个不完整块与重复选中的块会怎样影响 token 数上界？
    """
    raise NotImplementedError("请推导 decode 单 token FLOPs 上界")
