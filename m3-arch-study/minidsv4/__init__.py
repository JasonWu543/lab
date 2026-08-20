"""Phase 3.1 的三个最小机制。"""

from .hyper_conn import HyperConnection, collapse_stream, expand_stream
from .muon import Muon, newton_schulz_msign
from .sparse_attn import SparseAttnConfig, build_block_mask, sparse_attention, sparse_attn_flops

__all__ = [
    "HyperConnection", "Muon", "SparseAttnConfig", "build_block_mask",
    "collapse_stream", "expand_stream", "newton_schulz_msign",
    "sparse_attention", "sparse_attn_flops",
]
