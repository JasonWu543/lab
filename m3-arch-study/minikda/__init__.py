"""Phase 3.2: fixed-state delta attention."""

from .config import KDAConfig
from .delta import delta_rule_chunked, delta_rule_recurrent
from .layer import DeltaAttention, state_bytes

__all__ = [
    "KDAConfig",
    "DeltaAttention",
    "delta_rule_chunked",
    "delta_rule_recurrent",
    "state_bytes",
]
