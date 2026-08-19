from dataclasses import dataclass


@dataclass
class MoEConfig:
    hidden_size: int = 256
    intermediate_size: int = 1024      # dense FFN (control group)
    n_routed_experts: int = 16
    n_shared_experts: int = 1
    top_k: int = 2
    moe_intermediate_size: int = 128   # fine-grained: each expert is small
    bias_update_speed: float = 0.001   # gamma
    q_lora_rank: int = 96
    kv_lora_rank: int = 64
    qk_nope_head_dim: int = 32
    qk_rope_head_dim: int = 16
    v_head_dim: int = 32
    num_heads: int = 8
    vocab_size: int = 8192
    num_layers: int = 4
    max_seq_len: int = 512
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
