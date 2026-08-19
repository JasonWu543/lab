"""
Phase 1.2 参考答案：count_params, estimate_flops_per_token
学生文件：minilm/model/counting.py

count_params 是闭式手算：只用 config 里的数字推导参数量，不实例化模型。
这是 U2.3 的核心练习——每一块矩阵都数对，等式
count_params(cfg) == sum(p.numel() for p in MiniLM(cfg).parameters())
才成立（weight tying 时共享的 Parameter 在 parameters() 里只出现一次）。
"""
from .config_solution import ModelConfig


def count_params(cfg: ModelConfig) -> int:
    """闭式手算参数量（不实例化模型）。"""
    H = cfg.hidden_size
    V = cfg.vocab_size
    I = cfg.intermediate_size
    hd = cfg.head_dim if cfg.head_dim is not None else H // cfg.num_heads
    nq = cfg.num_heads
    nkv = cfg.num_kv_heads
    bias = 1 if cfg.attention_bias else 0

    embedding = V * H

    # attention：q/k/v 有可选 bias，o_proj 恒无 bias
    q = H * (nq * hd) + bias * (nq * hd)
    k = H * (nkv * hd) + bias * (nkv * hd)
    v = H * (nkv * hd) + bias * (nkv * hd)
    o = (nq * hd) * H
    # SwiGLU MLP：gate/up/down 全部无 bias
    mlp = 2 * H * I + I * H
    # 每层两个 RMSNorm，各 H 个缩放参数
    norms = 2 * H

    per_layer = q + k + v + o + mlp + norms
    final_norm = H
    lm_head = 0 if cfg.tie_word_embeddings else V * H

    return embedding + cfg.num_layers * per_layer + final_norm + lm_head


def estimate_flops_per_token(cfg: ModelConfig, seq_len: int) -> int:
    """
    估算每 token 的前向 FLOPs（只计矩阵乘，2mnk 记法；不含 softmax/norm）。
    """
    h = cfg.hidden_size
    i = cfg.intermediate_size
    nh = cfg.num_heads
    nkv = cfg.num_kv_heads
    hd = cfg.head_dim if cfg.head_dim is not None else h // nh
    L = cfg.num_layers
    V = cfg.vocab_size

    per_layer = (
        2 * h * (nh * hd) +            # q_proj
        2 * h * (nkv * hd) +           # k_proj
        2 * h * (nkv * hd) +           # v_proj
        2 * (nh * hd) * h +            # o_proj
        2 * seq_len * nh * hd +        # QK^T（每 token 对 seq_len 个位置）
        2 * seq_len * nh * hd +        # AV
        2 * h * i +                    # gate_proj
        2 * h * i +                    # up_proj
        2 * i * h                      # down_proj
    )

    total = L * per_layer + 2 * h * V  # lm_head
    return int(total)
