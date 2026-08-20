"""
Phase 1.2：Qwen2.5 → MiniLM 权重转换

需要下载 Qwen2.5-0.5B 权重后才能运行 T7 测试：
    python3 scripts/download_qwen.py
"""
import json
from pathlib import Path
from typing import Union
import torch

from .config import ModelConfig
from .model import MiniLM


def convert_qwen_config(qwen_config: dict) -> ModelConfig:
    """
    将 Qwen2.5 的 config.json（dict）转换为 ModelConfig。

    字段对应关系：
      vocab_size             → vocab_size
      hidden_size            → hidden_size
      intermediate_size      → intermediate_size
      num_hidden_layers      → num_layers
      num_attention_heads    → num_heads
      num_key_value_heads    → num_kv_heads
      head_dim（可选）       → head_dim（缺省时用 hidden_size // num_heads）
      max_position_embeddings→ max_seq_len
      rope_theta             → rope_theta
      rms_norm_eps           → rms_norm_eps
      attention_bias         → attention_bias
      tie_word_embeddings    → tie_word_embeddings
    """
    raise NotImplementedError


def load_qwen(
    weight_dir: Union[str, Path],
    device="cpu",
    dtype=torch.float32,
) -> MiniLM:
    """
    从 Qwen2.5 权重目录加载并返回 MiniLM 模型。

    权重映射表（Qwen2.5 key → MiniLM key）：
      model.embed_tokens.weight                        → embedding.weight
      model.norm.weight                                → norm.weight
      lm_head.weight                                   → lm_head.weight（tie 时跳过）
      model.layers.{i}.self_attn.q_proj.weight/bias   → layers.{i}.attn.q_proj.weight/bias
      model.layers.{i}.self_attn.k_proj.weight/bias   → layers.{i}.attn.k_proj.weight/bias
      model.layers.{i}.self_attn.v_proj.weight/bias   → layers.{i}.attn.v_proj.weight/bias
      model.layers.{i}.self_attn.o_proj.weight        → layers.{i}.attn.o_proj.weight
      model.layers.{i}.mlp.gate_proj.weight           → layers.{i}.mlp.gate_proj.weight
      model.layers.{i}.mlp.up_proj.weight             → layers.{i}.mlp.up_proj.weight
      model.layers.{i}.mlp.down_proj.weight           → layers.{i}.mlp.down_proj.weight
      model.layers.{i}.input_layernorm.weight         → layers.{i}.norm1.weight
      model.layers.{i}.post_attention_layernorm.weight→ layers.{i}.norm2.weight

    提示：
      - 先读 config.json → convert_qwen_config → 构建 MiniLM
      - 优先从 *.safetensors 加载（用 safetensors.torch.load_file），回退到 *.bin
      - load_state_dict(strict=False)，检查 missing/unexpected keys
    """
    raise NotImplementedError
