"""
Phase 1.2 参考答案：Qwen2.5 权重转换
学生文件：minilm/model/convert_qwen.py
"""
import os
from pathlib import Path
from typing import Union
import torch

from .config_solution import ModelConfig
from .model_solution import MiniLM


def convert_qwen_config(qwen_config: dict) -> ModelConfig:
    """将 Qwen2.5 config.json 转为 ModelConfig"""
    return ModelConfig(
        vocab_size=qwen_config['vocab_size'],
        hidden_size=qwen_config['hidden_size'],
        intermediate_size=qwen_config['intermediate_size'],
        num_layers=qwen_config['num_hidden_layers'],
        num_heads=qwen_config['num_attention_heads'],
        num_kv_heads=qwen_config['num_key_value_heads'],
        head_dim=qwen_config.get('head_dim', qwen_config['hidden_size'] // qwen_config['num_attention_heads']),
        max_seq_len=qwen_config.get('max_position_embeddings', 32768),
        rope_theta=qwen_config.get('rope_theta', 1000000.0),
        rms_norm_eps=qwen_config.get('rms_norm_eps', 1e-6),
        attention_bias=qwen_config.get('attention_bias', True),
        tie_word_embeddings=qwen_config.get('tie_word_embeddings', True),
    )


def load_qwen(weight_dir: Union[str, Path]) -> MiniLM:
    """
    从 Qwen2.5 权重目录加载 MiniLM 模型。

    Qwen2.5 → MiniLM 权重映射：
    model.embed_tokens.weight → embedding.weight
    model.norm.weight → norm.weight
    lm_head.weight → lm_head.weight (tied 时跳过)
    model.layers.{i}.self_attn.q_proj.{weight,bias} → layers.{i}.attn.q_proj.{weight,bias}
    model.layers.{i}.self_attn.k_proj.{weight,bias} → layers.{i}.attn.k_proj.{weight,bias}
    model.layers.{i}.self_attn.v_proj.{weight,bias} → layers.{i}.attn.v_proj.{weight,bias}
    model.layers.{i}.self_attn.o_proj.weight → layers.{i}.attn.o_proj.weight
    model.layers.{i}.mlp.gate_proj.weight → layers.{i}.mlp.gate_proj.weight
    model.layers.{i}.mlp.up_proj.weight → layers.{i}.mlp.up_proj.weight
    model.layers.{i}.mlp.down_proj.weight → layers.{i}.mlp.down_proj.weight
    model.layers.{i}.input_layernorm.weight → layers.{i}.norm1.weight
    model.layers.{i}.post_attention_layernorm.weight → layers.{i}.norm2.weight
    """
    import json
    weight_dir = Path(weight_dir)

    with open(weight_dir / 'config.json') as f:
        qwen_cfg = json.load(f)

    cfg = convert_qwen_config(qwen_cfg)
    model = MiniLM(cfg)

    # Load weights from safetensors or pytorch bin
    qwen_state = {}
    safetensor_files = sorted(weight_dir.glob('*.safetensors'))
    if safetensor_files:
        from safetensors.torch import load_file
        for sf in safetensor_files:
            qwen_state.update(load_file(str(sf)))
    else:
        bin_files = sorted(weight_dir.glob('*.bin'))
        for bf in bin_files:
            qwen_state.update(torch.load(str(bf), map_location='cpu'))

    # Build mapping
    new_state = {}
    new_state['embedding.weight'] = qwen_state['model.embed_tokens.weight']
    new_state['norm.weight'] = qwen_state['model.norm.weight']
    if not cfg.tie_word_embeddings and 'lm_head.weight' in qwen_state:
        new_state['lm_head.weight'] = qwen_state['lm_head.weight']

    for i in range(cfg.num_layers):
        prefix_q = f'model.layers.{i}.self_attn'
        prefix_m = f'layers.{i}.attn'
        for proj in ['q_proj', 'k_proj', 'v_proj']:
            new_state[f'{prefix_m}.{proj}.weight'] = qwen_state[f'{prefix_q}.{proj}.weight']
            bias_key = f'{prefix_q}.{proj}.bias'
            if bias_key in qwen_state:
                new_state[f'{prefix_m}.{proj}.bias'] = qwen_state[bias_key]
        new_state[f'{prefix_m}.o_proj.weight'] = qwen_state[f'{prefix_q}.o_proj.weight']

        prefix_mlp_q = f'model.layers.{i}.mlp'
        prefix_mlp_m = f'layers.{i}.mlp'
        for proj in ['gate_proj', 'up_proj', 'down_proj']:
            new_state[f'{prefix_mlp_m}.{proj}.weight'] = qwen_state[f'{prefix_mlp_q}.{proj}.weight']

        new_state[f'layers.{i}.norm1.weight'] = qwen_state[f'model.layers.{i}.input_layernorm.weight']
        new_state[f'layers.{i}.norm2.weight'] = qwen_state[f'model.layers.{i}.post_attention_layernorm.weight']

    missing, unexpected = model.load_state_dict(new_state, strict=False)
    # For tied embeddings, lm_head.weight is expected to be missing
    expected_missing = ['lm_head.weight'] if cfg.tie_word_embeddings else []
    assert set(missing) <= set(expected_missing), f"Unexpected missing keys: {set(missing) - set(expected_missing)}"
    assert len(unexpected) == 0, f"Unexpected keys: {unexpected}"

    return model
