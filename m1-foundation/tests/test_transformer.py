"""
Phase 1.2: Qwen-like Transformer 测试套件
所有测试在 CPU fp32 下运行，使用固定随机种子。

运行方式:
    python3 -m pytest tests/test_transformer.py -x -q
    python3 -m pytest tests/test_transformer.py -x -q -m "not slow"  # 跳过 T8

注意：实现前所有测试应该报 NotImplementedError（导入成功但函数体未实现）。
"""
import os
import sys
import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from minilm.model.config import ModelConfig
from minilm.model.model import (
    RMSNorm, MLP, Attention, DecoderLayer, MiniLM,
    precompute_rope, apply_rope, KVCache,
)
from minilm.model.generate import sample_next, generate
from minilm.model.counting import count_params, estimate_flops_per_token


# ──────────────────────────────── fixtures ────────────────────────────────

@pytest.fixture
def small_cfg():
    """小配置，用于快速单元测试"""
    return ModelConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        head_dim=16,
        max_seq_len=128,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
        attention_bias=True,
        tie_word_embeddings=True,
    )

@pytest.fixture
def mha_cfg():
    """MHA 配置（num_kv_heads == num_heads）"""
    return ModelConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_layers=2,
        num_heads=4,
        num_kv_heads=4,
        head_dim=16,
        max_seq_len=128,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
        attention_bias=True,
        tie_word_embeddings=True,
    )


# ──────────────────────────────────── T1 ───────────────────────────────────

class TestT1BasicModules:
    """T1: RMSNorm / MLP / Embedding 数值对齐"""

    def test_rmsnorm_shape(self, small_cfg):
        torch.manual_seed(42)
        x = torch.randn(2, 8, small_cfg.hidden_size)
        norm = RMSNorm(small_cfg.hidden_size, small_cfg.rms_norm_eps)
        out = norm(x)
        assert out.shape == x.shape

    def test_rmsnorm_numeric(self, small_cfg):
        """对比 PyTorch 手动实现"""
        torch.manual_seed(0)
        x = torch.randn(2, 8, small_cfg.hidden_size)
        norm = RMSNorm(small_cfg.hidden_size, small_cfg.rms_norm_eps)
        out = norm(x)

        # Reference: manual
        w = norm.weight.detach()
        eps = small_cfg.rms_norm_eps
        x_f = x.float()
        rms = (x_f.pow(2).mean(-1, keepdim=True) + eps).rsqrt()
        ref = (w * x_f * rms).to(x.dtype)
        assert torch.allclose(out, ref, rtol=1e-5, atol=1e-6)

    def test_rmsnorm_dtype_preserved(self, small_cfg):
        """输入 float32，输出 float32（内部可升精度）"""
        x = torch.randn(2, 8, small_cfg.hidden_size)
        norm = RMSNorm(small_cfg.hidden_size)
        out = norm(x)
        assert out.dtype == x.dtype

    def test_mlp_shape(self, small_cfg):
        torch.manual_seed(1)
        x = torch.randn(2, 8, small_cfg.hidden_size)
        mlp = MLP(small_cfg)
        out = mlp(x)
        assert out.shape == x.shape

    def test_mlp_swiglu(self, small_cfg):
        """验证 MLP = down(silu(gate(x)) * up(x))"""
        torch.manual_seed(2)
        x = torch.randn(1, 4, small_cfg.hidden_size)
        mlp = MLP(small_cfg)
        out = mlp(x)
        ref = mlp.down_proj(F.silu(mlp.gate_proj(x)) * mlp.up_proj(x))
        assert torch.allclose(out, ref, rtol=1e-5)

    def test_embedding_shape(self, small_cfg):
        model = MiniLM(small_cfg)
        ids = torch.randint(0, small_cfg.vocab_size, (2, 8))
        emb = model.embedding(ids)
        assert emb.shape == (2, 8, small_cfg.hidden_size)


# ──────────────────────────────────── T2 ───────────────────────────────────

class TestT2Attention:
    """T2: Attention 输出 vs F.scaled_dot_product_attention"""

    def test_attention_shape(self, small_cfg):
        torch.manual_seed(3)
        attn = Attention(small_cfg)
        cos, sin = precompute_rope(small_cfg.head_dim, small_cfg.max_seq_len, small_cfg.rope_theta)
        x = torch.randn(2, 8, small_cfg.hidden_size)
        positions = torch.arange(8)
        out = attn(x, cos, sin, positions)
        assert out.shape == (2, 8, small_cfg.hidden_size)

    def test_attention_vs_sdpa(self, mha_cfg):
        """MHA（GQA n_rep=1）时，与 F.scaled_dot_product_attention 对齐"""
        torch.manual_seed(4)
        cfg = mha_cfg
        attn = Attention(cfg)
        cos, sin = precompute_rope(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)

        B, T = 1, 6
        x = torch.randn(B, T, cfg.hidden_size)
        positions = torch.arange(T)
        out = attn(x, cos, sin, positions)

        # Reference: manually compute
        q = attn.q_proj(x).view(B, T, cfg.num_heads, cfg.head_dim).transpose(1, 2)
        k = attn.k_proj(x).view(B, T, cfg.num_kv_heads, cfg.head_dim).transpose(1, 2)
        v = attn.v_proj(x).view(B, T, cfg.num_kv_heads, cfg.head_dim).transpose(1, 2)
        q_r, k_r = apply_rope(q, k, cos, sin, positions)
        ref_out = F.scaled_dot_product_attention(q_r, k_r, v, is_causal=True)
        ref_out = ref_out.transpose(1, 2).contiguous().view(B, T, -1)
        ref_out = attn.o_proj(ref_out)

        assert torch.allclose(out, ref_out, atol=1e-5), f"max diff: {(out - ref_out).abs().max()}"

    def test_gqa_output_shape(self, small_cfg):
        """GQA（num_kv_heads < num_heads）输出形状正确"""
        torch.manual_seed(5)
        attn = Attention(small_cfg)
        cos, sin = precompute_rope(small_cfg.head_dim, small_cfg.max_seq_len)
        x = torch.randn(2, 8, small_cfg.hidden_size)
        positions = torch.arange(8)
        out = attn(x, cos, sin, positions)
        assert out.shape == (2, 8, small_cfg.hidden_size)

    def test_gqa_vs_sdpa_numeric(self, small_cfg):
        """GQA 数值对齐：kv head 重复展开后必须与 SDPA 完全一致，
        且每个 kv head 要对应「连续的一组」query head（不是交错分配）"""
        torch.manual_seed(15)
        cfg = small_cfg
        n_rep = cfg.num_heads // cfg.num_kv_heads
        attn = Attention(cfg)
        cos, sin = precompute_rope(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)

        B, T = 2, 7
        x = torch.randn(B, T, cfg.hidden_size)
        positions = torch.arange(T)
        out = attn(x, cos, sin, positions)

        q = attn.q_proj(x).view(B, T, cfg.num_heads, cfg.head_dim).transpose(1, 2)
        k = attn.k_proj(x).view(B, T, cfg.num_kv_heads, cfg.head_dim).transpose(1, 2)
        v = attn.v_proj(x).view(B, T, cfg.num_kv_heads, cfg.head_dim).transpose(1, 2)
        q_r, k_r = apply_rope(q, k, cos, sin, positions)
        # HF 语义的 repeat_kv：沿 head 维 interleave 展开
        k_full = k_r.repeat_interleave(n_rep, dim=1)
        v_full = v.repeat_interleave(n_rep, dim=1)
        ref_out = F.scaled_dot_product_attention(q_r, k_full, v_full, is_causal=True)
        ref_out = ref_out.transpose(1, 2).contiguous().view(B, T, -1)
        ref_out = attn.o_proj(ref_out)

        assert torch.allclose(out, ref_out, atol=1e-5), \
            f"GQA numeric mismatch, max diff: {(out - ref_out).abs().max()}"


# ──────────────────────────────────── T3 ───────────────────────────────────

class TestT3RoPE:
    """T3: RoPE 数值对齐 + 相对位置不变性"""

    def test_rope_shape(self, small_cfg):
        cos, sin = precompute_rope(small_cfg.head_dim, small_cfg.max_seq_len, small_cfg.rope_theta)
        assert cos.shape == (small_cfg.max_seq_len, small_cfg.head_dim)
        assert sin.shape == (small_cfg.max_seq_len, small_cfg.head_dim)

    def test_rope_vs_hf(self, small_cfg):
        """与 HF rotate_half 实现对齐"""
        head_dim = small_cfg.head_dim
        T = 8
        theta = small_cfg.rope_theta

        cos, sin = precompute_rope(head_dim, small_cfg.max_seq_len, theta)

        # HF reference
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(T).float()
        freqs = torch.outer(t, inv_freq)
        cos_ref = torch.cat([freqs.cos(), freqs.cos()], dim=-1)
        sin_ref = torch.cat([freqs.sin(), freqs.sin()], dim=-1)

        assert torch.allclose(cos[:T], cos_ref, rtol=1e-5)
        assert torch.allclose(sin[:T], sin_ref, rtol=1e-5)

    def test_relative_position_invariance(self, small_cfg):
        """RoPE 相对位置不变性：shift positions by offset，attention scores 不变"""
        torch.manual_seed(6)
        head_dim = small_cfg.head_dim
        T = 4
        B, nh = 1, 2
        offset = 10

        cos, sin = precompute_rope(head_dim, small_cfg.max_seq_len + offset, small_cfg.rope_theta)

        q = torch.randn(B, nh, T, head_dim)
        k = torch.randn(B, nh, T, head_dim)
        positions = torch.arange(T)
        positions_shifted = positions + offset

        q1, k1 = apply_rope(q, k, cos, sin, positions)
        q2, k2 = apply_rope(q, k, cos, sin, positions_shifted)

        # Attention scores at relative distances should match
        scores1 = torch.matmul(q1, k1.transpose(-2, -1))  # (B, nh, T, T)
        scores2 = torch.matmul(q2, k2.transpose(-2, -1))

        assert torch.allclose(scores1, scores2, atol=1e-5), \
            f"Relative position invariance failed, max diff: {(scores1-scores2).abs().max()}"


# ──────────────────────────────────── T4 ───────────────────────────────────

class TestT4Counting:
    """T4: count_params 准确性 + weight tying"""

    @staticmethod
    def _actual_params(model) -> int:
        # weight tying 时共享的 Parameter 在 parameters() 里只出现一次，
        # 再用 data_ptr 去重兜底
        seen = set()
        actual = 0
        for p in model.parameters():
            if p.data_ptr() not in seen:
                seen.add(p.data_ptr())
                actual += p.numel()
        return actual

    def test_count_params_closed_form_gqa(self, small_cfg):
        # 闭式手算：只给 config，不实例化模型
        torch.manual_seed(7)
        model = MiniLM(small_cfg)
        assert count_params(small_cfg) == self._actual_params(model)

    def test_count_params_closed_form_mha(self, mha_cfg):
        torch.manual_seed(8)
        model = MiniLM(mha_cfg)
        assert count_params(mha_cfg) == self._actual_params(model)

    def test_count_params_untied_custom_headdim(self):
        # untie + head_dim != hidden_size // num_heads + 无 attention bias：
        # 三个最容易数错的分支一起验
        cfg = ModelConfig(vocab_size=128, hidden_size=64, intermediate_size=192,
                          num_layers=3, num_heads=4, num_kv_heads=2, head_dim=24,
                          attention_bias=False, tie_word_embeddings=False,
                          max_seq_len=64)
        torch.manual_seed(9)
        model = MiniLM(cfg)
        assert count_params(cfg) == self._actual_params(model)

    def test_weight_tying(self, small_cfg):
        model = MiniLM(small_cfg)
        assert model.lm_head.weight.data_ptr() == model.embedding.weight.data_ptr(), \
            "lm_head.weight 和 embedding.weight 应共享存储（weight tying）"

    def test_no_tying_when_disabled(self):
        cfg = ModelConfig(tie_word_embeddings=False)
        model = MiniLM(cfg)
        assert model.lm_head.weight.data_ptr() != model.embedding.weight.data_ptr()

    def test_estimate_flops_type(self, small_cfg):
        flops = estimate_flops_per_token(small_cfg, seq_len=64)
        assert isinstance(flops, int)
        assert flops > 0

    def test_estimate_flops_scales_with_seqlen(self, small_cfg):
        f1 = estimate_flops_per_token(small_cfg, seq_len=64)
        f2 = estimate_flops_per_token(small_cfg, seq_len=128)
        assert f2 > f1, "FLOPs 应随 seq_len 增大"


# ──────────────────────────────────── T5 ───────────────────────────────────

class TestT5CachedDecode:
    """T5: cached vs non-cached 输出一致（最重要的测试）"""

    def _run_cached_vs_nocache(self, cfg, B=1, T_prefill=4, T_decode=3):
        torch.manual_seed(42)
        model = MiniLM(cfg)
        model.eval()

        input_ids = torch.randint(0, cfg.vocab_size, (B, T_prefill))

        with torch.no_grad():
            # Non-cached: full forward on all tokens at each step
            logits_nc = []
            tokens = input_ids.clone()
            logits_full = model(tokens)
            logits_nc.append(logits_full[:, -1, :])
            for _ in range(T_decode - 1):
                # Generate one token greedily
                next_tok = logits_full[:, -1, :].argmax(-1, keepdim=True)
                tokens = torch.cat([tokens, next_tok], dim=1)
                logits_full = model(tokens)
                logits_nc.append(logits_full[:, -1, :])

            # Cached: prefill + decode one at a time
            logits_c = []
            cache = KVCache(cfg, B, cfg.max_seq_len, input_ids.device, torch.float32)
            logits_prefill = model(input_ids, kv_cache=cache)
            logits_c.append(logits_prefill[:, -1, :])
            # Use same greedy tokens
            tokens_c = input_ids.clone()
            logits_for_decode = logits_prefill
            for _ in range(T_decode - 1):
                next_tok = logits_for_decode[:, -1, :].argmax(-1, keepdim=True)
                tokens_c = torch.cat([tokens_c, next_tok], dim=1)
                logits_step = model(next_tok, kv_cache=cache)
                logits_c.append(logits_step[:, -1, :])
                logits_for_decode = logits_step

        for step, (lnc, lc) in enumerate(zip(logits_nc, logits_c)):
            assert torch.allclose(lnc, lc, atol=1e-5), \
                f"Step {step}: cached vs non-cached mismatch, max diff={( lnc-lc).abs().max():.2e}"

    def test_cached_vs_nocache_gqa(self, small_cfg):
        self._run_cached_vs_nocache(small_cfg)

    def test_cached_vs_nocache_mha(self, mha_cfg):
        self._run_cached_vs_nocache(mha_cfg)

    def test_cached_vs_nocache_batch(self, small_cfg):
        """batch size > 1"""
        self._run_cached_vs_nocache(small_cfg, B=2)

    def test_prefill_logits_shape(self, small_cfg):
        model = MiniLM(small_cfg)
        input_ids = torch.randint(0, small_cfg.vocab_size, (2, 6))
        logits = model(input_ids)
        assert logits.shape == (2, 6, small_cfg.vocab_size)

    def test_cache_seqlen_increments(self, small_cfg):
        model = MiniLM(small_cfg)
        model.eval()
        cache = KVCache(small_cfg, 1, small_cfg.max_seq_len, 'cpu', torch.float32)
        assert cache.seq_len == 0
        ids = torch.randint(0, small_cfg.vocab_size, (1, 4))
        with torch.no_grad():
            model(ids, kv_cache=cache)
        assert cache.seq_len == 4


# ──────────────────────────────────── T6 ───────────────────────────────────

class TestT6Sampling:
    """T6: sample_next / generate 采样正确性"""

    def test_temperature_zero_is_argmax(self):
        torch.manual_seed(10)
        logits = torch.randn(4, 512)
        out = sample_next(logits, temperature=0.0)
        assert torch.equal(out, logits.argmax(dim=-1))

    def test_top_p_nucleus(self):
        """top_p=0.5 只从概率最高的 token 中采样"""
        torch.manual_seed(11)
        B, V = 1, 100
        # Make one token dominate
        logits = torch.full((B, V), -10.0)
        logits[0, 0] = 10.0  # token 0 has prob ≈ 1
        gen = torch.Generator()
        gen.manual_seed(0)
        for _ in range(20):
            tok = sample_next(logits, temperature=1.0, top_p=0.5, generator=gen)
            assert tok.item() == 0, f"Expected token 0, got {tok.item()}"

    def test_top_p_excludes_tail(self):
        """top_p 确保尾部概率为 0"""
        torch.manual_seed(12)
        B, V = 1, 10
        logits = torch.arange(V, dtype=torch.float).unsqueeze(0)  # increasing probs
        probs = torch.softmax(logits, dim=-1)
        cumsum = probs.sort(descending=True)[0].cumsum(-1)
        # With top_p=0.5, only sample from top tokens
        sampled = set()
        gen = torch.Generator()
        gen.manual_seed(99)
        for _ in range(100):
            tok = sample_next(logits, temperature=1.0, top_p=0.5, generator=gen)
            sampled.add(tok.item())
        # All sampled tokens should be from the top-p nucleus
        sorted_probs, sorted_idx = probs.sort(descending=True)
        cum = sorted_probs.cumsum(-1)
        # Nucleus: positions where cumsum <= top_p after shift
        nucleus = sorted_idx[0, (cum[0] - sorted_probs[0]) <= 0.5].tolist()
        for tok in sampled:
            assert tok in nucleus, f"Token {tok} outside nucleus {nucleus}"

    def test_top_p_one_keeps_full_support(self):
        """top_p=1.0 时整个词表都可达（漏写 top_p==1 分支/边界会挂）"""
        B, V = 1, 5
        logits = torch.zeros(B, V)  # 均匀分布
        gen = torch.Generator()
        gen.manual_seed(7)
        sampled = set()
        for _ in range(200):
            tok = sample_next(logits, temperature=1.0, top_p=1.0, generator=gen)
            sampled.add(tok.item())
        assert sampled == set(range(V)), \
            f"top_p=1.0 应能采到全部 {V} 个 token，实际只有 {sorted(sampled)}"

    def test_reproducible_with_generator(self, small_cfg):
        torch.manual_seed(13)
        model = MiniLM(small_cfg)
        model.eval()
        input_ids = torch.randint(0, small_cfg.vocab_size, (1, 4))

        gen1 = torch.Generator()
        gen1.manual_seed(42)
        result1 = generate(model, input_ids, max_new_tokens=5, temperature=0.8, top_p=0.9, generator=gen1)

        gen2 = torch.Generator()
        gen2.manual_seed(42)
        result2 = generate(model, input_ids, max_new_tokens=5, temperature=0.8, top_p=0.9, generator=gen2)

        assert torch.equal(result1, result2)

    def test_generate_output_shape(self, small_cfg):
        model = MiniLM(small_cfg)
        model.eval()
        input_ids = torch.randint(0, small_cfg.vocab_size, (2, 4))
        result = generate(model, input_ids, max_new_tokens=6)
        assert result.shape == (2, 10)

    def test_generate_no_cache(self, small_cfg):
        model = MiniLM(small_cfg)
        model.eval()
        input_ids = torch.randint(0, small_cfg.vocab_size, (1, 3))
        result = generate(model, input_ids, max_new_tokens=4, use_cache=False)
        assert result.shape == (1, 7)


# ──────────────────────────────────── T7 ───────────────────────────────────

QWEN_WEIGHT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'qwen2.5-0.5b')

@pytest.mark.skipif(
    not (os.path.isfile(os.path.join(QWEN_WEIGHT_DIR, 'config.json'))
         and os.path.isfile(os.path.join(QWEN_WEIGHT_DIR, 'model.safetensors'))),
    reason="Qwen2.5-0.5B weights incomplete; run: python3 scripts/download_qwen.py"
)
class TestT7QwenLoad:
    """T7: 加载 Qwen2.5-0.5B 官方权重，与 HF transformers 对比"""

    def test_load_qwen_shape(self):
        from minilm.model.convert_qwen import load_qwen
        model = load_qwen(QWEN_WEIGHT_DIR)
        assert isinstance(model, MiniLM)
        # Qwen2.5-0.5B: 24 layers, hidden=896
        assert model.cfg.num_layers == 24
        assert model.cfg.hidden_size == 896

    def test_logits_vs_transformers(self):
        from minilm.model.convert_qwen import load_qwen
        try:
            from transformers import AutoModelForCausalLM, AutoConfig
        except ImportError:
            pytest.skip("transformers not installed")

        model = load_qwen(QWEN_WEIGHT_DIR)
        model.eval()

        hf_model = AutoModelForCausalLM.from_pretrained(QWEN_WEIGHT_DIR, torch_dtype=torch.float32)
        hf_model.eval()

        torch.manual_seed(0)
        input_ids = torch.randint(0, 1000, (1, 8))

        with torch.no_grad():
            our_logits = model(input_ids)
            hf_logits = hf_model(input_ids).logits

        assert torch.allclose(our_logits, hf_logits, atol=5e-4), \
            f"Max diff: {(our_logits - hf_logits).abs().max():.2e}"


# ──────────────────────────────────── T8 ───────────────────────────────────

@pytest.mark.slow
class TestT8Overfit:
    """T8: 10M 参数小模型在 32 条随机序列上过拟合（≤300步, loss < 0.2）"""

    def test_overfit(self):
        torch.manual_seed(0)
        cfg = ModelConfig(
            vocab_size=512,
            hidden_size=256,
            intermediate_size=512,
            num_layers=6,
            num_heads=4,
            num_kv_heads=4,
            head_dim=64,
            max_seq_len=128,
            rope_theta=10000.0,
            attention_bias=False,
            tie_word_embeddings=True,
        )
        model = MiniLM(cfg)
        n_params = count_params(cfg)
        assert 1_000_000 < n_params < 50_000_000, f"Unexpected param count: {n_params:,}"

        B, T = 32, 64
        data = torch.randint(0, cfg.vocab_size, (B, T + 1))
        input_ids = data[:, :-1]
        labels = data[:, 1:]

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        model.train()

        final_loss = float('inf')
        for step in range(300):
            optimizer.zero_grad()
            logits = model(input_ids)  # (B, T, V)
            loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), labels.reshape(-1))
            assert not torch.isnan(loss), f"NaN loss at step {step}"
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            final_loss = loss.item()
            if final_loss < 0.2:
                break

        assert final_loss < 0.2, f"Model failed to overfit: final loss = {final_loss:.4f}"
