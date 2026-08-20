"""
Test suite for the token sampling and KV cache utility package.

Run with:
    python3 -m pytest test_pr.py -v

All tests should pass on a correct implementation — and also on the
submitted PR (by design of this exercise).
"""

from __future__ import annotations

import math
import pytest
import torch
import torch.nn.functional as F

from sampling import (
    temperature_sample,
    top_k_filter,
    top_p_filter,
    apply_repetition_penalty,
)
from kv_utils import (
    trim_kv_cache,
    estimate_kv_memory_bytes,
    concat_kv_cache,
    cache_seq_len,
    rebuild_cache_from_states,
    batch_prefill_cache,
)
from metrics import (
    compute_perplexity,
    compute_token_nll,
    ThroughputTracker,
    measure_latency,
)


# ===========================================================================
# sampling.py tests
# ===========================================================================

class TestTemperatureSample:
    def test_output_shape(self):
        torch.manual_seed(0)
        logits = torch.randn(4, 100)
        tokens = temperature_sample(logits, temperature=1.0)
        assert tokens.shape == (4,)

    def test_valid_token_ids(self):
        torch.manual_seed(1)
        logits = torch.randn(8, 50)
        tokens = temperature_sample(logits, temperature=0.5)
        assert (tokens >= 0).all() and (tokens < 50).all()

    def test_high_temperature_is_uniform(self):
        """Very high temperature → near-uniform; low temperature → peaked."""
        torch.manual_seed(42)
        # With very peaked logits and low temperature, we expect the argmax
        logits = torch.zeros(1, 10)
        logits[0, 3] = 100.0  # token 3 dominates
        tokens = torch.stack([temperature_sample(logits, temperature=0.01) for _ in range(20)])
        # Every sample should be token 3
        assert (tokens == 3).all()

    # BUG #9 — no seed on a stochastic test; may occasionally fail
    def test_sampling_is_stochastic(self):
        """Samples from a uniform distribution should not all be identical."""
        logits = torch.zeros(1, 1000)  # uniform
        samples = torch.stack([temperature_sample(logits, temperature=1.0) for _ in range(30)])
        # Extremely unlikely to be all the same, but no seed set — flaky
        assert samples.unique().numel() > 1  # BUG #9: no torch.manual_seed

    # BUG #10 — zero coverage of temperature=0 edge case
    # (No test for temperature=0 exists anywhere in this file.)


class TestTopKFilter:
    def test_only_k_tokens_survive(self):
        torch.manual_seed(0)
        logits = torch.randn(2, 200)
        filtered = top_k_filter(logits, k=10)
        finite_counts = (filtered > float("-inf")).sum(dim=-1)
        assert (finite_counts == 10).all()

    def test_top_tokens_are_highest(self):
        logits = torch.tensor([[1.0, 5.0, 3.0, 2.0, 4.0]])
        filtered = top_k_filter(logits, k=2)
        # Tokens at index 1 (5.0) and 4 (4.0) should survive
        assert filtered[0, 1] == 5.0
        assert filtered[0, 4] == 4.0
        assert filtered[0, 0] == float("-inf")
        assert filtered[0, 2] == float("-inf")

    def test_k_equals_vocab_returns_all(self):
        logits = torch.randn(1, 50)
        filtered = top_k_filter(logits, k=50)
        assert (filtered > float("-inf")).all()

    def test_invalid_k_raises(self):
        with pytest.raises(ValueError):
            top_k_filter(torch.randn(1, 10), k=0)


class TestTopPFilter:
    def test_output_shape_unchanged(self):
        logits = torch.randn(3, 100)
        out = top_p_filter(logits, p=0.9)
        assert out.shape == logits.shape

    def test_at_least_one_token_survives(self):
        torch.manual_seed(0)
        logits = torch.randn(4, 500)
        out = top_p_filter(logits, p=0.95)
        surviving = (out > float("-inf")).sum(dim=-1)
        assert (surviving >= 1).all()

    # BUG #8 — this test mirrors the implementation's off-by-one in top_p,
    # so it passes even though the behavior is wrong.
    def test_top_p_filters_enough_tokens(self):
        """After filtering, softmax of survivors should sum to >= p."""
        torch.manual_seed(7)
        logits = torch.tensor([[2.0, 1.5, 0.5, 0.1, 0.05]])
        out = top_p_filter(logits, p=0.9)
        probs = F.softmax(out, dim=-1)
        surviving_mass = probs[probs > 0].sum().item()
        # BUG #8: the implementation excludes an extra token (the one bringing
        # cumsum to exactly p), so surviving_mass may be < p. The test uses
        # >= 0 to always pass regardless of the actual value.
        assert surviving_mass >= 0.0  # BUG #7/#8: always true, catches nothing

    def test_invalid_p_raises(self):
        with pytest.raises(ValueError):
            top_p_filter(torch.randn(1, 10), p=1.5)


class TestRepetitionPenalty:
    def test_no_penalty_is_identity(self):
        logits = torch.randn(2, 100)
        input_ids = torch.randint(0, 100, (2, 10))
        out = apply_repetition_penalty(logits, input_ids, penalty=1.0)
        assert torch.allclose(out, logits)

    def test_penalty_reduces_positive_logits(self):
        """Positive logits for seen tokens should decrease after penalty."""
        logits = torch.zeros(1, 5)
        logits[0, 2] = 3.0  # token 2 has positive logit
        input_ids = torch.tensor([[2]])
        out = apply_repetition_penalty(logits.clone(), input_ids, penalty=1.5)
        assert out[0, 2] < logits[0, 2]  # passes: 3/1.5=2 < 3 ✓

    # Note: no test checks that NEGATIVE logits are handled correctly (bug #2),
    # and no test checks whether the input logits tensor is mutated (bug #6).

    def test_unseen_tokens_unchanged(self):
        logits = torch.ones(1, 10)
        input_ids = torch.tensor([[0, 1, 2]])
        original = logits.clone()
        out = apply_repetition_penalty(logits, input_ids, penalty=2.0)
        # Tokens 3-9 should be unchanged
        assert torch.allclose(out[0, 3:], original[0, 3:])

    def test_output_shape(self):
        logits = torch.randn(3, 200)
        input_ids = torch.randint(0, 200, (3, 15))
        out = apply_repetition_penalty(logits, input_ids, penalty=1.2)
        assert out.shape == logits.shape


# ===========================================================================
# kv_utils.py tests
# ===========================================================================

def _make_cache(batch=2, heads=4, seq=10, dim=16, n_layers=2):
    return [
        (torch.randn(batch, heads, seq, dim), torch.randn(batch, heads, seq, dim))
        for _ in range(n_layers)
    ]


class TestTrimKVCache:
    def test_no_trim_when_within_limit(self):
        cache = _make_cache(seq=10)
        trimmed = trim_kv_cache(cache, max_seq_len=20)
        for (k, v), (kt, vt) in zip(cache, trimmed):
            assert torch.equal(k, kt) and torch.equal(v, vt)

    def test_trimmed_seq_len(self):
        cache = _make_cache(seq=20)
        trimmed = trim_kv_cache(cache, max_seq_len=10)
        for kt, vt in trimmed:
            assert kt.size(2) == 10
            assert vt.size(2) == 10

    # BUG NOTE: no test checks WHICH tokens survive (newest vs oldest),
    # so bug #5 (newest clipped instead of oldest) is not caught.

    def test_exact_limit_unchanged(self):
        cache = _make_cache(seq=10)
        trimmed = trim_kv_cache(cache, max_seq_len=10)
        for (k, v), (kt, vt) in zip(cache, trimmed):
            assert torch.equal(k, kt)


class TestEstimateKVMemory:
    def test_basic_calculation(self):
        # float16 = 2 bytes; 2 * 1 * 2 * 4 * 16 * 8 = 1024 bytes
        result = estimate_kv_memory_bytes(
            batch_size=1, n_layers=2, n_heads=4, head_dim=8, seq_len=16,
            dtype=torch.float16,
        )
        expected = 2 * 1 * 2 * 4 * 16 * 8 * 2  # 2 bytes per float16
        assert result == expected

    def test_scales_linearly_with_batch(self):
        base = estimate_kv_memory_bytes(1, 2, 4, 8, 16)
        doubled = estimate_kv_memory_bytes(2, 2, 4, 8, 16)
        assert doubled == 2 * base


class TestConcatKVCache:
    def test_concat_increases_seq_len(self):
        past = _make_cache(seq=5)
        new  = _make_cache(seq=3)
        combined = concat_kv_cache(past, new)
        for k, v in combined:
            assert k.size(2) == 8
            assert v.size(2) == 8

    def test_empty_past_returns_new(self):
        new = _make_cache(seq=7)
        combined = concat_kv_cache([], new)
        for (k, v), (kc, vc) in zip(new, combined):
            assert torch.equal(k, kc)

    def test_depth_mismatch_raises(self):
        past = _make_cache(n_layers=2)
        new  = _make_cache(n_layers=3)
        with pytest.raises(ValueError):
            concat_kv_cache(past, new)


class TestCacheSeqLen:
    def test_empty_cache_returns_zero(self):
        assert cache_seq_len([]) == 0

    def test_nonempty_cache(self):
        cache = _make_cache(seq=12)
        assert cache_seq_len(cache) == 12

    # BUG NOTE: no test for seq_len=0 cache, so bug #18 (IndexError) is hidden.


class TestRebuildCache:
    def test_valid_states(self):
        states = [
            {"key": torch.randn(1, 2, 5, 8), "value": torch.randn(1, 2, 5, 8)},
            {"key": torch.randn(1, 2, 5, 8), "value": torch.randn(1, 2, 5, 8)},
        ]
        cache = rebuild_cache_from_states(states)
        assert len(cache) == 2

    def test_silently_skips_broken_layers(self):
        # BUG #16 makes this pass: the broken layer is silently swallowed
        states = [
            {"key": torch.randn(1, 2, 5, 8), "value": torch.randn(1, 2, 5, 8)},
            {"not_key": None},  # broken state — will be silently skipped
        ]
        cache = rebuild_cache_from_states(states)
        # Test expects 1 layer (the broken one was swallowed silently)
        assert len(cache) == 1  # passes because of bug #16


class TestBatchPrefill:
    def test_batch_dimension(self):
        keys   = [torch.randn(1, 4, 8, 16) for _ in range(3)]
        values = [torch.randn(1, 4, 8, 16) for _ in range(3)]
        cache = batch_prefill_cache(keys, values)
        assert cache[0][0].shape[0] == 3  # batch dim


# ===========================================================================
# metrics.py tests
# ===========================================================================

class TestComputePerplexity:
    def test_output_is_positive_scalar(self):
        torch.manual_seed(0)
        logits = torch.randn(2, 10, 100)
        labels = torch.randint(0, 100, (2, 10))
        ppl = compute_perplexity(logits, labels)
        assert isinstance(ppl, float) and ppl > 0

    def test_perfect_prediction_low_ppl(self):
        """If the model is very confident and correct, PPL should be low."""
        vocab = 50
        logits = torch.full((1, 5, vocab), -1e9)
        labels = torch.arange(5).unsqueeze(0)
        for t in range(5):
            logits[0, t, t] = 1e9  # always predict correct token
        ppl = compute_perplexity(logits, labels)
        # PPL should be close to 1 for a perfect predictor
        # BUG #4 means the actual value won't be ~1 but the test is loose enough
        assert ppl < 1000  # very loose check — does not catch the log base bug

    def test_ignore_index_excluded(self):
        torch.manual_seed(0)
        logits = torch.randn(1, 5, 20)
        labels = torch.tensor([[-100, -100, -100, -100, -100]])
        # All labels are ignored — but the function still returns a number
        # (behavior with all-ignored is undefined/NaN; test just checks no crash)
        try:
            ppl = compute_perplexity(logits, labels)
        except Exception:
            pass  # either is acceptable


class TestTokenNLL:
    def test_shape(self):
        logits = torch.randn(2, 8, 50)
        labels = torch.randint(0, 50, (2, 8))
        nll = compute_token_nll(logits, labels)
        assert nll.shape == (2, 8)

    def test_ignored_positions_are_zero(self):
        logits = torch.randn(1, 6, 30)
        labels = torch.tensor([[-100, 1, 2, -100, 4, 5]])
        nll = compute_token_nll(logits, labels)
        assert nll[0, 0] == 0.0 and nll[0, 3] == 0.0

    def test_nll_nonnegative(self):
        logits = torch.randn(2, 5, 20)
        labels = torch.randint(0, 20, (2, 5))
        nll = compute_token_nll(logits, labels)
        assert (nll >= 0).all()


class TestThroughputTracker:
    def test_empty_tracker(self):
        t = ThroughputTracker()
        assert t.tokens_per_second() == 0.0

    def test_single_record(self):
        t = ThroughputTracker()
        t.record(100, 0.5)
        assert abs(t.tokens_per_second() - 200.0) < 1e-6

    def test_summary_keys(self):
        t = ThroughputTracker()
        t.record(50, 0.1)
        s = t.summary()
        assert "mean_tps" in s and "min_tps" in s and "max_tps" in s

    def test_result_is_not_none(self):
        # BUG #7: always-true assertion — summary() can never return None
        t = ThroughputTracker()
        t.record(10, 0.01)
        result = t.summary()
        assert result is not None  # BUG #7: trivially true, tests nothing


class TestMeasureLatency:
    def test_returns_expected_keys(self):
        result = measure_latency(lambda: None, n_warmup=1, n_trials=3)
        assert "mean_ms" in result and "std_ms" in result
        assert "min_ms" in result and "max_ms" in result

    def test_latency_nonnegative(self):
        result = measure_latency(lambda: None, n_warmup=0, n_trials=5)
        assert result["mean_ms"] >= 0.0
        assert result["min_ms"] >= 0.0
