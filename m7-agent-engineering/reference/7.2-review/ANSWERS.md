# ANSWERS — 7.2 Fast PR Review

**Do not open until you have spent at least 30 minutes on the review.**

This document is the answer key for the 18 deliberately planted bugs in the
`pr/` package. Each entry gives: location, category, severity, mechanism,
a minimal reproducer, and the correct fix.

---

## Bug Catalogue

### Bug 1 — top-p off-by-one (cumsum threshold)

- **File**: `sampling.py` line 113–121
- **Category**: Numerical / Correctness
- **Severity**: P1 — wrong results at an exact cumulative-probability boundary;
  the retained set is one token narrower than the conventional strict-boundary
  top-p policy used by common implementations
- **Mechanism**: `sorted_indices_to_remove = cumulative_probs >= p` uses `>=`
  instead of `>`. The shift (`[..., 1:] = [..., :-1]`) is intended to keep
  the token that first exceeds the threshold, but because the mask is already
  set on tokens that hit the boundary exactly, the shift doesn't fully
  compensate. At `cumsum == p`, the next token is excluded even though this
  exercise defines top-p as retaining through the first prefix that *strictly
  exceeds* `p`.
- **Minimal reproducer**:

```python
import torch, torch.nn.functional as F
from sampling import top_p_filter

# Two tokens have equal probability summing to exactly p
logits = torch.tensor([[2.0, 2.0, -100.0, -100.0]])
out = top_p_filter(logits, p=0.5)
probs = F.softmax(out, dim=-1)
surviving = (out > float("-inf")).sum().item()
print(surviving)  # buggy: 1 token; correct under this contract: 2
```

- **Fix**: Change `>=` to `>`:
  ```python
  sorted_indices_to_remove = cumulative_probs > p
  ```

---

### Bug 2 — repetition penalty wrong direction for negative logits

- **File**: `sampling.py` line 165
- **Category**: Numerical / Correctness
- **Severity**: P1 — seen tokens with negative logits are made more likely
  instead of being suppressed, biasing sampling toward repetition
- **Mechanism**: The code does `score[b, token_ids] /= penalty` for all seen
  tokens. For a positive logit L, dividing by penalty > 1 makes it smaller
  (correct: reduces probability). For a negative logit -L, dividing by
  penalty > 1 makes it *less* negative (less negative = higher logit = higher
  probability, the wrong direction). The sign-aware repetition-penalty rule uses
  `logit / penalty` for positive logits and `logit * penalty` for negative
  logits. By doing only `/= penalty` for all, negative logits become *less*
  negative, which *increases* their probability — the opposite of penalizing
  them. Example: logit = -2, penalty = 1.5 → -2/1.5 = -1.33 (less negative,
  higher prob). Correct should be: -2 * 1.5 = -3.0 (more negative, lower prob).
- **Minimal reproducer**:

```python
import torch
from sampling import apply_repetition_penalty

logits = torch.tensor([[-3.0, 1.0, -1.0, 0.5]])
input_ids = torch.tensor([[0, 1, 2, 3]])  # all tokens are "seen"
out = apply_repetition_penalty(logits.clone(), input_ids, penalty=1.5)
# Bug: token 0 logit goes from -3.0 -> -2.0 (less negative = more likely)
# Correct: should go from -3.0 -> -4.5 (more negative = less likely)
print(out[0, 0].item())  # buggy: -2.0; correct: -4.5
```

- **Fix**:
  ```python
  pos_mask = score[b, token_ids] > 0
  score[b, token_ids[pos_mask]]  /= penalty
  score[b, token_ids[~pos_mask]] *= penalty
  ```

---

### Bug 3 — temperature=0 reaches invalid multinomial probabilities

- **File**: `sampling.py` lines 38 and 65
- **Category**: Numerical / Correctness
- **Severity**: P0 — crashes when callers use temperature=0 to request greedy
  decoding (a common API convention)
- **Mechanism**: `_validate_temperature` is a no-op (just `pass`). Line 65
  does tensor division by `temperature` with no guard. With `temperature=0`,
  PyTorch produces `inf`/`nan`; `torch.multinomial` then raises `RuntimeError`
  because its probability tensor is invalid.
- **Minimal reproducer**:

```python
import torch
from sampling import temperature_sample

logits = torch.randn(1, 100)
temperature_sample(logits, temperature=0)  # RuntimeError: invalid probabilities
```

- **Fix**: Guard in `_validate_temperature` or inline:
  ```python
  if temperature == 0:
      return logits.argmax(dim=-1)
  if temperature < 0:
      raise ValueError("temperature must be > 0")
  ```

---

### Bug 4 — perplexity log base confusion (nats vs bits)

- **File**: `metrics.py` line 60
- **Category**: Numerical / Correctness
- **Severity**: P0 — perplexity values are systematically inflated by a factor
  of `exp(H / ln(2) - H)` where H is cross-entropy in nats; for H≈3, this is
  about 3.77x
- **Mechanism**: `F.cross_entropy` returns cross-entropy in *nats* (natural
  log). Line 60 divides by `math.log(2)` to convert to *bits*, but then line
  64 exponentiates with `exp()` as if the loss is still in nats. The correct
  formula is `exp(avg_loss_in_nats)`. The code computes `exp(avg_loss_in_bits)`
  which equals `exp(H / ln(2))` instead of `exp(H)`.
- **Minimal reproducer**:

```python
import torch, math
from metrics import compute_perplexity

# Uniform logits over 10 tokens → true PPL = 10
logits = torch.zeros(1, 5, 10)
labels = torch.zeros(1, 5, dtype=torch.long)
ppl = compute_perplexity(logits, labels)
# Correct PPL ≈ 10; buggy PPL ≈ 10^(1/log2(e)) ≈ 10^0.693 ≈ 4.97... no
# Actually: loss_nats = ln(10) ≈ 2.303; loss_bits = 2.303/0.693 ≈ 3.322
# buggy PPL = exp(3.322) ≈ 27.7; correct PPL = exp(2.303) = 10
print(f"PPL={ppl:.2f}")  # buggy: ~27.7; correct: 10.0
```

- **Fix**: Remove the `/ math.log(2)` conversion and exponentiate the raw loss:
  ```python
  avg_loss = loss_per_token[valid_mask].mean()
  return avg_loss.exp().item()
  ```

---

### Bug 5 — sliding window clips newest tokens instead of oldest

- **File**: `kv_utils.py` lines 59–60
- **Category**: Numerical / Correctness
- **Severity**: P0 — the entire KV cache context is lost; model attends to
  the wrong tokens, causing incoherent generation
- **Mechanism**: `k[:, :, :-overflow, :]` removes the *last* `overflow` tokens
  (the most recently generated). Correct: `k[:, :, overflow:, :]` removes the
  *first* `overflow` tokens (the oldest, to slide the window forward).
- **Minimal reproducer**:

```python
import torch
from kv_utils import trim_kv_cache

# Cache contains tokens 0..19 (simulated by filling with position index)
k = torch.arange(20, dtype=torch.float).view(1, 1, 20, 1).expand(1, 1, 20, 8).clone()
v = k.clone()
cache = [(k, v)]

trimmed = trim_kv_cache(cache, max_seq_len=10)
kt, vt = trimmed[0]
# Which 10 tokens survived?
# Correct (oldest removed): tokens 10..19 survive → first value ≈ 10
# Buggy (newest removed):   tokens  0..9 survive → first value ≈  0
print(kt[0, 0, 0, 0].item())  # buggy: 0.0; correct: 10.0
```

- **Fix**:
  ```python
  trimmed.append((
      k[:, :, overflow:, :],
      v[:, :, overflow:, :],
  ))
  ```

---

### Bug 6 — in-place modification of caller's logits tensor

- **File**: `sampling.py` line 158 (`score = logits`) + line 165
- **Category**: Numerical / Correctness
- **Severity**: P0 — silently mutates the caller's tensor; any downstream use
  of the original logits (e.g., top-k after repetition penalty) is corrupted
- **Mechanism**: `score = logits` creates an alias (no copy). The subsequent
  `score[b, token_ids] /= penalty` modifies `logits` in place.
- **Minimal reproducer**:

```python
import torch
from sampling import apply_repetition_penalty

logits = torch.ones(1, 10)
original = logits.clone()
input_ids = torch.tensor([[0, 1, 2]])

apply_repetition_penalty(logits, input_ids, penalty=2.0)
print(torch.allclose(logits, original))  # False — caller's tensor mutated!
```

- **Fix**: `score = logits.clone()` on line 157.

---

### Bug 7 — always-true assertion

- **File**: `test_pr.py` line 360
- **Category**: Test Bug
- **Severity**: P2 — provides false confidence by barely constraining the result
- **Mechanism**: `assert result is not None` where `result` is the return
  value of `ThroughputTracker.summary()`. It catches a regression to `None`, but
  accepts `{}` or a dict with incorrect values, so it never verifies throughput.
- **Minimal reproducer**:

```python
# The following assertion would pass even if summary() returned {} or
# any other non-None value:
result = {"mean_tps": 0}
assert result is not None  # trivially True
```

- **Fix**: Assert something meaningful:
  ```python
  assert result["mean_tps"] > 0
  assert result["n_batches"] == 1
  ```

---

### Bug 8 — test mirrors implementation bug (top-p coverage)

- **File**: `test_pr.py` lines 119–131
- **Category**: Test Bug
- **Severity**: P1 — the test for top-p nucleus filtering passes even when
  the implementation is wrong
- **Mechanism**: `TestTopPFilter::test_top_p_filters_enough_tokens` ends with
  `assert surviving_mass >= 0.0`, which cannot constrain a softmax probability.
  Also, softmax is applied *after* filtering, so surviving entries renormalize
  to 1; merely changing the threshold to `p` would still be vacuous. Measure the
  retained mask against the original distribution instead.
- **Fix**:
  ```python
  original_probs = F.softmax(logits, dim=-1)
  retained = torch.isfinite(out)
  assert original_probs[retained].sum().item() >= 0.9
  ```

---

### Bug 9 — flaky test with no random seed

- **File**: `test_pr.py` lines 67–73
- **Category**: Test Bug
- **Severity**: P2 — non-deterministic; very rarely fails but theoretically
  possible (probability `1000^-29` for 30 uniform draws — negligible here, but
  *principle* is wrong and will bite on smaller vocabs)
- **Mechanism**: `test_sampling_is_stochastic` calls `temperature_sample`
  30 times inside a loop with no `torch.manual_seed`, producing different
  results each CI run. With a vocabulary of 1000 and 30 samples the failure
  probability is astronomically small, but the design is incorrect.
- **Fix**: Either fix the seed and redesign the test to check entropy, or
  use a statistical test with a known distribution.

---

### Bug 10 — zero coverage of temperature=0 edge case

- **File**: `test_pr.py` (missing test)
- **Category**: Test Bug
- **Severity**: P0 — bug #3 (ZeroDivisionError on temperature=0) is
  completely invisible to the test suite
- **Mechanism**: No test in the file calls `temperature_sample` with
  `temperature=0`. A comment at line 75 even notes this explicitly. Any
  reviewer who only reads test results would believe temperature=0 works.
- **Fix**: Add:
  ```python
  def test_temperature_zero_greedy(self):
      logits = torch.tensor([[1.0, 5.0, 2.0]])
      tokens = temperature_sample(logits, temperature=0)
      assert tokens.item() == 1  # argmax
  ```

---

### Bug 11 — warmup only for compiled variant

- **File**: `bench.py` lines 63–71
- **Category**: Benchmark Fairness Bug
- **Severity**: P2 — makes the eager implementation look slower than it is;
  first few timed iterations absorb Python/PyTorch startup overhead
- **Mechanism**: The benchmark runs 5 warmup iterations for `_compiled_top_p`
  but zero warmup iterations for `_eager_top_p`. The eager variant's first
  calls can include one-time dispatcher, allocator, and backend initialization.
  Module import has already completed, and eager execution does not JIT-trace.
- **Fix**: Add equivalent warmup for the eager variant:
  ```python
  for _ in range(5):
      _eager_top_p(warmup_logits, p)
  for _ in range(5):
      _compiled_top_p(warmup_logits, p)
  ```

---

### Bug 12 — unequal batch sizes in throughput comparison

- **File**: `bench.py` lines 76–77
- **Category**: Benchmark Fairness Bug
- **Severity**: P1 — throughput (tokens/s) numbers are incomparable; the
  comparison is meaningless and will mislead whoever reads it
- **Mechanism**: `eager_batch = 64` but `compiled_batch = 1`. The eager
  method processes 64× more tokens per call, making its tokens/s artificially
  higher. A fair benchmark must use identical batch sizes.
- **Fix**: Use the same batch size for both:
  ```python
  batch_size = 64
  eager_logits    = torch.randn(batch_size, vocab_size)
  compiled_logits = torch.randn(batch_size, vocab_size)
  ```

---

### Bug 13 — per-row top-k calls serialize batched work in Python

- **File**: `sampling.py` lines 88–92
- **Category**: Performance Bug
- **Severity**: P2 — O(batch) framework/kernel invocations and Python dispatches
  replace one batched call, losing vectorization and device parallelism
- **Mechanism**: `top_k_filter` loops over `range(logits.size(0))` and calls
  `torch.topk(logits[i], k)` inside the loop. PyTorch's `topk` supports batched
  input. Both versions process the same asymptotic `batch * vocab` elements;
  the defect is dispatch/launch overhead and serialization, not 512x more sort
  work.
- **Fix**:
  ```python
  top_vals, top_idx = torch.topk(logits, k, dim=-1)
  result = torch.full_like(logits, float("-inf"))
  result.scatter_(-1, top_idx, top_vals)
  return result
  ```

---

### Bug 14 — zero-initialize then refill batch tensors in Python

- **File**: `kv_utils.py` lines 213–215
- **Category**: Performance Bug
- **Severity**: P2 — needlessly zero-initializes output tensors and then fills
  them row by row in Python instead of using a stacking primitive
- **Mechanism**: Under the annotated input type (`list[Tensor]`), `n_layers` is
  always 1, so the earlier 64-allocation claim is false. The real issue is that
  the two output tensors are zero-filled and every element is then overwritten
  through Python assignments. The attempted nested-list branch is not a working
  multi-layer API.
- **Fix**: Use `torch.empty(...)` and let the inner loop fill values, or
  use `torch.stack` outside the loop:
  ```python
  batch_k = torch.stack([k[0] for k in keys_list], dim=0)
  batch_v = torch.stack([v[0] for v in values_list], dim=0)
  result.append((batch_k, batch_v))
  ```

---

### Bug 15 — mutable default argument

- **File**: `sampling.py` line 133
- **Category**: Design / Robustness Bug
- **Severity**: P1 — `_seen_cache` list is shared across all calls; it grows
  unboundedly, leaking memory and accumulating stale state across unrelated
  invocations
- **Mechanism**: `def apply_repetition_penalty(..., _seen_cache: list = []):`
  The default `[]` is evaluated once at function definition time and shared.
  Every call appends to the same list.
- **Minimal reproducer**:

```python
from sampling import apply_repetition_penalty
import torch

logits = torch.randn(1, 10)
ids = torch.randint(0, 10, (1, 5))
apply_repetition_penalty(logits, ids, penalty=1.2)
apply_repetition_penalty(logits, ids, penalty=1.2)
# Now inspect the default cache — it has 2 entries from separate calls!
import inspect
sig = inspect.signature(apply_repetition_penalty)
print(sig.parameters["_seen_cache"].default)  # [(1,5), (1,5)]
```

- **Fix**: Use `None` as default and create a new list inside:
  ```python
  def apply_repetition_penalty(..., _seen_cache=None):
      if _seen_cache is None:
          _seen_cache = []
  ```

---

### Bug 16 — silent exception swallowing

- **File**: `kv_utils.py` lines 178–179
- **Category**: Design / Robustness Bug
- **Severity**: P1 — silently drops corrupted or missing KV cache layers;
  the caller has no way to know data was lost, leading to subtle model errors
- **Mechanism**: `except Exception: pass` swallows all exceptions including
  `KeyError` (missing "key"/"value"), shape mismatches, or device errors.
  The function returns a shorter-than-expected cache with no indication of error.
- **Fix**: At minimum, log the error; better to re-raise or raise a
  descriptive `ValueError`:
  ```python
  except Exception as e:
      raise ValueError(f"Failed to reconstruct layer {len(cache)}: {e}") from e
  ```

---

### Bug 17 — docstring contradicts behavior

- **File**: `sampling.py` lines 29–36 and lines 148–149
- **Category**: Design / Robustness Bug
- **Severity**: P2 — misleads callers; the docstring for `apply_repetition_penalty`
  says "returns probabilities summing to 1" but the function returns logits;
  the docstring for `_validate_temperature` says it raises `ValueError` for
  non-positive temperature, but the body is `pass`
- **Mechanism**: Both docstrings make promises the implementation does not keep.
  A caller reading the docstring would expect validated inputs and normalized
  outputs.
- **Fix**: Update docstrings to match actual behavior, or fix the behavior to
  match the docstrings.

---

### Bug 18 — crash on edge input (empty KV cache seq_len=0)

- **File**: `kv_utils.py` line 152
- **Category**: Design / Robustness Bug
- **Severity**: P1 — crashes with `IndexError` on a valid input that can
  arise after aggressive trimming (max_seq_len=0) or padded batches
- **Mechanism**: `cache_seq_len` does `_ = k[:, :, 0, :]` before returning
  `k.size(2)`. When `seq_len == 0`, indexing position 0 raises `IndexError:
  index 0 is out of bounds for dimension 2 with size 0`.
- **Minimal reproducer**:

```python
import torch
from kv_utils import cache_seq_len

# Empty sequence — valid after over-aggressive trimming
k = torch.zeros(2, 4, 0, 16)
v = torch.zeros(2, 4, 0, 16)
cache_seq_len([(k, v)])  # IndexError!
```

- **Fix**: Remove the spurious indexing:
  ```python
  k, _ = cache[0]
  return k.size(2)
  ```

---

## Summary table

| # | File | Line | Category | Severity | Short name |
|---|------|------|----------|----------|-----------|
| 1 | sampling.py | 116 | Numerical | P1 | top-p off-by-one (>= vs >) |
| 2 | sampling.py | 165 | Numerical | P1 | repetition penalty wrong sign branch |
| 3 | sampling.py | 38/65 | Numerical | P0 | temperature=0 invalid probabilities/crash |
| 4 | metrics.py | 60 | Numerical | P0 | perplexity log base confusion |
| 5 | kv_utils.py | 59 | Numerical | P0 | sliding window clips newest not oldest |
| 6 | sampling.py | 158 | Numerical | P0 | in-place mutation of caller tensor |
| 7 | test_pr.py | 360 | Test | P2 | severely under-specified assertion |
| 8 | test_pr.py | 131 | Test | P1 | vacuous top-p mass assertion |
| 9 | test_pr.py | 73 | Test | P2 | flaky test — no seed |
| 10 | test_pr.py | — | Test | P0 | zero coverage of temperature=0 |
| 11 | bench.py | 63 | Benchmark | P2 | warmup only for compiled variant |
| 12 | bench.py | 76 | Benchmark | P1 | unequal batch sizes (64 vs 1) |
| 13 | sampling.py | 88 | Performance | P2 | per-row topk serializes batched work |
| 14 | kv_utils.py | 213 | Performance | P2 | wasted zero-fill and Python copy loop |
| 15 | sampling.py | 133 | Design | P1 | mutable default argument |
| 16 | kv_utils.py | 178 | Design | P1 | silent exception swallowing |
| 17 | sampling.py | 29/148 | Design | P2 | docstring contradicts behavior |
| 18 | kv_utils.py | 152 | Design | P1 | crash on seq_len=0 edge case |

## Additional accidental issue (not part of scored 18)

This issue was not one of the deliberately planted review targets and does
**not** count toward the 18-point score. The exercise code is intentionally
left unchanged so the scored material remains stable.

### `batch_prefill_cache` does not preserve or validate its inputs

- **File**: `kv_utils.py` lines 204–219
- **Category**: Correctness / Robustness
- **Severity**: P1
- **Mechanism**:
  - `zip(keys_list, values_list)` silently truncates unequal lists, while the
    output batch is allocated from `len(keys_list)`. Missing values therefore
    leave zero-filled rows instead of raising an error.
  - Both outputs use `k_ref.dtype`, silently converting values with a different
    dtype.
  - `torch.zeros(...)` does not inherit `k_ref.device`, so CUDA inputs allocate
    CPU destinations and fail during assignment.
- **Minimal reproducer**:

  ```python
  import torch
  from kv_utils import batch_prefill_cache

  k = torch.ones(1, 1, 1, 1, dtype=torch.float32)
  v = torch.ones(1, 1, 1, 1, dtype=torch.float64)

  # Unequal lengths are accepted; the second output row remains zero-filled.
  out = batch_prefill_cache([k, k], [v])
  print(out[0][1].shape, out[0][1][1].item())  # (2, 1, 1, 1), 0.0

  # The float64 value is silently converted to the key's float32 dtype.
  print(out[0][1].dtype)  # torch.float32

  # On a CUDA machine, CUDA inputs still create CPU destination tensors and
  # assignment raises a device-mismatch error.
  if torch.cuda.is_available():
      batch_prefill_cache([k.cuda()], [v.cuda()])
  ```

- **Expected resolution outside this exercise**: Validate non-empty, equal
  input lengths and compatible shapes; preserve the key and value devices and
  dtypes independently. Do not apply that fix to this scored `pr/` snapshot.
