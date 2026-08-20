# ANSWERS — 7.2c Data-pipeline review

Do not open this file until the 45-minute review is complete. Every snippet
below was executed against the submitted `pr/` on 2026-08-19.

## Catalogue

### 1. Packed documents can attend across record boundaries

- **File:line:** `packing.py:26-28`; **severity:** P0; **category:** isolation.
- **Mechanism:** one global triangular mask treats earlier documents as context.
- **Reproducer:** packing `[[10], [20]]` prints
  `cross_document_edge True` for query token 20 attending token 10.
- **Correction:** construct block-diagonal causal attention by document, or
  expose segment IDs consumed by the model.

### 2. Labels are not shifted for next-token prediction

- **File:line:** `packing.py:23-25`; **severity:** P0; **category:** numerical.
- **Mechanism:** labels clone inputs, teaching identity at each position.
- **Reproducer:** output is `inputs_labels [10, 2, 20, 2] [10, 2, 20, 2]`.
- **Correction:** pair `input_ids[:-1]` with `tokens[1:]` (or shift labels with
  the framework's documented convention) and mask the undefined endpoint.

### 3. Temporal data is shuffled before partitioning

- **File:line:** `splitting.py:12-17`; **severity:** P0; **category:** temporal.
- **Mechanism:** future and past records mix across train/validation boundaries.
- **Reproducer:** records 0–9 with seed 4 produce validation `[4, 3]`, while a
  chronological holdout is `[8, 9]`.
- **Correction:** split the ordered records first; shuffle only the train side.

### 4. Validation values influence normalization statistics

- **File:line:** `splitting.py:22-25`; **severity:** P0; **category:** isolation.
- **Mechanism:** concatenating partitions leaks holdout distribution into train
  preprocessing. Train `[0,0]`, validation `[100]` prints mean
  `33.33333206176758`, whereas train-only mean is `0.0`.
- **Correction:** fit mean/std on train and apply those frozen statistics to both.

### 5. Padding contributes to language-model loss

- **File:line:** `packing.py:25`; **severity:** P1; **category:** correctness.
- **Mechanism:** every loss-mask entry is true. `[7,2,0,0]` prints mask
  `[True, True, True, True]`.
- **Correction:** mask pad positions (and any undefined shifted-label position).

### 6. Tokenization duplicates an existing EOS

- **File:line:** `packing.py:38-39`; **severity:** P1; **category:** boundary.
- **Mechanism:** unlike the packer, this function appends unconditionally.
- **Reproducer:** a tokenizer returning `[7,2]` yields `[[7, 2, 2]]`.
- **Correction:** append only when the returned IDs do not already end in EOS;
  define one owner for termination.

### 7. Packing uses shared mutable workspace state

- **File:line:** `packing.py:6,10-14`; **severity:** P1; **category:** design.
- **Mechanism:** every default call owns the same list and clears/mutates it,
  making overlapping calls unsafe. Inspecting the default twice prints the same
  object identity (observed `5075616768 5075616768`).
- **Correction:** default to `None` and allocate per invocation.

### 8. Buffered shuffle ignores its seed

- **File:line:** `record_sampling.py:6,10`; **severity:** P1; **category:** determinism.
- **Mechanism:** `Random()` is constructed without the supplied seed. Two calls
  with seed 9 print `same_seed_equal False`.
- **Correction:** construct `random.Random(seed)`.

### 9. Buffered shuffle drops its final buffer

- **File:line:** `record_sampling.py:18-21`; **severity:** P1; **category:** boundary.
- **Mechanism:** once the source ends, buffered records are never yielded. Ten
  records with buffer three print `emitted_count 7 expected 10`.
- **Correction:** randomly drain the remaining buffer after input exhaustion.

### 10. Reservoir replacement range excludes the current slot

- **File:line:** `record_sampling.py:36`; **severity:** P1; **category:** numerical.
- **Mechanism:** the standard inclusive draw is from `0..index`; `randrange(index)`
  makes the second record always replace a size-one reservoir. Four seeds all
  print `[[1], [1], [1], [1]]` for two records.
- **Correction:** use `rng.randrange(index + 1)`.

### 11. Batches alias one list that is cleared after every yield

- **File:line:** `record_sampling.py:46-53`; **severity:** P1; **category:** mutation.
- **Mechanism:** consumers that retain batches see the same emptied container.
- **Reproducer:** materializing six records prints `batches [[], [], []]
  same_ids 1`.
- **Correction:** yield a new list/copy and replace the local accumulator.

### 12. Small positive validation fractions can yield no validation data

- **File:line:** `splitting.py:14-16`; **severity:** P1; **category:** boundary.
- **Mechanism:** floor rounding maps three records at 20% to zero; output sizes
  are `[3, 0]`.
- **Correction:** for nonzero fractions and nonempty eligible inputs, use an
  explicitly documented minimum of one (while preserving train data).

### 13. Ordered queue has quadratic list-front deletion

- **File:line:** `record_sampling.py:56-60`; **severity:** P2; **category:** performance.
- **Mechanism:** every `pop(0)` shifts the remaining list; 1000 records entail
  roughly `500500` shifted positions.
- **Correction:** iterate directly or use `collections.deque.popleft()`.

### 14. Label test asserts the submitted behavior as truth

- **File:line:** `test_data.py:13-15`; **severity:** P2; **category:** test.
- **Mechanism:** equality of labels and inputs is precisely the semantic failure
  in item 2; the reproducer prints `self_labels True`.
- **Correction:** assert a hand-written shifted example including EOS/padding.

### 15. Normalization oracle repeats the full-data statistic

- **File:line:** `test_data.py:35-41`; **severity:** P2; **category:** test.
- **Mechanism:** the oracle concatenates train and validation just like the
  implementation. The self-comparison prints `shared_oracle True`.
- **Correction:** calculate expected statistics from train alone and use a
  deliberately shifted validation distribution.

### 16. Shuffle test encodes truncated output as expected

- **File:line:** `test_data.py:44-47`; **severity:** P2; **category:** test.
- **Mechanism:** for eight inputs and buffer three it explicitly expects five;
  `expected_truncated_length True`.
- **Correction:** assert all eight identities occur exactly once and same-seed
  order is reproducible.

### 17. Attention test checks dimensions only

- **File:line:** `test_data.py:8-10`; **severity:** P2; **category:** test.
- **Mechanism:** a dense, causal, or cross-record mask all share `(4,4)`;
  the submitted incorrect mask prints `shape_only True`.
- **Correction:** assert individual allowed/blocked edges around a document
  boundary.

## Copyable reproducer fragments

Run this setup once from the repository root, then run any numbered fragment.
These are the exact operations used to obtain the outputs quoted above.

```python
import sys, torch
sys.path.insert(0, "m7-agent-engineering/exercises/7.2c-review-data/pr")
from packing import pack_documents, tokenize_records
from splitting import temporal_train_val_split, normalize_partitions
from record_sampling import buffered_shuffle, reservoir_sample, batch_records
e = pack_documents([[10], [20]], 4)[0]
```

1.
```python
print(bool(e["attention_mask"][2,0]))
```

2.
```python
print(e["input_ids"].tolist(),e["labels"].tolist())
```

3.
```python
print(temporal_train_val_split(list(range(10)),.2,seed=4)[1], [8,9])
```

4.
```python
_,_,mean,_=normalize_partitions(torch.tensor([[0.],[0.]]),torch.tensor([[100.]])); print(mean.item())
```

5.
```python
x=pack_documents([[7]],4)[0]; print(x["input_ids"].tolist(),x["loss_mask"].tolist())
```

6.
```python
print(tokenize_records([{"text":"x"}],lambda _: [7,2]))
```

7.
```python
print(id(pack_documents.__defaults__[-1]),id(pack_documents.__defaults__[-1]))
```

8.
```python
a=list(buffered_shuffle(range(10),3,seed=9)); b=list(buffered_shuffle(range(10),3,seed=9)); print(a==b)
```

9.
```python
print(len(list(buffered_shuffle(range(10),3))),10)
```

10.
```python
print([reservoir_sample(range(2),1,seed=s) for s in range(4)])
```

11.
```python
b=list(batch_records(range(6),2)); print(b,len({id(x) for x in b}))
```

12.
```python
print([len(x) for x in temporal_train_val_split(range(3),.2)])
```

13.
```python
print(sum(range(1,1001)))
```

14.
```python
print(torch.equal(e["labels"],e["input_ids"]))
```

15.
```python
c=torch.tensor([[1.],[1.],[1.]]); print(torch.allclose(c.mean(0),c.mean(0)))
```

16.
```python
print(len(list(buffered_shuffle(range(8),3)))==5)
```

17.
```python
print(e["attention_mask"].shape==(4,4))
```

## Summary

| Severity | Count | Items |
|---|---:|---|
| P0 | 4 | 1–4 |
| P1 | 8 | 5–12 |
| P2 | 5 | 13–17 |

Total: **17**. Temporal/isolation defects: **3** (items 1, 3, 4). Test-suite
blind spots include all production findings; items 14–17 are test defects.
