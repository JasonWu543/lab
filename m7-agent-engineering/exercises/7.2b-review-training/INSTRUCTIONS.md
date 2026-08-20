# Exercise 7.2b — Review a training-loop PR

**Time limit:** 45 minutes
**Goal:** review a green PR that adds mixed precision, gradient accumulation,
cosine scheduling, and checkpoint resume support.

## Setup

```bash
cd exercises/7.2b-review-training/pr
python3 -m pytest -q
```

## Review procedure

1. Spend 5 minutes reading `PR_DESCRIPTION.md` and mapping control flow.
2. Spend 25 minutes checking numerical semantics, state transitions, boundary
   cases, and ownership of mutable objects.
3. Spend 10 minutes reviewing the tests as production code: inspect oracles,
   inputs, assertions, and coverage gaps.
4. Spend 5 minutes ranking findings and completing `review_template.md`.

Record a file and approximate line, mechanism, severity, and a concrete
correction for every finding. Do not run the answer key during the timed review.

## Scoring

| Result | Score |
|---|---:|
| Correct P0 finding | 4 points |
| Correct P1 finding | 2 points |
| Correct P2 finding | 1 point |
| Useful exposing test | 1 bonus point each (max 3) |

Maximum base score: **37**. A finding must identify the mechanism, not only a
symptom. Finding a real defect outside the answer key may earn verbal bonus
credit but does not count toward the total score.

After 45 minutes, compare with `reference/7.2b-review-training/ANSWERS.md` and
complete the reflection section.
