# Exercise 7.2c — Review a data-pipeline PR

**Time limit:** 45 minutes
**Goal:** review a green PR that adds tokenize/pack, shuffle, and train/validation
partitioning for sequence-model data.

## Setup

```bash
cd exercises/7.2c-review-data/pr
python3 -m pytest -q
```

## Review procedure

1. Spend 5 minutes mapping data ownership and the order of transformations.
2. Spend 25 minutes checking sequence semantics, partition isolation, edge
   cases, determinism, and mutation boundaries.
3. Spend 10 minutes auditing fixtures, assertions, and test oracles.
4. Spend 5 minutes ranking findings in `review_template.md`.

For each finding, record a location, mechanism, impact, severity, and practical
correction. Treat tests as part of the submitted production change.

## Scoring

| Result | Score |
|---|---:|
| Correct P0 finding | 4 points |
| Correct P1 finding | 2 points |
| Correct P2 finding | 1 point |
| Useful exposing test | 1 bonus point each (max 3) |

Maximum base score: **37**. Mechanism-level explanations receive credit;
symptom-only observations do not. Finding a real defect outside the answer key
may earn verbal bonus credit but does not count toward the total score.

After time expires, compare with
`reference/7.2c-review-data/ANSWERS.md` and finish the reflection prompts.
