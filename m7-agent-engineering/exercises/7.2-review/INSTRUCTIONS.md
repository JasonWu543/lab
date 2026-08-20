# Exercise 7.2 — Fast PR Review

**Estimated time**: 45–60 minutes  
**Goal**: Find all 18 bugs in a "passing" PR before the timer runs out.

---

## Setup

```bash
cd exercises/7.2-review/pr
python3 -m pytest test_pr.py -v   # should be 41 passed, 0 failed
```

Confirm you see a green run. You are now looking at a PR that the AI agent
certified as correct.

---

## Your task

Read the four source files in `pr/`:

| File | What it implements |
|------|--------------------|
| `sampling.py` | Temperature sampling, top-k, top-p, repetition penalty |
| `kv_utils.py` | KV cache trimming, memory estimation, concat |
| `metrics.py` | Perplexity, throughput tracking |
| `bench.py` | Throughput benchmark (eager vs. compiled) |

Find **18 bugs** across the codebase. For each one, fill in a row of the
review template (`review_template.md`).

---

## Bug categories to hunt for

There are exactly **18 bugs** distributed across these categories:

| Category | Count |
|----------|-------|
| Numerical / Correctness | 6 |
| Test bugs | 4 |
| Benchmark fairness | 2 |
| Performance | 2 |
| Design / Robustness | 4 |

---

## Scoring rubric

**Finding a bug** = naming the file, approximate line, and correctly
describing what is wrong. You don't need the exact fix.

| Score | Meaning |
|-------|---------|
| 18/18 | Elite reviewer — ready to supervise AI agents |
| 14–17 | Strong — missed some subtle ones |
| 10–13 | Solid — found the obvious ones, missed design/test bugs |
| < 10 | Need more practice reading adversarially |

Bonus points:
- Identify which bugs the tests *could* have caught but don't (hint: several)
- Write the 2–3 tests that would catch the most P0 bugs

---

## Strategy hints (read only if stuck after 20 min)

<details>
<summary>Hint 1 — where to look first</summary>

Read `test_pr.py` carefully *as code*, not just to see what passes. Ask: what
is each assertion actually checking? Is it possible for the assertion to fail?
</details>

<details>
<summary>Hint 2 — numerical bugs</summary>

For each formula, mentally run it with a simple input and compute the expected
output by hand. Compare to what the code produces.
</details>

<details>
<summary>Hint 3 — performance bugs</summary>

Look for loops. Ask: does anything inside this loop do work that could be done
once outside it?
</details>

<details>
<summary>Hint 4 — design bugs</summary>

Classic Python gotchas: mutable defaults, bare `except`, docstrings that make
promises the code doesn't keep.
</details>

---

## After the exercise

1. Check your findings against `reference/7.2-review/ANSWERS.md`.
2. For every bug you missed: read the mechanism section and write a
   one-sentence explanation *in your own words* of why it's wrong.
3. Write the missing tests (the ones that would expose the P0 correctness bugs).
4. Fill in the POSTMORTEM in `docs/7.2/POSTMORTEM.md` (template provided in
   the exercise docs folder).

---

## Key lesson

Tests passing does not mean code is correct. An AI agent can produce
professional-looking code with plausible docstrings, sensible variable names,
and a green test suite — and still have 18 bugs. Your job as a reviewer is
not to admire the aesthetics but to verify the semantics.
