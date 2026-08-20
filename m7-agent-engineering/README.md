# M7 — Agent Engineering

**Module theme**: Building, evaluating, and supervising LLM agents.

The central skill in this module is **not** writing agent code — it is
developing the judgment to know when agent-generated code is correct,
when it is subtly wrong, and how to design verification harnesses that
actually catch the difference.

---

## Phases

| Phase | Title | Status |
|-------|-------|--------|
| 7.1 | Tool-use and agent loops (TBD) | — |
| **7.2** | **Fast PR review** | Ready |
| 7.3 | Writing agent evaluation harnesses (TBD) | — |

---

## Phase 7.2 — Fast PR Review

**Location**: `exercises/7.2-review/`

An AI agent submitted a token sampling and KV cache utility package with
41 passing tests and a professional-looking implementation. Your job: find
all 18 bugs in 45–60 minutes.

**Quick start**:

```bash
cd exercises/7.2-review
cat INSTRUCTIONS.md          # read this first
cat PR_DESCRIPTION.md        # read the "agent's" PR description
cd pr
python3 -m pytest test_pr.py -v   # confirm all 41 pass
# Now open sampling.py, kv_utils.py, metrics.py, bench.py and start reviewing
```

**Key files**:

```
exercises/7.2-review/
├── INSTRUCTIONS.md        — exercise setup, scoring, strategy hints
├── PR_DESCRIPTION.md      — the agent's PR description (read this as a reviewer)
├── review_template.md     — fill this in as you find bugs
└── pr/
    ├── sampling.py        — token sampling (temp, top-k, top-p, rep penalty)
    ├── kv_utils.py        — KV cache management
    ├── metrics.py         — perplexity and throughput metrics
    ├── bench.py           — throughput benchmark
    └── test_pr.py         — 41 tests, all green
```

**Answer key**: `reference/7.2-review/ANSWERS.md` (open only after your review)

---

## Learning objective

After this module you should be able to:

1. Read AI-generated code skeptically, checking semantics not just syntax
2. Identify the 5 categories of bugs AI agents commonly introduce
3. Design test suites that actually fail when the implementation is wrong
4. Articulate *why* green tests do not imply correct code
