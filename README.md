# Teacher → student reasoning, measured

[![CI](https://github.com/rohanpc0701/agent-self-improvement/actions/workflows/ci.yml/badge.svg)](https://github.com/rohanpc0701/agent-self-improvement/actions/workflows/ci.yml)

A research harness asking whether a **stronger teacher model** can improve a **cheaper student
model** on rubric-graded expert finance reasoning — **without fine-tuning**.

Honest nulls over optimistic claims. Every headline below is reproducible from committed data
with no API calls.

> **History:** this repo began as a hackathon demo (drift detection over a text-to-SQL agent on
> Spider). That project has been removed. Nothing about Spider, drift detection, or the old
> four-stage loop is current.

---

## What we found

**Transferring the teacher's general knowledge fails. Transferring its per-task guidance works.**

| Intervention | Result |
|---|---|
| Frozen general lessons (playbooks, traps, skeletons) | **Null** across 4 delivery forms, 3 domains, 2 serving stacks. Best case +1.39, n.s. |
| Per-task teacher plan (≤250 words) | **+6.0 / +6.1** (Prime), **+5.21** (OpenRouter) — significant on both |

Four measured reasons the memory approach fails:

1. **Text cannot fix arithmetic.** Frozen memory recovered **−0.1** points of Financial Accuracy
   (predicted in advance); a per-task plan recovered **+3.5**, because it names which quantities
   to compute.
2. **Unconditional injection is a wash by construction.** It helps weak drafts and hurts strong
   ones — r(Δ, baseline score) = −0.44, independently reproduced at −0.34 on a second benchmark.
3. **Validation-uplift gating selects noise:** +7.1 on validation → +1.5 held-out.
4. **The student's failure is indexing, not ignorance.** 75% of its lost points are *omissions*.
   The lessons were correct and in context; it couldn't connect a general principle to the case
   in front of it. That connective step is what the teacher's per-task plan performs.

**Q1 (latest, n=40 × k=3, 360 cells, OpenRouter provider-pinned):**

| Arm | Score | vs baseline |
|---|--:|---|
| A — student restates the question, then answers | 36.81 | — |
| B — student writes its own ≤250w plan | 40.43 | +3.62 [−0.23, +7.73], p=0.067 |
| C — teacher writes a ≤250w plan | 42.02 | **+5.21 [+0.47, +9.76], p=0.031** |

The pre-registered branch rule returned **STOP / re-scope**: recovery = 0.695 but its confidence
interval is [0.00, 1.81], spanning every possible conclusion. Under the *original* rule the same
data would have printed "STRUCTURE confirmed" — a tightening committed while blind to all
per-arm means refused it.

Full result: [`docs/RESULTS_Q1_SELF_PLAN.md`](docs/RESULTS_Q1_SELF_PLAN.md).

---

## Measurement discipline

The rig is the durable asset. Bugs it has caught, each of which silently corrupted scores:

- **Reasoning models return empty content** when `max_tokens` caps reasoning and content
  together — a judge spent 1527/2048 tokens reasoning and emitted nothing; a teacher spent
  10,924/12,000.
- **Truncated judge output was scored, not retried** — the parser summed surviving rubric items,
  deflating scores in proportion to answer length, i.e. correlated with the arm under test. This
  affected 49% of one run's grades.
- **Normalization denominators disagree** — rubrics declare a scaled basis the item ladder
  doesn't match, and the judge's basis can vary *within* a question.
- **Decision rules can be sign-blind** — a two-sided "CI excludes 0" gate accepts a significantly
  *negative* effect and divides by it.

Standing rules: ≥3 repeats averaged before believing any delta (temp-0 noise is ±10–20 points);
pre-registration committed before the first generation call; held-out is the constraint and
validation is only a filter; every claim scoped to its serving stack.

---

## Quickstart

```bash
pip install -e .          # Python ≥ 3.10
python3 -m pytest -q      # 202 tests, hermetic, no API key needed
```

Reproduce the latest result from committed cells — no API calls, no credentials:

```bash
python3 scripts/q1_analyze.py     # -> runs/q1_summary.json
```

Run an experiment (needs `.env` with `OPENROUTER_API_KEY`; the entrypoint asserts model slugs,
base URLs and judge≠teacher before spending anything):

```bash
python3 scripts/q1_self_plan.py draw                          # freeze the question set, commit it
python3 scripts/q1_self_plan.py run --shard 0 --nshards 4     # one shard of four
```

---

## Layout

```
adapters/finance.py    Dataset, splits, student prompt, teacher repair/distill
correction/judge.py    Weighted-criteria rubric judge  ← read the comments before editing
correction/provider.py Teacher/judge client resolution (OpenRouter / Prime / MiniMax)
harness/agent.py       Student hot path: retries, provider pinning, fallback assertions
analysis/bootstrap.py  Paired bootstrap (note: sorts its resample array in place)
scripts/               Experiment harnesses and analysis
contracts/, fixtures/  Shared record shapes, dataset cache, frozen manifests
docs/prereg/           Pre-registrations and deviation logs
```

## Where to start reading

| Doc | Contents |
|---|---|
| [`CLAUDE.md`](CLAUDE.md) | Orientation for AI assistants — constraints and known failure modes |
| [`docs/CONTEXT_SAVE_2026-07-30.md`](docs/CONTEXT_SAVE_2026-07-30.md) | Current state in full |
| [`docs/prereg/PREREG_Q1_SELF_PLAN.md`](docs/prereg/PREREG_Q1_SELF_PLAN.md) | Pre-registration + deviations D0–D4 |
| [`docs/STRATEGY_LADDER.md`](docs/STRATEGY_LADDER.md) | Pre-registered escalation if the current bet nulls |

Benchmarks: [PRBench](https://arxiv.org/abs/2511.11562) (Scale AI) and FinancePro-Bench
(`Sanscritic/finance-pro-bench`, CC-BY-4.0).

## Author

[Rohan Chavan](https://github.com/rohanpc0701)
