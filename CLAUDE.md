# CLAUDE.md — teacher→student reasoning research

> Orientation for AI assistants working in this repo. `AGENTS.md` points here; keep one source
> of truth. **Current-state detail lives in `docs/CONTEXT_SAVE_2026-07-30.md` — read that next.**

## What this repo is

A research harness measuring whether a **stronger teacher model** can improve a **cheaper
student model** on rubric-graded expert finance reasoning, **without fine-tuning**.

It began (July 2026) as a hackathon demo: drift detection over a text-to-SQL agent on Spider.
**That project is gone** — no Spider, no `orchestrator.py`, no viewer, no drift loop. If you find
a doc describing those, it is stale; delete it rather than following it.

## The one-line result so far

Transferring the teacher's **general knowledge** fails. Transferring its **per-task guidance**
works. Frozen memory is null across 4 delivery forms, 3 domains and 2 serving stacks; a
≤250-word per-task teacher plan is worth roughly +5 to +6 points and is significant on two
stacks. The open question is whether the student can produce that plan itself.

## Live code

| Path | What it is |
|---|---|
| `adapters/finance.py` | FinancePro-Bench: dataset, splits, student prompt, teacher repair/distill |
| `correction/judge.py` | Weighted-criteria rubric judge (`grade`, `parse_judge_output`) — **read the comments before touching the parser; two silent scoring bugs lived here** |
| `correction/provider.py` | Teacher/judge client resolution (OpenRouter / Prime / MiniMax) |
| `harness/agent.py` | Student hot path: `_chat_with_retry`, OpenRouter provider pinning, fallback assertions |
| `analysis/bootstrap.py` | Paired bootstrap. Note: it **sorts its resample array in place** |
| `scripts/q1_self_plan.py` | Q1 harness — draw / run / pilot-check, sharded via `--shard N --nshards 4` |
| `scripts/q1_analyze.py` | Corrected analysis, deliberately separate so the harness is never edited mid-run |
| `contracts/schemas.py`, `fixtures/` | Shared record shapes, dataset cache, manifests |

`detector/` is empty and `harness/{feed,evaluator}.py` are hackathon leftovers — not used by any
current experiment.

## Non-negotiable methodology

These were each learned by getting them wrong. Do not relitigate them.

- **≥3 repeats, averaged, before believing any delta.** Temp-0 nondeterminism is ±10–20
  normalized points per question. A single-pass +5.6 already evaporated to +0.0 under k=3.
- **Pre-registration is committed before the first generation call**, or it is not
  pre-registration. Deviations get an appended, committed entry (see `docs/prereg/`).
- **Held-out is the constraint; validation is a filter, never evidence.** A validation uplift
  gate selected noise once already: +7.1 → +1.5 held-out.
- **Scope every claim to its serving stack.** Identical model slugs on different platforms
  produce different winning interventions. Provider-pin the student (`allow_fallbacks=false`).
- **Any arm needing the teacher at inference is a cost story, not a learning story.** Label it.
- **No fine-tuning** without an explicit CTO decision.
- **Never edit the harness mid-run.** Mixed instrumentation invalidates the comparison; put
  analysis changes in a separate script.
- Budgets are **fallbacks** — warn and continue, hard-stop only on a 4× runaway.
- Long runs go on the VM in tmux, set up **before** launch.

## Failure modes this codebase has actually produced

Check for these before trusting a number:

1. **Reasoning models silently return empty content.** `max_tokens` caps reasoning *and*
   content together. Measured: a judge spent 1527/2048 tokens reasoning and emitted nothing; a
   teacher spent 10,924/12,000. Budget generously and fail loudly on empty.
2. **Truncated judge output was scored, not retried** — the parser summed surviving items,
   deflating scores in proportion to answer length, i.e. correlated with arm.
3. **Normalization denominators disagree.** Rubrics declare a scaled basis the item ladder
   doesn't match, and the judge's basis can vary *within* a question.
4. **Decision rules can be sign-blind.** A two-sided "CI excludes 0" gate accepts a
   significantly *negative* effect and divides by it.

## Running things

Everything imports from repo root. Credentials in `.env` (gitignored) — load explicitly in the
entrypoint and assert the model slugs and base URLs before spending anything.

```bash
python3 scripts/q1_analyze.py                          # regenerate all Q1 numbers, no API calls
python3 -m pytest correction/tests -q                  # 49 tests
python3 scripts/q1_self_plan.py run --shard 0 --nshards 4
```

## Where the findings live

| Doc | Contents |
|---|---|
| `docs/CONTEXT_SAVE_2026-07-30.md` | **Start here** — current state, Q1 verdict, deviations, limits |
| `docs/RESULTS_Q1_SELF_PLAN.md` | Q1 result in full |
| `docs/prereg/PREREG_Q1_SELF_PLAN.md` | Pre-registration + deviation log D0–D4 |
| `docs/FINDINGS_FINANCE.md` | FinancePro history, including the k=3 reversal (§H) |
| `docs/STRATEGY_LADDER.md` | Pre-registered escalation if the current bet nulls |
| `~/Desktop/Projects/reasoning-rsi/docs/DECISION_MEMO.md` | Where the memory question was settled |
