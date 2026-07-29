# Q7 — Similarity-Matched Memory: pre-registration

**Recorded 2026-07-28, committed before the first generation call.**
**Operational constraint (CTO, 2026-07-28):** OpenRouter is funded but *no large runs or big
batches for now* — execution proceeds in the brief's own gated steps: 3-sample review (~$0.30)
→ 4+4 pilot (~$4) → full 20+20 run **only after explicit go-ahead**.

## Question

Prior work established that frozen teacher-built memory does not transfer to **novel, diverse**
expert questions (four memory forms, two platforms, pre-registered nulls). Q7 tests the
**production case** instead: does memory help when tomorrow's question is structurally similar
to the failures the memory was built from?

Motivating (cross-stack, weak) evidence — PRBench/DeepSeek, computed 2026-07-28, lexical
TF-IDF cosine: max similarity of any held-out question to any lesson source was 0.14 (the
benchmark is dissimilar-by-construction); most-similar tercile showed +2.7 mean memory delta
vs −3.3 for the least-similar (r=+0.23, n=28, n.s.). **This motivates; it does not evidence
the present stack.**

## Stack (pinned)

| Role | Model | Platform |
|---|---|---|
| Student | `qwen/qwen3.6-27b` (reasoning disabled) | **OpenRouter (single platform; every number labeled)** |
| Teacher (variants only) | `z-ai/glm-5.2`, `max_tokens ≥ 4000` | OpenRouter |
| Judge | `openai/gpt-5.2` (≠ teacher, hard assert; TOTAL-fallback verified present at `correction/judge.py:110`) | OpenRouter |

**Memory store (fixed, not rebuilt):** `runs/finance_memory_good.json` —
10 items (4 Credit, 6 Trading), sha256 `7e07b1a3e912d9ab…`. The independent variable is the
*question*, never the memory.

**Similarity method (canonical, committed as `analysis/similarity.py`):** lexical TF-IDF cosine
over lowercased word tokens (≥3 chars), IDF over the compared document pool. No repo method
predated this; the PRBench 0.14 figure used the same formula inline.

**Lesson-source questions:** reconstructed from `runs/build_mem.log` (the store lacks source
ids — a verified mismatch vs the brief). Reconstruction is asserted in code by matching logged
category + artifact lengths to store items; on assertion failure the fallback V-set is
category-matched (Credit/Trading) train failures, and the report must say which path was used.

## Design (4 cells, fully paired)

| Cell | Questions | Memory |
|---|---|---|
| V+ / V− | teacher-generated **variants** of lesson-source train questions | injected / none |
| U+ / U− | **unrelated** held-out questions (bottom-tercile max-cosine to all sources) | injected / none |

Variant hard rules: blind generation (no memory text in the generation context — asserted);
variant rubric derived from source rubric, must parse under `Item R*(max N)` and preserve the
max-point total; cosine(variant, source) required in **[0.30, 0.90]**, full distribution
reported; unscoreable sources (e.g. `fpb-00108`) excluded up front; 3 source/variant/rubric
samples dumped for human review **before** the full run.

Injection: the existing category-keyed path in `adapters/finance.py`, unmodified. Per-question
injected-items logged; retrieval coverage % reported per cell.

Sample size: 20 variants + 20 unrelated × {mem, no-mem} × 3 generation repeats, 3 judge passes
per answer (≈240 generations, ≈720 judge calls). **Pilot gate first:** 4+4, all cells; if
3-pass-averaged per-question σ has not fallen meaningfully below the historical ±10–20 points,
STOP and report — the full run cannot resolve ±3. Budget ceiling **$25**, tracked per call.

## Decision rule (verbatim from the approved brief)

> **WORKS:** `delta_V ≥ +3` normalized points with 95% paired-bootstrap CI excluding 0, AND
> `delta_U ≤ 0`. → Memory functions as recurrence-prevention. Next step: similarity-gated
> injection (memory fires only above a match threshold; prompt stays clean otherwise),
> composed with the cascade.
>
> **DEAD:** `delta_V` CI includes 0. → Memory is null in both its hard (novel) and easy
> (recurring) settings. The memory line is closed permanently and is not reopened without a
> new mechanism.
>
> **AMBIGUOUS:** `delta_V > 0` but CI includes 0, or `delta_U > 0` too (no crossover) →
> underpowered / leakage suspected. Report as such; do not claim a positive.

Primary: `delta_V = mean(V+) − mean(V−)`, paired bootstrap. Control: `delta_U`. Crossover:
`delta_V − delta_U`. The verdict is computed mechanically by `analysis/q7_report.py`, not
narrated.

## Known deviations from the task brief (declared up front)

1. Work happens on branch `feat/q7-similarity-memory` (branched from `feat/finance-tracelift`,
   never switched mid-run) in an isolated worktree — honors the brief's no-mid-run-switch
   intent under background-session isolation rules.
2. The "same embedding method used for the 0.14 figure" existed in no repo; it is defined by
   `analysis/similarity.py` in this commit and used for all Q7 similarity numbers.
3. Store lacks source-task ids; reconstruction-from-log with declared fallback (above).

---

## PARKED — 2026-07-28, before pilot, ~$0.60 spent

Decision (Rohan): Q7's best-case outcome — frozen memory helps on near-duplicate questions —
is **dominated by plan-caching on the mechanism that already won**: for recurring questions,
cache the teacher's per-task plan (HINT, +6.0 significant on the Prime grid) and re-inject it
on lookalikes. Same recurrence benefit, no distillation, no gate, built on a proven effect.
An experiment whose best case is second place does not justify its construction cost — and the
samples phase confirmed variant generation is the fiddly part (3/3 rejected by the validation
harness for three different reasons: cosine 0.96 too-close; rubric total rewritten 100→10;
rubric format dropped).

State at parking: prereg + build committed; provenance reconstruction verified (10/10 store
items → 5 sources); validation harness working as designed; resumable from
`runs/q7/state.json` if ever revived. No held-out question was touched. The memory line's
standing verdict (null in 4 delivery forms, 2 platforms) is unchanged; its recurrence variant
is *untested-and-parked*, not disproven.

Successor question: the **plan-caching probe** — does a cached teacher plan retain its +6 when
re-used on paraphrased/recurring questions? Uses the winning mechanism directly.
