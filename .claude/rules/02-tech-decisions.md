# LOCKED RULE: Technical decisions

**Do not change without agreement.** These were settled by measurement; re-litigating wastes runs.

## No fine-tuning
In-context memory only (CTO constraint, 2026-07-20). No LoRA, no SFT, no RL. The student's weights
are never touched; the intervention is text in the prompt. Lifting this is a CTO decision, taken
only after the in-context rungs null — see `docs/STRATEGY_LADDER.md`.

## Three disjoint models: student ≠ teacher ≠ judge
A judge that shares a family with the teacher scores its own dialect (self-preference). Enforced by
a hard assert in `correction/prbench_judge.grade`; keep it. All roles run on **OpenRouter** so the
same slug means the same route across arms.

## Nothing single-pass is trusted — k ≥ 3
Temp-0 output on this stack is **not** deterministic: per-question spread is ±10–20 normalized
points (student sampling + judge variance). Every reported delta is a per-task mean over k ≥ 3
repeats. This rule exists because a single-pass `+5.6` on FinancePro averaged to `+0.0` at k=3
(`docs/FINDINGS_FINANCE.md` §H) — the variance protocol caught a false positive that would have
been published.

Corollary: deltas below roughly ±5 points at n≈12 are inside noise. Report the CI or don't report.

## Deltas are paired, and bootstrapped
Δ is computed **per task** (`MEM(t) − PLAIN(t)`), then averaged — never as a difference of arm
means over possibly-different task sets. Use `analysis/bootstrap.paired_bootstrap` for the CI and
p-value, plus the per-task win/tie/loss count as a distribution-free companion.

## Rubric-graded free text, normalized 0–100
`score = clamp(Σ applied weights, 0..max) / max × 100`, where `max` = sum of positive weights.
Positive criterion satisfied adds its weight; a detrimental criterion *committed* subtracts. The
judge emits one `C<n>: yes|no` line per criterion — no prose, no partial credit, temp 0.
Judge reliability was gated before use (test–retest MAD ≤ 5; `docs/FINDINGS_FINANCE.md` §B).

## Memory is frozen, category-keyed, and capped
Built from train-split failures only, then frozen (rules/00). Retrieval is **category match**, no
embeddings. Injection is capped (`select_category_memory`: 1 playbook + ≤2 traps + ≤1 skeleton,
≤4 items) so context stays small and comparable across arms. Know the cap before you claim how
many lessons a run injected — see CLAUDE.md "Known gaps".

## Uplift gating is the mechanism, not a nice-to-have
A lesson earns its place by measurably helping the frozen student on **validation** (`u > threshold`),
not by looking wise. `correction/tracelift.py` is the gate; `scripts/finance_tracelift.py` has the
judge-scored variant. Every published number must state whether it was gated.

## Provider pinning
The student slug is pinned to one OpenRouter provider (`OPENROUTER_PIN_MODEL`,
`OPENROUTER_PROVIDER_ORDER`, `allow_fallbacks=false`, re-applied on every retry). A permitted backup
provider warns loudly and bumps `provider_fallback_count()`; anything outside the allow-list aborts
the run. **Report the fallback count with every result** — a fallback means part of the data came
from a different serving config.

## Benchmarks are frozen before anything runs
Split manifests are seeded (42), committed under `fixtures/`, and pin the dataset SHA-256.
`--check` verifies splits are disjoint and the dataset has not moved. Held-out is touched once per
arm; validation is for gating only.
