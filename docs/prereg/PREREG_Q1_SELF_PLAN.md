# Pre-registration — Q1 SELF-PLAN
# COMMIT THIS FILE BEFORE THE FIRST GENERATION CALL. If it isn't committed first, it isn't preregistered.

date: 2026-07-29
branch: feat/q1-self-plan          # pin; never switch mid-run
budget_cap: $10
platform: student=OpenRouter, provider-pinned (single stack; claim scoped to student)
          teacher=OpenRouter (plan generation, offline role)
          judge=OpenRouter
          [AMENDED 2026-07-30 — see Deviation log D1 below; original: student=Prime]
student: qwen/qwen3.6-27b  (reasoning.enabled=false — verify in code)
teacher: z-ai/glm-5.2      (max_tokens>=12000, hard exit on empty content;
                            OpenRouter — GLM empty-content behavior calibrated
                            there in FINDINGS_FINANCE §E; no new serving path)
judge:   openai/gpt-5.2    (assert judge != teacher; verify TOTAL-fallback present on branch)

## Question
When the teacher plan produces its gain, is the mechanism plan STRUCTURE
(student can generate it itself, free to serve) or teacher KNOWLEDGE
(teacher load-bearing, cost story)?

## Arms (same questions, paired)
A. NOPLAN-MATCHED — student writes a ~250-word neutral RESTATEMENT of the
   question (no plan, no strategy words), then answers. This is the
   compute-matched baseline (RALPH confound control).
B. SELF-PLAN — student writes its own <=250-word plan, then answers.
   Plan prompt gives NO domain content, only: "write a brief plan: what is
   being asked, what determines the answer, what steps you will take, what
   mistakes to avoid."
C. TEACHER-PLAN — GLM 5.2 writes <=250-word plan from the question only
   (no rubric — firewall), injected before the student answers.

## Sample & power
- Pilot: 8 questions x 3 arms, k=3 gen repeats, 3 judge passes averaged.
  Gate: per-arm sigma consistent with historical noise floor; no empty
  outputs; spend <= $2.5. If pilot fails mechanically -> fix, re-run pilot.
- Full: 40 validation questions (category-stratified draw from the frozen
  80, seed 42), same protocol. Paired deltas, analysis/bootstrap.py CI.

## Primary metric
recovery = (SELF-PLAN − NOPLAN-MATCHED) / (TEACHER-PLAN − NOPLAN-MATCHED)
computed on paired per-question means.

## Pre-committed branch rule (decided now, not after the number)
- TEACHER-PLAN − NOPLAN-MATCHED CI includes 0  -> the +6 does not replicate
  under matched conditions on this stack; STOP, report, re-scope. recovery
  is undefined and must not be quoted.
- recovery >= 2/3  -> mechanism is STRUCTURE. Next: Q2 (platform check).
- recovery <= 1/3  -> teacher KNOWLEDGE load-bearing. Next: plan-caching
  probe on the critical path.
- 1/3 < recovery < 2/3 -> hybrid. Next: plan-caching, self-plan kept as a
  fallback layer.

## Author prediction (calibration, recorded before data)
prediction: recovery <= 1/3        # per user's stated expectation
confidence: 70 %                   # author (Rohan), recorded before data
notes: reviewer estimate ~50%

## Deviation log (append-only; each entry committed BEFORE the affected run)

**D0 — 2026-07-30, answer token cap 2048 -> 8192.** Pilot gate found 55/72 answers
truncated mid-sentence, skewed by arm (A 16/24, B 19/24, C 20/24) — a confound against
the plan-conditioned arms. Per the "pilot fails mechanically -> fix" rule the cap was
raised. User decision: no separate pilot re-run; all questions run uniformly at 8192 in
the full run. Pilot data at 2048 archived (`runs/q1_pilot2048_archive.json`), excluded
from analysis.

**D1 — 2026-07-30, student platform Prime -> OpenRouter.** The Prime team balance hit
insufficient funds mid-run (HTTP 402); all 4 shards fail-fast aborted at 78/360 graded.
User elected to switch platforms rather than refund Prime. Consequences, accepted
knowingly:
- The 78 Prime-graded units and 96 banked Prime answers are **discarded**, not merged —
  mixing serving stacks inside one paired analysis is the confound this project already
  documented (platform flip). Archived to `runs/q1_prime_partial_archive.json`.
- The 13 teacher plans are **retained**: they were generated on OpenRouter/GLM from the
  question text only, so they are platform-independent w.r.t. the student.
- Student is provider-pinned on OpenRouter (`allow_fallbacks=false`, fp8) because 9
  providers serve this slug; without a pin, serving config would drift within the run.
- Cost rises to ~$36 vs the $10 planned cap (gpt-5.2 judge at $14/M output x 3 passes).
  Budget is a fallback, not a stop: run continues, overage recorded here.
- **The claim is now scoped to OpenRouter, not Prime.** Note the prior HINT/TRACE +6
  result was measured on Prime with a different student and benchmark; Q1 was already a
  generalization test, and is now also a different serving stack. If `C - A` CI includes
  0, the pre-committed rule fires (STOP / re-scope) and the honest reading includes
  "the teacher-plan effect was not established on this stack" as a live explanation.

**D2 — 2026-07-30, judge fixed before the OpenRouter run (two scoring bugs).** A
diagnosis of fpb-00103's ~90% grade-failure rate found defects affecting *all*
questions, not just that one:
- *Truncated grades were scored instead of retried.* gpt-5.2 burned ~75% of the
  2048-token judge budget on hidden reasoning (measured: 1527/2048 reasoning tokens,
  `finish_reason='length'`), cutting output off before `TOTAL`; the parser then summed
  whatever R items survived. **38/78 grades in the aborted Prime run (49%) and 22/72 in
  the pilot (31%) came through that path.** Deflation grows with judge-output length,
  which grows with answer length, which differs by arm — a bias against the
  plan-conditioned arms B and C. Fix: reasoning disabled on judge calls, budget
  2048→8192, repair retry escalates, missing `TOTAL` now raises.
- *Wrong denominator.* 38/40 rubrics instruct `TOTAL` on a declared basis (`MAX: 100`)
  while the item ladder sums to 70–92 → 30/40 questions over-scaled 1.06–1.35×. Fix:
  denominator read from the rubric text (deterministic per question, identical across
  arms/reps), explicitly not from the judge's output.

Both fixes precede any OpenRouter data, so the Q1 run is uniform. Consequence for
history: normalized scores in `docs/FINDINGS_FINANCE.md` were produced under the old
denominator and the old truncation path, and are not comparable to Q1's numbers.

## Declared limitations (written now)
- Single stack (OpenRouter, provider-pinned). Any positive is scoped until Q2.
- n=40 resolves ~+/-4-5 points at this protocol; recovery ratio inherits
  wide CI if teacher-plan effect is small.
- Restatement control could itself carry a small effect; accepted, declared.
