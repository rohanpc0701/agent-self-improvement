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

**D3 — 2026-07-30, analysis-layer corrections, committed while blind to the results.** A
13-agent adversarial audit of the measurement and analysis path (run while the data was still
being collected; no per-arm means were computed or seen) cleared the run itself — no
arm-biasing defect, fairness and firewall seams verified — and found four analysis defects.
All are offline arithmetic over persisted `grade.total`/`grade.max`, so they are fixed after
the run, not by re-running units. **Recorded now, before any analysis, so the decision rule
cannot be shaped by the outcome:**

1. *Sign-blind gate on the recovery denominator* (`scripts/q1_self_plan.py:439`). The coded
   gate `not (ci_low <= 0 <= ci_high)` is two-sided, so a teacher effect whose CI lies
   entirely **below** zero passes it; `recovery` then divides by a negative denominator and
   prints a confident branch. Verified: `C−A = −6` (CI −6,−6) with `B−A = +2` yields
   recovery −0.333 → "KNOWLEDGE load-bearing", i.e. the rule would recommend caching teacher
   plans from data showing they hurt. A tiny positive denominator fails the same way
   (`C−A = +0.5`, `B−A = +2` → recovery 4.0 → "STRUCTURE").
   **Correction (a tightening, not a loosening):** the gate becomes one-sided with a stated
   magnitude floor — `C−A` CI strictly above `MIN_DENOM = +1.0` normalized points. Outside
   that, `recovery` is undefined and the STOP/re-scope branch fires. This matches the
   registered intent, which defines recovery as a fraction *of a gain* and states the stop
   condition as "the +6 does not replicate".
2. *Recovery reported without uncertainty* (`:441`). The pre-committed point-estimate branch
   is kept, but a paired bootstrap CI on the ratio and the fraction of resamples with
   `C−A ≤ 0` are published beside it, and every branch the CI spans is named. On the archived
   pilot the point estimate was 1.05 with a ratio CI of 0.61–4.06 — two branches wide.
   Implementation note: `analysis/bootstrap.py` sorts its resample array in place, so zipping
   two calls pairs order statistics rather than co-indexed resamples (measured ~2.4× too
   narrow); the ratio must be formed inside one resampling loop.
3. *Normalization basis — D2's fix was incomplete.* D2 read the denominator from the rubric
   text, assuming one basis per question. The judge's actual basis is **unstable within a
   question**: `TOTAL` disagrees with its own item arithmetic in 67 of 127 units, sometimes by
   exactly the rubric's rescale factor and sometimes not (fpb-00231: 1.5, 1.5, 1.5, then
   1.121). Four rubrics carry a rescale instruction with no echo line and are therefore
   inflated. **Correction:** re-derive the basis per unit from that unit's persisted `raw`
   and re-normalize offline; report the primary metric's sensitivity to both candidate bases
   and say so explicitly if the branch flips between them. This defect is **arm-symmetric**
   (basis-flip rate A 22/45, B 23/43, C 22/39; answer lengths matched A 1873 / B 1871 / C 1833
   words), so it attenuates and re-weights deltas without favouring an arm.
4. *Wrong stack in the artifact* (`:432`). The summary label hardcodes "Prime"; D1 moved the
   student to OpenRouter and scoped the claim to it. Corrected to OpenRouter, provider-pinned.

Deferred to the next run (not changed mid-run, since altering the harness now would create the
mixed-instrumentation confound D1 exists to prevent): `_word_trim` collapses newlines when the
cap bites (layout only — content, ordering and headings survive; truncation actually bites arm
A hardest, 35/42 vs B 32/36 vs C 27/33, and always on a word boundary); and unit keys carry no
config stamp, so a future mid-run deviation could silently mix configurations.

**Honest limits this audit surfaced and could not repair:**
- `grade()` averages `normalized` across 3 judge passes but persists only the last pass's raw
  text. Stored total differs from last-pass TOTAL in 111/126 units, so the passes genuinely
  disagree; if a unit's basis flipped between passes, its mean mixes two bases and no offline
  re-normalization can unmix it. Per-unit basis reconstruction therefore rests on one of three
  passes, and the report must say so.
- **Arm C is compute-matched at answer time only.** A and B each make two student calls; C
  makes one, because its preamble comes from the teacher. The answer call is stateless and
  re-injects the preamble identically in every arm, so nothing carries over from the extra
  forward pass — but C's student consumes ~2/3 the student tokens. This must be described as
  "compute-matched at answer time", never as "compute-matched".
- Whether the judge's basis choice correlates with question difficulty or answer style (rather
  than being random) is not falsifiable from partial data; a full-run recheck is free and will
  be run.
- 13 of 40 arm-C questions are instrumented by teacher plans generated before the D1 platform
  switch. They are byte-identical across archives and derived from question text only, so the
  teacher path never touched the student stack — but a skeptic may ask for regeneration.

## Declared limitations (written now)
- Single stack (OpenRouter, provider-pinned). Any positive is scoped until Q2.
- n=40 resolves ~+/-4-5 points at this protocol; recovery ratio inherits
  wide CI if teacher-plan effect is small.
- Restatement control could itself carry a small effect; accepted, declared.
