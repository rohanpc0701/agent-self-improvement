# Q1 SELF-PLAN — results

**Date:** 2026-07-30 · **Prereg:** `docs/prereg/PREREG_Q1_SELF_PLAN.md` (D0–D4, all committed
before the data they affect) · **Regenerates from committed state with no API calls:**
`python3 scripts/q1_analyze.py`

**Label:** OpenRouter (student provider-pinned DeepInfra fp8), matched-baseline, n=40, k=3,
360/360 cells graded, 0 errors outstanding.

Student `qwen/qwen3.6-27b` · teacher `z-ai/glm-5.2` (plans only, offline) · judge
`openai/gpt-5.2` · 40 stratified validation questions, seed 42.

## The question

When a teacher-written plan improves a cheaper model, is the mechanism **plan STRUCTURE**
(which the student can generate itself, free to serve) or **teacher KNOWLEDGE** (teacher
load-bearing, so it stays in the hot path)?

## The ladder

| Arm | Score | vs A |
|---|--:|---|
| **A** NOPLAN-MATCHED (student restates the question, then answers) | 36.81 | baseline |
| **B** SELF-PLAN (student writes its own ≤250w plan) | 40.43 | **+3.62** [−0.23, +7.73], p=0.067 |
| **C** TEACHER-PLAN (GLM writes a ≤250w plan, question-only) | 42.02 | **+5.21** [+0.47, +9.76], p=0.031 |

Point-estimate recovery = 3.62 / 5.21 = **0.695**. Ratio 95% CI = **[−0.00, 1.81]**.

## Verdict under the pre-committed rule: STOP / re-scope

`C − A`'s CI lower bound is **+0.47**, which does not clear the **+1.0** magnitude floor
registered in D3. So **recovery is undefined and must not be quoted**, and the registered
branch is STOP / re-scope.

**This is where the blind pre-registration earns its keep, so state it plainly:** under the
*original* rule (denominator CI merely excludes 0) the gate passes, recovery reads 0.695, and
the printed branch would have been **"STRUCTURE → Q2 platform check"**. The D3 tightening —
written and committed while nobody had seen a single per-arm mean — is what converts a
confident-looking mechanism claim into an honest "not resolved". Both readings are reported
here rather than the flattering one.

The ratio CI is the substantive reason to distrust the point estimate: **[0.00, 1.81] spans
all three pre-registered branches.** At n=40 with a ~5-point teacher effect, the ratio is not
identifiable. A 0.695 point estimate sitting barely above the 2/3 STRUCTURE threshold, with a
CI covering KNOWLEDGE, HYBRID and STRUCTURE alike, is not evidence for a mechanism.

## What *is* established

1. **The teacher-plan effect replicates on this stack.** `C − A = +5.21`, CI excludes 0,
   p=0.031, 4.0% of resamples have an unusable denominator. A ≤250-word question-only plan
   from a stronger model measurably lifts a 27B student on rubric-graded finance reasoning —
   now shown on OpenRouter, not only on Prime. It is **smaller** than the +6 measured on Prime,
   which used a different student and benchmark, so this is generalization, not replication.
2. **Self-planning captures most of the point estimate but is not individually significant.**
   `B − A = +3.62`, CI [−0.23, +7.73], p=0.067 — it touches zero. "The student can plan for
   itself for free" is directionally supported and statistically unproven.
3. **The author's recorded prediction (recovery ≤ 1/3, i.e. teacher KNOWLEDGE load-bearing,
   70% confidence) is not supported.** The point estimate points the other way (0.695,
   structure-leaning). Given the CI, it is also not refuted. Calibration note: the direction
   of the miss is more informative than the width.

## Robustness

Normalization basis is the one measurement question the audit left open (the judge's reporting
basis is unstable within a question). Both candidate bases were computed:

| Basis | A | B | C | B−A | C−A | recovery (unfloored) |
|---|--:|--:|--:|---|---|--:|
| per-unit (primary) | 36.81 | 40.43 | 42.02 | +3.62 [−0.23, +7.73] | +5.21 [+0.47, +9.76] | 0.695 |
| as-stored | 34.57 | 38.28 | 39.21 | +3.72 [+0.01, +7.69] | +4.64 [+0.30, +8.75] | 0.800 |

**Both bases give the same registered branch (STOP).** 93 of 360 units re-normalize under the
per-unit basis; basis reasons: 149 ladder-reconciled, 152 rescaled-to-declared, 59
ladder-fallback. The qualitative picture — teacher plan helps, self-plan close behind, ratio
unidentifiable — is invariant to the choice.

## Honest limits

- **Arm C is compute-matched at answer time only.** A and B each make two student calls; C
  makes one, since its plan comes from the teacher. Every arm's answer call is stateless and
  re-injects the preamble identically, so nothing carries over from the extra forward pass, but
  C's student consumes ~2/3 the student tokens. Never describe C as flatly "compute-matched".
- **The restatement control may carry its own effect.** No arm measures a bare no-preamble
  answer, so if restating helps, both B−A and C−A are attenuated and the design is biased
  toward STOP. Not resolvable from this data.
- **Per-unit basis reconstruction rests on one of three judge passes.** `grade()` averages
  `normalized` across passes but persists only the last pass's raw text; if a unit's basis
  flipped between passes, the mean mixes bases and cannot be unmixed offline.
- **Single stack, single student, single split, n=40.** Power is the binding constraint: the
  registered ±4–5 point resolution is the same order as the effect being divided.
- **13 of 40 arm-C questions** use teacher plans generated before the D1 platform switch. They
  are byte-identical across archives and derived from question text only.

## What this run cost, and what it caught

~$36 of API spend against a $10 planned cap (recorded as overage in D1; budget is a fallback,
not a stop). The measurement bugs found and fixed along the way are the durable output:

- **Judge truncation deflation** (D2) — gpt-5.2 spent ~75% of a 2048-token budget on hidden
  reasoning, and truncated grades were *scored* by summing surviving items. 49% of the aborted
  Prime run's grades came through that path, and deflation scaled with answer length, i.e. with
  arm. Caught before it produced a reportable number.
- **Wrong normalization denominator** (D2, refined in D3) — 30 of 40 questions over-scaled
  1.06–1.35×.
- **Sign-blind decision gate** (D3) — would have printed "KNOWLEDGE load-bearing" from a
  significantly *negative* teacher effect.
- **Teacher over-reasoning to empty content** (D4) — GLM burned 10,924 of 12,000 tokens on
  reasoning for fpb-00134 and returned nothing.

## Next step

The registered branch is STOP / re-scope, so the honest options are to raise power on the
`C − A` estimate (more questions, or a second student where the teacher effect is larger and
the ratio is therefore estimable) or to drop the ratio framing and test the two candidate
mechanisms directly rather than by division. A plan-caching probe is **not** justified by this
data: it was the KNOWLEDGE branch's action, and the KNOWLEDGE branch did not fire.
