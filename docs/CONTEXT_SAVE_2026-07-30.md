# Context save — 2026-07-30 (post-Q1)

Supersedes `docs/CODEX_BRAINSTORM_CONTEXT.md` as the current-state doc. Read this first.

## One paragraph

Frozen/broadcast teacher memory is dead (four delivery forms, three domains, two serving
stacks). Per-task teacher **guidance** survives and is now measured on a second stack. Q1 asked
whether the guidance works because of plan *structure* (student could generate it itself, free
to serve) or teacher *knowledge* (teacher stays in the hot path). The answer: **the mechanism is
unresolved, and the pre-registered rule says so** rather than guessing. The ask forward is one
experiment (~$35, one week) that tests the mechanism directly instead of by division.

## Q1 verdict

**Label: preregistered-powered, OpenRouter (student provider-pinned DeepInfra fp8), n=40.**
360/360 cells, k=3, 0 errors outstanding. Student `qwen/qwen3.6-27b`, teacher `z-ai/glm-5.2`
(plans only, offline role), judge `openai/gpt-5.2`. 40 stratified validation questions, seed 42.

| Arm | Score | vs A | 95% CI | p |
|---|--:|--:|---|--:|
| A NOPLAN-MATCHED (restatement, then answer) | 36.81 | — | — | — |
| B SELF-PLAN (student's own ≤250w plan) | 40.43 | +3.62 | [−0.23, +7.73] | 0.067 |
| C TEACHER-PLAN (glm-5.2, question-only) | 42.02 | **+5.21** | **[+0.47, +9.76]** | **0.031** |

Point recovery = 3.62 / 5.21 = **0.695**. Ratio CI = **[−0.00, 1.81]** — spans all three
pre-registered branches.

**Registered branch: STOP / re-scope.** `C − A`'s CI lower bound (+0.47) fails the +1.0
magnitude floor registered in D3, so recovery is undefined and must not be quoted.

**Established:** the teacher-plan effect replicates on a second stack (+5.21, CI excludes zero).
Smaller than Prime's +6, which used a different student and benchmark — so this is
generalization, not replication.
**Not established:** that self-planning alone works (+3.62 touches zero, p=0.067).
**Author prediction (recovery ≤ 1/3, i.e. KNOWLEDGE, 70% confidence): not supported by
direction** — the estimate leaned structure. Not refuted either, given the CI. The miss
direction is itself calibration data.

**Robustness:** both normalization bases give the same branch; 93/360 units re-normalize under
the per-unit basis.

**Key interpretive point:** STOP is a verdict about the *ratio*, not the mechanism. Per-question
mechanisms genuinely differ — fpb-00297 (Research): A 10.7 with 4 traps → B 89.0 with zero traps
→ C 49.0 (self-plan beat the teacher); fpb-00149 (Investment Banking): A 47.7 → B 49.5 flat →
C 62.5 (only teacher knowledge helped). Averaging opposite mechanisms is what makes the ratio
meaningless.

## Deviations D0–D4 (each committed before the data it affected)

| ID | Change | Why |
|---|---|---|
| **D0** | Answer token cap 2048 → 8192 | Pilot gate: 55/72 answers truncated mid-sentence, skewed by arm (A 16/24, B 19/24, C 20/24) — a confound against the plan-conditioned arms. No separate pilot re-run; all questions run uniformly at 8192. Pilot data archived, excluded. |
| **D1** | Student platform Prime → OpenRouter | Prime team balance hit HTTP 402 mid-run; all 4 shards fail-fast aborted at 78/360. 78 graded units + 96 banked answers **discarded, not merged** (mixing serving stacks is the documented platform-flip confound). 13 teacher plans retained (question-text only, student-independent). Claim scoped to OpenRouter. Cost rose to ~$36 vs the $10 planned cap — budget is a fallback, overage recorded. |
| **D2** | Judge: reasoning disabled, budget 2048 → 8192, missing TOTAL raises; denominator from rubric text | Two scoring bugs, see below. |
| **D3** | One-sided gate + MIN_DENOM=+1.0 floor; ratio CI inside one resampling loop; per-unit basis re-derivation; OpenRouter label | Found by a 13-agent adversarial audit run **while blind to all per-arm means**. |
| **D4** | Teacher plan budget 12000 → 32000 (`TEACHER_MAX_TOKENS`, no code change) | glm-5.2 spent 10,924 of 12,000 tokens on hidden reasoning for fpb-00134 and returned 0 chars. Protocol-neutral: a cap only truncates, so plans that already fit are byte-identical; only previously-impossible ones change. Excluding the question was rejected — it would cut n below 40 and bias toward questions the teacher finds easy to plan. |

## The four measurement bugs (the durable output)

1. **Judge truncation deflation.** gpt-5.2 spent ~75% of a 2048-token judge budget on hidden
   reasoning (`finish_reason='length'`, measured 1527/2048 reasoning tokens). Truncated grades
   were then *scored* by summing whatever R-items survived. **38/78 (49%) of the aborted Prime
   run's grades and 22/72 of the pilot's** came through that path. Deflation scales with judge
   output length → answer length → **arm**. This is the "partial judge parses" defect from
   reasoning-rsi's `AUDIT.md`, live in this repo.
2. **Wrong normalization denominator.** 38/40 rubrics instruct `TOTAL` on a declared basis
   (`MAX: 100`) while the item ladder sums to 70–92 → **30/40 questions over-scaled 1.06–1.35×**.
   Refined in D3: the judge's basis is unstable *within* a question (`TOTAL` disagrees with its
   own item arithmetic in 67/127 units; fpb-00231: 1.5, 1.5, 1.5, then 1.121), so the basis is
   re-derived per unit from persisted raw and reported both ways.
3. **Sign-blind decision gate.** `not (ci_low <= 0 <= ci_high)` admits a CI entirely *below*
   zero, then divides by a negative denominator: verified `C−A = −6`, `B−A = +2` → recovery
   −0.333 → prints "KNOWLEDGE load-bearing" from data showing teacher plans **hurt**.
   **This is the first caught failure mode that was flattering rather than deflating** — the
   eleventh overall.
4. **Teacher over-reasoning to empty content** (D4 above).

## What the audit cleared (so it isn't re-litigated)

A 13-agent adversarial audit found **no arm-biasing defect**. Both candidates died under
measurement: `_word_trim` sits outside the arm branch, cuts on word boundaries, and bites arm A
*hardest* (35/42 at the cap vs B 32/36, C 27/33); and pilot-key aliasing cannot fire because
sharded runs read per-shard files and shard units win the merge (0 of 72 pilot and 0 of 95 Prime
answers reused, all live units timestamped after the D1/D2 commits). Firewall verified: the
rubric never reaches student or teacher, the judge sees no arm label. The draw reproduces
`runs/q1_question_ids.json` bit-exactly from `validation_ids` with zero train/heldout overlap.
D0 truncation gone: 0 of 95 live answers cut mid-sentence.

## Honest limits carried forward

- **Arm C is compute-matched at answer time only** — A and B make two student calls, C makes
  one. Never describe it as flatly "compute-matched".
- **The restatement control may carry its own effect.** No arm measures a bare no-preamble
  answer, so the design is biased *toward* STOP. Q2 adds a bare arm (n=20) to bound this.
- **Per-unit basis reconstruction rests on one of three judge passes** — `grade()` averages
  `normalized` but persists only the last pass's raw text; a basis flip between passes cannot be
  unmixed offline.
- Single stack, single student, single split, n=40. Power is the binding constraint.
- 13 of 40 arm-C questions use teacher plans generated before the D1 platform switch
  (byte-identical across archives, question-text-derived).

## Standing constraints — do not relitigate

- Broadcast/frozen memory: dead, four ways, three domains.
- Generic self-critique: ~0 (+0.77 on Prime). **Directed ≠ broadcast** — do not conflate.
- Validation uplift gates select noise (+7.1 → +1.5 held-out). Held-out is the constraint;
  validation is a filter, never evidence.
- Serving platform changes results at identical model slugs. Scope every claim to its stack.
- Any arm needing the teacher at inference is a **cost story, not a learning story**. Label it.
- Pre-registration is committed before the first generation call, or it is not pre-registration.
- Long runs go on the VM (tmux) — set up before launch, not mid-run.

## Next: Q2-SELF-PLAN (~$35, ~1 week)

Test `B − A` directly; **drop arm C** (its job is done, and removing it also removes the
compute-matching asymmetry from the primary comparison). n=80 (the 40 already run plus the
remaining 40 of the frozen split), k=3, 3 judge passes, paired. Plus a bare arm at n=20 to bound
the restatement-control bias (secondary; changes no branch). Carry forward all D0–D4 fixes.
Record an author prediction with a confidence % before running.

CI half-width on `B − A` was ~4.0 at n=40; expect ~2.8 at n=80 `[design — assumes Q1 variance
holds]`.

**Pre-registered branch rule:** CI excludes 0 → self-plan works at this student scale; next is
held-out confirmation, then the SkillOpt-fork arm. CI includes 0 at n=80 → structure-alone is
dead at 27B; teacher-priced plans (caching) become the product and research moves to
cheap-verification domains.

**Not justified yet:** cached teacher plans (that was the KNOWLEDGE branch's action, and
KNOWLEDGE did not fire).

## Artifacts

| What | Where |
|---|---|
| Q1 results writeup | `docs/RESULTS_Q1_SELF_PLAN.md` |
| Pre-registration + D0–D4 | `docs/prereg/PREREG_Q1_SELF_PLAN.md` |
| Numbers (regenerate, $0) | `python3 scripts/q1_analyze.py` → `runs/q1_summary.json` |
| Harness | `scripts/q1_self_plan.py` (draw/run/pilot-check), sharded via `--shard N --nshards 4` |
| Corrected analysis | `scripts/q1_analyze.py` (kept separate so the harness is never edited mid-run) |
| Cells | `runs/q1_state_shard{0..3}.json` (git-tracked) |
| Archives | `runs/q1_pilot2048_archive.json`, `runs/q1_prime_partial_archive.json` |
| Regression tests | `correction/tests/test_judge_parse_fixes.py`, `test_q1_analyze.py` (49 pass) |
| Friday deck | https://claude.ai/code/artifact/4f25fd7f-12dc-4485-9d8b-c9ad527a436a |

## Open loose end (security)

The DigitalOcean droplet (143.198.72.81) is idle and its SSH private key has been pasted into a
chat transcript. Rotate the key regardless of whether the box is kept for Q2.
