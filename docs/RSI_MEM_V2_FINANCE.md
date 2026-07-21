# RSI-Mem v2 — Runtime Self-Improvement on FinancePro-Bench

**Version 2.1 · adopted 2026-07-20.** Supersedes `docs/RSI_MEM_PLAN.md` (v1, coding)
as the active track. v1 machinery carries over (per-item uplift audit, multi-seed
eval runner + paired bootstrap, variance protocol). Dataset verified on HF:
400 rows, fields {id, category, question, rubric}, rubrics contain R/T/B items,
CC-BY-4.0.

**Target benchmark:** [Sanscritic/finance-pro-bench](https://huggingface.co/datasets/Sanscritic/finance-pro-bench) — 400 expert-level finance questions, 33 categories, free-text answers graded by point rubrics (tiered items R1–Rn, trap penalties T1–Tn, insight bonuses B1–Bn). All information is embedded in the question; pure reasoning, no retrieval.

**Incorporates:** the nine-paper reference (Memp, TraceLift, RA-RFT, SelfMem, OPHSD, SIA, MEMO, MRAgent, survey) + the Intelligent Internet harness study (*From RALPH to Zenith*, May 2026), which contributes the compute-matched retry control, the memory-write stopping rule, and the persistent-skill-library harness phase.

> **North star (upgraded):** `student + frozen memory > compute-matched student retries` on held-out questions, measured in normalized rubric score, with memory built only from training-stream failures. Beating "student alone" is necessary but no longer sufficient — memory must beat raw test-time compute spent on fresh attempts (RALPH finding: fresh gap-finding sessions improve scores with no memory at all).

## 0. What changes vs. the coding plan

| Dimension | Coding (v1) | FinancePro-Bench (v2) |
|---|---|---|
| Verification | Binary unit tests | Rubric-following LLM judge → continuous score 0–100 (normalized) |
| Uplift signal | Pass-rate delta | Score delta — denser, K can be smaller |
| Task shape | Single-shot generation | Single-shot but deep multi-step reasoning (NOT interactive long-horizon) |
| Memory target | Repaired code few-shots + rules | **Category playbooks + trap-avoidance rules** + compressed worked exemplars |
| New diagnostic | — | **Trap-hit rate**: which named trap penalties (T1–Tn) the student triggers, per category |
| Data budget | Large problem pools | **400 questions total** — splits are the binding constraint |
| Teacher ceiling | ~0.93 unaided | ~66% (frontier) — teacher is NOT near-perfect; repairs are rubric-guided |
| New risk | — | Judge noise; judge-gaming ("rubric-speak" without better reasoning) |

**Honest framing note:** this benchmark is *deep* reasoning, not *long-horizon interactive* reasoning. If the research claim you want is about long-horizon agents (ALFWorld-class), keep that as a separate phase (now Phase 4b, rebuilt on the RALPH→Zenith harness). What this benchmark uniquely tests: whether teacher-built memory can transfer **domain reasoning procedure** (framework-application patterns, trap avoidance) — closer to Memp's "scripts" than anything the coding domain offered.

## 1. Data design (the binding constraint)

400 questions, 33 categories, category sizes 3–39.

```
Split (category-stratified, frozen before anything else runs):
  TRAIN-STREAM : 200  — student runs, fails; teacher repairs; memory is built here
  VALIDATION   :  80  — uplift gating + pipeline decisions ONLY
  HELD-OUT     : 120  — touched exactly once per evaluated arm; never for selection
```

- Stratify by category so every split covers most domains; document the seed.
- **Rubric access policy:** the teacher may see rubrics for TRAIN-STREAM questions only (to guide repairs). Rubrics for VALIDATION are seen only by the judge. HELD-OUT rubrics: judge-only, ever. Memory items must never contain text traceable to a specific held-out question.
- **Leakage rule:** memory items are per-category principles/playbooks, not per-question answers. Audit: no named entities from any benchmark question may appear in memory.

## 2. Verification & judging protocol

- **Judge:** one fixed strong model, temperature 0, given the official rubric verbatim, required output format (`R<n>: pts — evidence quote; trap T<n>: −pts; TOTAL`).
- **Normalize:** `score = TOTAL / MAX × 100` per question.
- **Judge reliability check (Phase 0):** grade a 40-question sample twice (fresh contexts) + hand-audit 15. Report test–retest correlation and mean absolute difference. If MAD > ~5 normalized points, average 2 judge passes for all experiments.
- **Judge-gaming control:** memory that teaches "cite ASC codes, use rubric-style headers" may inflate scores stylistically. Control: a *style-only placebo memory* arm (generic "cite standards, show steps" rules written without any teacher repair). Real memory must beat the placebo, not just beat nothing.
- **Blindness:** the judge never sees which arm produced an answer.
- **Judge ≠ teacher:** the judge model must differ from the teacher model (self-preference bias). Enforced by a hard assert in code.

### Evaluation arms (fixed roster)

| Arm | What it is | What it controls for |
|---|---|---|
| A1 Student alone | Single attempt, no memory | Floor |
| A2 Student + compute-matched retries | No memory; same total token budget as A4 spent on fresh gap-finding attempts (RALPH-style: reopen question, critique own draft, retry; best attempt judged) | **Is memory better than raw test-time compute?** |
| A3 Student + placebo memory | Style-only rules ("cite standards, show steps"), no teacher repair | Judge-gaming / rubric-speak |
| A4 Student + real memory | The frozen, uplift-gated store | The claim |
| A5 Unaided teacher | Single attempt | Ceiling |

Token accounting for A2: log A4's mean total tokens per question (memory injection + generation); give A2 the same budget as extra attempts. Report both per-attempt-matched and per-token-matched variants if they diverge.

### Metrics
- **Primary:** `GAP_retry = mean score(A4) − mean score(A2)` on HELD-OUT, ≥3 seeds, paired bootstrap over questions. (`GAP_alone = A4 − A1` reported as secondary — necessary but not sufficient.)
- **Success bar (proposal):** GAP_retry ≥ +4 normalized points, p < 0.05, AND A4 > A3 (placebo).
- **Secondary:** GAP_alone; trap-penalty rate (per category, per named trap); rubric-item profile (which R-items improve); insight-bonus rate; % of teacher–student gap closed; memory tokens per query.

## 3. Memory design (adapted to rubric structure)

Three artifact types, all teacher-built from train-stream failures:

1. **Category playbooks (rules/scripts — Memp's "script" finding says these generalize).** Per-category analysis checklists, e.g. "VIE analysis: (1) equity-sufficiency gate before voting model, (2) kick-out rights: check single-party exercisability, (3) decision-maker fee: aggregate ALL other interests before safe harbor."
2. **Trap registry.** Named failure modes with trigger conditions, harvested from trap penalties the student actually hit: "TRAP: applying fee safe-harbor without aggregation → always list the manager's other economic interests first." Highest-value artifact this benchmark enables — traps are explicit, named, recurring.
3. **Compressed exemplars (few-shots).** Questions are 3–10k chars and expert answers are long — raw few-shots blow the context and dilute (SelfMem). Store *skeletonized* worked solutions: issue → framework → 5-step reasoning → conclusion, ≤300 tokens each.

Write path: teacher repairs a failed answer against the rubric → distills into playbook deltas / trap entries / one skeleton → **uplift gate** → compaction/merge → frozen store.

**Memory-write stopping rule (RALPH lesson — no principled stop = unbounded cost).** RALPH stops only by budget cutoff because another pass can always "still change something"; the memory-build loop has the same hole. Rule: track the mean measured uplift of *newly admitted* items over a sliding window of the last W=15 candidate writes; **freeze the store when window-mean uplift < +0.5 normalized pts or the gate's admission rate < 20%** for a full window. Log the stopping iteration — the saturation curve is itself a reportable result (expect RALPH-shaped diminishing returns: fast early gains, oscillating tail).

## 4. Phase plan (20 weeks)

### Phase 0 — Splits, judge harness, honest baselines (wk 1–2)
- G0.1 Freeze the 200/80/120 split (stratified, seeded, published).
- G0.2 Build the judge harness + reliability check (§2). **Exit blocker:** proceed only if judge test–retest MAD ≤ 5 pts (or doubled passes fix it).
- G0.3 Baselines with ≥3 seeds: student alone · unaided teacher · (existing memory if ported). Also record per-category student scores.
- G0.4 **Choose student bands by headroom, not dogma:** if a model scores <10 normalized on held-out, it has no sensitivity to measure lift. Pick the smallest student scoring roughly 15–40. (Frontier ≈ 66; a 3B may simply floor.)

> **Exit:** trusted judge + baseline table with CIs + chosen student band(s).

### Phase 1 — Diagnose (wk 2–4)
- G1.1 **Trap census:** on train-stream failures, tabulate which trap penalties fire per category. Hypothesis: a small set of traps accounts for most lost points.
- G1.2 **Rubric-item profile:** are points lost on calculations, framework selection, or synthesis? Different memory helps each differently.
- G1.3 **Uplift audit** of any existing/ported memory: `u = mean score(with item) − mean score(without)`, K=2–3, on VALIDATION.
- G1.4 **Procedural-vs-informational probe (OPHSD):** does the teacher's advantage come from reasoning procedure (checklists — promptable) or finance knowledge the small model lacks (ASC 810 content — likely not promptable)? Test by comparing uplift on quote-included vs quote-free questions.

> **Decision gate:** failure taxonomy across {knowledge gap, procedure gap, trap susceptibility, calculation error} with % attribution → sets Phase 2 emphasis.

### Phase 2 — Best-possible ICL memory (wk 4–9)
- G2.1 **Uplift-gated writes** (u > +1 normalized pt on validation; continuous threshold clears judge noise).
- G2.2 **Compaction (SelfMem):** hard cap ~4 items injected per query: 1 category playbook + ≤2 trap entries + ≤1 skeleton exemplar. Merge over append.
- G2.3 **Retrieval = category + trap-pattern match (RA-RFT adapted):** primary key is the question's category; secondary, teacher-labeled "which traps is this question likely to spring." No embedding similarity.
- G2.4 **Apply the memory-write stopping rule** (§3); report the saturation curve.
- G2.5 Full held-out evaluation on all five arms (A1–A5), including **A2 compute-matched retries** — the arm separating "memory works" from "more inference works."

> **Exit:** SUCCESS = GAP_retry ≥ +4 AND A4 > A3 → write-up track. If A4 > A1 but A4 ≤ A2 → memory is a worse way to spend compute at this scale → Phases 3/5. NULL everywhere with clean pipeline → capacity/knowledge floor → Phases 3/5.

### Phase 3 — Capacity bands (wk 9–12)
- Same pipeline at 2–3 student scales chosen for headroom (7–8B, 14B, 32B if budget allows).
- Deliverable: GAP-vs-scale curve + trap-rate-vs-scale curve (prediction: trap avoidance transfers at smaller scale than calculation accuracy).

### Phase 4 — Cross-category generalization (wk 12–15)
- Build memory from train-stream questions in ~22 categories; evaluate held-out questions from the ~11 categories never seen in training.
- Tests whether playbooks encode *transferable reasoning discipline* vs. category-local knowledge. Either answer is a finding.

### Phase 4b (optional) — Persistent skill library on a long-running harness (wk 12–16, parallel)
*The long-horizon claim, rebuilt on the RALPH→Zenith study instead of ALFWorld.* Zenith synthesizes reusable "skills" within a run and reads them in later sessions — but skills die with the run, are never uplift-verified, never transfer across tasks. That's the RSI-Mem mechanism, missing.

- **Design:** run a Milestone-RALPH / Zenith-style harness on task A → teacher repairs failure traces → distilled skills pass the **uplift gate** (validation subtasks) → written to a **persistent skill library** → harness runs task B with the library mounted.
- **Delivery is free:** Zenith workers already read skill files — no new plumbing; the intervention is what's *in* the directory and how it got there.
- **Comparison arms:** ephemeral (within-run) skills · persistent ungated library · persistent **uplift-gated** library. Metrics: task score, cost per task (Zenith won rank at 43% of RALPH's cost), skill-reuse counts.
- **Claim if positive:** cross-task, uplift-gated skill persistence beats within-run skill synthesis — the study's own open gap.

### Phase 5 — Prompting vs LoRA (wk 15–18)
- OPHSD-style: generate rollouts with memory attached, distill (LoRA) into the bare student, evaluate memory-removed; test reattachment.
- Extra arm: LoRA on teacher-repaired full answers (SFT) vs on-policy distillation — the knowledge-gap hypothesis predicts SFT injects finance content ICL cannot.

### Phase 6 — Write-up (wk 18–20)
- Positive: *"Trap-Registry Memory: rubric-grounded runtime self-improvement."*
- Negative: *"Procedure is promptable, knowledge is not: limits of ICL self-improvement in expert domains."*
- Release: splits, judge harness, memory store, all arms' graded outputs.

## 5. Kill criteria, risks, predictions

### Kill criteria
- **K1:** Judge test–retest MAD > 5 pts even with doubled passes → benchmark unusable for small-effect measurement; renegotiate metric (trap-rate only) or switch benchmark.
- **K2:** After Phase 2, real memory ≤ placebo → the "lift" is rubric-style, not reasoning; pivot to LoRA (Phase 5).
- **K2′:** Real memory > student alone but ≤ compute-matched retries (A4 ≤ A2) at every scale → memory is an inferior way to spend inference for this task shape; honest paper is "retries beat memory below scale X," pivot to Phase 4b (retries far costlier per unit progress there) or Phase 5.
- **K3:** GAP null at largest affordable scale with clean pipeline AND cross-category null → ICL memory dead here; paper is the negative result + SFT/LoRA comparison.

### Risks & mitigations
| Risk | Mitigation |
|---|---|
| Judge noise swamps a +4pt effect | Reliability gate (G0.2); doubled judge passes; paired tests |
| Judge-gaming via rubric-speak | Placebo control arm; rubric-item profile |
| Held-out rubric leakage through teacher | Teacher sees train-stream rubrics only; per-item provenance; entity-name audit |
| N=120 held-out underpowered | Continuous scores + paired bootstrap; pre-register +4pt bar; report CIs |
| Student floors at ~0 | Headroom-based band selection (G0.4) before committing |
| Teacher scores ~66% — repairs may be wrong | Repairs rubric-guided (teacher + rubric ≫ teacher alone); log repair-vs-rubric agreement |
| Memory token bloat | Hard 4-item / ~1k-token injection cap; skeletonized exemplars |
| A2 retry arm mis-specified (strawman or leakage) | Retries are self-critique only — no rubric, no judge feedback between attempts; final answer = student's own pick, judged once; publish the retry prompt |
| Memory-build cost unbounded (RALPH's failure mode) | Write-stopping rule (§3): freeze on uplift-window stall or admission-rate collapse |

### Pre-registered predictions
1. **P1:** ≤10 named traps account for ≥50% of the student's lost penalty points.
2. **P2:** Trap registry + playbooks (rules) > skeleton exemplars on held-out.
3. **P3:** Uplift larger on questions that quote the governing standard than on ones that don't.
4. **P4:** Cross-category transfer positive but smaller than within-category.
5. **P5:** At the smallest band, LoRA-SFT on repaired answers > ICL memory; ordering flips/equalizes by the largest band.
6. **P6:** Compute-matched retries (A2) beat student-alone (A1) by a nontrivial margin — why A2, not A1, is the honest baseline.
7. **P7:** Memory-write saturation: window-mean uplift of newly admitted items decays below threshold within ~120 train-stream repairs.

> **One-sentence pitch v2.1:** On expert rubric-graded reasoning, teacher-built memory earns its keep only if it beats compute-matched retries — and it does so to the extent the student's failures are *procedural* (trap susceptibility, framework mis-sequencing), with a rubric-derived, uplift-gated trap registry as the highest-leverage artifact; knowledge gaps require the weights lever, and cross-task persistence of gated skills (Phase 4b) is the unpublished frontier the RALPH→Zenith study leaves open.

*Bars (+4 pts, thresholds, W=15 window, week counts) are proposals — recalibrate after Phase 0 baselines.*

## Appendix — Repo status (2026-07-20, v2.1 adoption)

**Phase 0 nearly complete** (`docs/FINDINGS_FINANCE.md`):
- G0.1 splits frozen 200/80/120 seed 42, rubric firewall (test-enforced) — done.
- G0.2 judge (gpt-5.2) ≠ teacher; reliability PASS: low-range MAD 4.46 / r 0.83, band-range MAD 4.19 / r 0.96 → JUDGE_PASSES=1.
- **Teacher = `deepseek/deepseek-v3.2`** (swapped from minimax-m3 on 2026-07-20: minimax hung/timed out at 240s on long finance answers; deepseek ~1.5s/ping, strong reasoner, disjoint from judge and student family). Ceiling arm (A5) AND Phase-2 repair engine.
### CTO directive (2026-07-20, 21:55) — focused single test

> "Let's test one implementation at a time. Take finance-pro-bench. Use **GLM 5.2 as the teacher**, **Qwen 3.6 27B as the student**. See how **TraceLift** is able to improve. Don't go for methods that require finetuning."

This narrows the active plan to ONE vertical slice, superseding the multi-phase sweep for now:
- **Student:** `qwen/qwen3.6-27b` (fixed by CTO — no headroom re-selection needed).
- **Teacher:** `z-ai/glm-5.2` (was deepseek-v3.2). Heavy reasoner: needs ≥4000 max_tokens or content returns empty (truncates mid-thinking). Judge stays `openai/gpt-5.2` — still ≠ teacher ✓.
- **Method:** TraceLift = uplift-gated memory only (`correction/tracelift.py`, ported from GSM8K). **NO fine-tuning / LoRA** — Phase 5 is OFF.
- **Deliverable:** A1 (student alone) vs A4 (student + TraceLift memory) on held-out, with A2 (compute-matched retries) as the honest bar and A5 (GLM teacher) as ceiling.
- Everything on OpenRouter.

### Platform

- **PLATFORM = OpenRouter only** (CTO requirement, 2026-07-20). All three roles use identical model slugs on OpenRouter (`qwen/qwen3.6-27b`, `deepseek/deepseek-v3.2`, `openai/gpt-5.2`) — all confirmed serving. Entrypoint: `scripts/use_openrouter_finance.sh`. **Prime is abandoned.** Consequence: the Prime-measured Phase-0 gates (judge MAD 4.19, student headroom 26.3) must be **re-validated on OpenRouter** before trusting them (same slug ≠ same serving backend), and the Prime baseline partial is discarded — the full held-out baseline re-runs on OpenRouter so no comparison mixes platforms.
- G0.4 student = **qwen/qwen3.6-27b** (26.3/100, mid-band; thinking disabled). 8B floored at 9.3.
- G0.3 held-out baselines (A1 student + A5 teacher) — running at v2.1 adoption.
- Pending human gate: 15-Q hand-audit (`runs/judge_audit_sample.md`).

**v2.1 deltas that affect the build (not yet implemented):**
- **A2 compute-matched retry arm** — new module: self-critique retry loop, token-budget-matched to A4, no rubric/judge feedback between attempts, best-of-attempts by student's own pick. Needed for Phase-2 eval, NOT Phase 0. Primary metric becomes `GAP_retry = A4 − A2`.
- **Memory-write stopping rule** (§3) — W=15 window, uplift < +0.5 or admission < 20% → freeze. Built into the Phase-2 memory-build loop.
- **A3 placebo memory arm** — already planned; keep.
- **Phase 4b** replaces the parked ALFWorld B1 as the long-horizon claim (RALPH→Zenith harness, gated persistent skill library).

**Carry-over machinery:** `analysis/bootstrap.py`, uplift-audit structure, `_chat_with_retry`, chunked+resumable [LIVE] pattern, injection audit.
