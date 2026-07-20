# RSI-Mem v2 — Runtime Self-Improvement on FinancePro-Bench

**Adopted 2026-07-20. Supersedes `docs/RSI_MEM_PLAN.md` (v1, coding) as the active
track.** v1 machinery carries over where noted (per-item uplift audit, multi-seed
eval runner + paired bootstrap, variance protocol). Dataset verified on HF:
400 rows, fields {id, category, question, rubric}, rubrics contain R/T/B items,
CC-BY-4.0.

**Target benchmark:** [Sanscritic/finance-pro-bench](https://huggingface.co/datasets/Sanscritic/finance-pro-bench) — 400 expert-level finance questions, 33 categories, free-text answers graded by point rubrics (tiered items R1–Rn, trap penalties T1–Tn, insight bonuses B1–Bn). All information is embedded in the question; pure reasoning, no retrieval.

> **North star:** `student + frozen memory > student alone` on held-out questions, measured in normalized rubric score, with memory built only from training-stream failures.

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

**Honest framing note:** this benchmark is *deep* reasoning, not *long-horizon interactive* reasoning. If the research claim you want is about long-horizon agents (ALFWorld-class), keep that as a separate phase. What this benchmark uniquely tests: whether teacher-built memory can transfer **domain reasoning procedure** (framework-application patterns, trap avoidance) — which is closer to Memp's "scripts" than anything the coding domain offered.

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
- **Judge ≠ teacher (repo addition):** the judge model must differ from the teacher model, or the judge will favor teacher-styled repairs — self-preference bias contaminates GAP.

### Metrics
- **Primary:** `GAP = mean normalized score(student+memory) − mean score(student alone)` on HELD-OUT, ≥3 seeds, paired bootstrap over questions.
- **Success bar (proposal):** GAP ≥ +4 normalized points, p < 0.05, AND memory arm > placebo arm.
- **Secondary:** trap-penalty rate (per category, per named trap); rubric-item profile (which R-items improve); insight-bonus rate; % of teacher–student gap closed; memory tokens per query.

## 3. Memory design (adapted to rubric structure)

Three artifact types, all teacher-built from train-stream failures:

1. **Category playbooks (rules/scripts — Memp's "script" finding says these generalize).** Per-category analysis checklists, e.g. "VIE analysis: (1) equity-sufficiency gate before voting model, (2) kick-out rights: check single-party exercisability, (3) decision-maker fee: aggregate ALL other interests before safe harbor."
2. **Trap registry.** Named failure modes with trigger conditions, harvested from trap penalties the student actually hit: "TRAP: applying fee safe-harbor without aggregation → always list the manager's other economic interests first." This is the highest-value artifact this benchmark enables — traps are explicit, named, and recurring.
3. **Compressed exemplars (few-shots).** Questions are 3–10k chars and expert answers are long — raw few-shots will blow the context and dilute (SelfMem). Store *skeletonized* worked solutions: issue → framework → 5-step reasoning → conclusion, ≤300 tokens each.

Write path: teacher repairs a failed answer against the rubric → distills the repair into playbook deltas / trap entries / one skeleton → **uplift gate** → compaction/merge → frozen store.

## 4. Phase plan (20 weeks)

### Phase 0 — Splits, judge harness, honest baselines (wk 1–2)
- G0.1 Freeze the 200/80/120 split (stratified, seeded, published).
- G0.2 Build the judge harness + reliability check (§2). **Exit blocker:** proceed only if judge test–retest MAD ≤ 5 pts (or doubled passes fix it).
- G0.3 Baselines with ≥3 seeds: student alone · unaided teacher · (existing memory if ported). Also record per-category student scores.
- G0.4 **Choose student bands by headroom, not dogma:** if the 3B scores <10 normalized on held-out, it has no sensitivity to measure lift. Pick the smallest student scoring roughly 15–40 — likely 7–8B for this benchmark. (Frontier ≈ 66; a 3B may simply floor.)

> **Exit:** trusted judge + baseline table with CIs + chosen student band(s).

### Phase 1 — Diagnose (wk 2–4)
- G1.1 **Trap census:** on train-stream failures, tabulate which trap penalties fire per category. Hypothesis: a small set of traps accounts for most lost points (if so, the trap registry is the main lever).
- G1.2 **Rubric-item profile:** are points lost on calculations (R-items requiring math), framework selection, or synthesis? Different memory helps each differently.
- G1.3 **Uplift audit** of any existing/ported memory: `u = mean score(with item) − mean score(without)`, K=2–3 (continuous signal → smaller K than coding), on VALIDATION.
- G1.4 **Procedural-vs-informational probe (OPHSD):** does the teacher's repair advantage come from reasoning procedure (checklists — promptable) or from finance knowledge the small model lacks (e.g., knowing ASC 810 content — likely NOT promptable at small scale)? For questions where the standard's text is quoted in exhibits (many are), procedure should dominate — test this directly by comparing uplift on quote-included vs quote-free questions.

> **Decision gate:** failure taxonomy across {knowledge gap, procedure gap, trap susceptibility, calculation error} with % attribution → sets Phase 2 emphasis.

### Phase 2 — Best-possible ICL memory (wk 4–9)
- G2.1 **Uplift-gated writes** (u > +1 normalized pt on validation; continuous threshold replaces u>0 to clear judge noise).
- G2.2 **Compaction (SelfMem):** hard cap ~4 items injected per query: 1 category playbook + ≤2 trap entries + ≤1 skeleton exemplar. Merge over append.
- G2.3 **Retrieval = category + trap-pattern match (RA-RFT adapted):** primary key is the question's category; secondary, teacher-labeled "which traps is this question likely to spring" from surface features. No embedding similarity.
- G2.4 **Placebo arm** (style-only memory) runs alongside.
- G2.5 Full held-out evaluation: student alone · +placebo · +real memory · teacher.

> **Exit:** SUCCESS = GAP ≥ +4 AND real > placebo → write-up track. NULL with clean pipeline → capacity/knowledge floor confirmed → Phases 3/5.

### Phase 3 — Capacity bands (wk 9–12)
- Same pipeline at 2–3 student scales chosen for headroom (e.g., 7–8B, 14B, 32B if budget allows — finance knowledge scales steeply).
- Deliverable: GAP-vs-scale curve + trap-rate-vs-scale curve (prediction: trap avoidance transfers at smaller scale than calculation accuracy).

### Phase 4 — Cross-category generalization (wk 12–15) *(replaces ALFWorld)*
- Build memory from train-stream questions in ~22 categories; evaluate held-out questions from the ~11 categories never seen in training.
- Tests whether playbooks encode *transferable reasoning discipline* (framework-gating, trap vigilance) vs. category-local knowledge. Either answer is a finding.
- *(Optional Phase 4b: keep an ALFWorld run if you still want the interactive long-horizon claim — but it's now a separate claim, not this benchmark's.)*

### Phase 5 — Prompting vs LoRA (wk 15–18)
- OPHSD-style: generate rollouts with memory attached, distill (LoRA) into the bare student, evaluate memory-removed; test reattachment.
- Extra arm this domain motivates: LoRA on teacher-repaired full answers (SFT) vs on-policy distillation — the knowledge-gap hypothesis predicts SFT injects finance content ICL cannot.

### Phase 6 — Write-up (wk 18–20)
- Positive: *"Trap-Registry Memory: rubric-grounded runtime self-improvement."* The trap-avoidance mechanism is the novel, benchmark-native contribution.
- Negative: *"Procedure is promptable, knowledge is not: limits of ICL self-improvement in expert domains."*
- Release: splits, judge harness, memory store, all arms' graded outputs.

## 5. Kill criteria, risks, predictions

### Kill criteria
- **K1:** Judge test–retest MAD > 5 pts even with doubled passes → benchmark unusable for small-effect measurement; renegotiate metric (e.g., trap-rate only, which is more discrete/reliable) or switch benchmark.
- **K2:** After Phase 2, real memory ≤ placebo → the "lift" channel is rubric-style, not reasoning; pivot to LoRA (Phase 5) immediately.
- **K3:** GAP null at the largest affordable scale with clean pipeline AND cross-category null → ICL memory dead here; the paper is the negative result + the SFT/LoRA comparison.

### Risks & mitigations
| Risk | Mitigation |
|---|---|
| Judge noise swamps a +4pt effect | Reliability check gate (G0.2); doubled judge passes; paired tests |
| Judge-gaming via rubric-speak | Placebo-memory control arm; rubric-item profile (style inflates evidence-quote items differently than calculation items) |
| Held-out rubric leakage through teacher | Teacher sees train-stream rubrics only; per-item provenance; entity-name audit |
| N=120 held-out underpowered | Continuous scores + paired bootstrap; pre-register +4pt bar; report CIs not just p |
| Student floors at ~0 (no sensitivity) | Headroom-based band selection (G0.4) before committing |
| Teacher itself scores ~66% — repairs may be wrong | Repairs are rubric-guided (teacher + rubric ≫ teacher alone); log repair-vs-rubric agreement |
| Memory token bloat (long questions already 3–10k chars) | Hard 4-item / ~1k-token injection cap; skeletonized exemplars |

### Pre-registered predictions
1. **P1:** ≤10 named traps account for ≥50% of the student's lost penalty points (trap concentration).
2. **P2:** Trap registry + playbooks (rules) > skeleton exemplars on held-out (Memp scripts finding, restated).
3. **P3:** Uplift is larger on questions that quote the governing standard in exhibits than on ones that don't (procedure vs knowledge).
4. **P4:** Cross-category transfer is positive but smaller than within-category (reasoning discipline transfers; content doesn't).
5. **P5:** At the smallest band, LoRA-SFT on repaired answers > ICL memory; the ordering flips or equalizes by the largest band.

> **One-sentence pitch v2:** On expert rubric-graded reasoning, teacher-built memory works to the extent the student's failures are *procedural* — trap susceptibility and framework mis-sequencing — and a rubric-derived trap registry is the highest-leverage memory artifact; knowledge gaps require the weights lever.

*Bars (+4 pts, thresholds, week counts) are proposals — recalibrate after Phase 0 baselines.*

## Appendix — Repo carry-over notes (2026-07-20)

- **v1 work order status at pivot:** Cursor mid-Task-1 on `feat/rsi-mem-phase0`
  (coding pool expansion; probe 71 candidates in, uncommitted). Disposition:
  commit as-is and stop; Tasks 2–3 machinery (uplift audit, bootstrap runner)
  gets built against the finance adapter instead.
- **Transfers directly:** `analysis/bootstrap.py` design, uplift-audit
  plan/summarize/resume structure, variance protocol discipline (≥3 repeats),
  `_chat_with_retry`, chunked+resumable [LIVE] run pattern, injection audit.
- **Judge model:** must differ from teacher (self-preference bias). Both
  configurable; reliability check (G0.2) is the acceptance gate.
- **Adapter:** new `adapters/finance.py` implementing the TaskAdapter protocol;
  `execution_accuracy` carries the normalized judge score (0–1) — contracts
  unchanged.
