# RSI-Mem — Runtime Self-Improvement via Teacher-Built Memory

**Project plan · 6 phases · 24 weeks · adopted 2026-07-20**

> **North star:** `student + frozen memory > student alone` on held-out hard problems, with a meaningful gap. No fine-tuning, no model swap.

> **Supersedes:** the 2026-07-19 "coding closed" direction — Phases 0–3 run in the
> coding domain; ALFWorld becomes Phase 4 (domain-leverage test), where the
> existing B1 spec (`docs/superpowers/specs/2026-07-19-alfworld-longhorizon-b1-design.md`)
> plugs in.

## 1. Problem statement & thesis

### The claim under test
A weak student agent fails on hard work → a stronger teacher turns those failures into memory (verified few-shots + short rules) → the **same student**, with that frozen memory, does better on **new held-out** problems.

### Where the project stands
- **Memory injection works** — verified, not a plumbing bug.
- **No held-out lift at 3B** — memory doesn't help hard held-out problems; sometimes hurts.
- **Unaided strong coder ≈ 0.93** on the same held-out set — the ceiling is far above the student.

### The open research question
> **RQ:** Given a student strong enough to sometimes solve hard problems unaided, can teacher-built memory still produce a real held-out lift — or is self-improvement via ICL/rules a dead end in this setting?

### What the literature predicts

| Evidence | Implication for this project |
|---|---|
| **Memp:** GPT-4o memory → Qwen2.5-14B, +5% completion, −1.6 steps | The claim IS achievable — but demonstrated at 14B on long-horizon tasks, not 3B single-shot coding. Capacity floor + domain leverage are the two suspects. |
| **TraceLift:** quality-scored-but-unfiltered traces score WORSE than none | Teacher-verified ≠ student-useful. Memory items need a measured-uplift gate. |
| **RA-RFT:** utility retrieval lifts a 1.7B student +7.1 on AIME25 | Sub-3B models CAN be moved by ICL — if exemplars are selected for reasoning transfer, not similarity. |
| **SelfMem:** optimizer converges to compact memory; verbose memory dilutes | Memory bloat is a live failure channel; strict write gate + merge. |
| **Memp:** scripts generalize to dissimilar tasks; trajectories help similar ones | On held-out-by-construction sets, rules should carry; few-shots may be the hurting component. |
| **OPHSD/SIA:** distill scaffold into weights when ICL stalls | The fallback lever: same teacher artifacts, delivered via LoRA instead of prompt. |

## 2. Success criteria & metrics

### Primary metric
```
GAP = pass@1(student + frozen memory) − pass@1(student alone)
on HELD-OUT HARD problems, greedy decoding, ≥3 seeds

SUCCESS: GAP ≥ +5 pp, p < 0.05 (paired bootstrap), on ≥200 problems,
and the memory was built ONLY from training-stream failures (no leakage)
```

### Secondary metrics
- **Δ vs unaided teacher** — % of the teacher–student gap closed by memory.
- **Per-item uplift distribution** — fraction of memory items with u > 0 on validation.
- **Token cost** — memory tokens injected per problem; report gap per 1k tokens.
- **Long-horizon:** success rate AND steps (Memp's dual metric).

### Honesty rules (fixed for all phases)
- Held-out set frozen before any memory is built; never used for selection, iteration, or gating.
- Three arms always reported: student alone · student + memory · unaided teacher.
- Same decoding config across arms; ≥3 seeds; paired tests.
- Difficulty banding fixed up front (by unaided-student solve-rate buckets), not post hoc.

## 3. Phase plan (24 weeks)

### Phase 0 — Instrumentation & honest baselines (wk 1–2)
**Goal:** a reproducible harness where every later comparison is trustworthy.

- **G0.1** Freeze held-out hard set (≥200 problems, unit-test verified) + difficulty bands.
- **G0.2** Eval runner: 3 arms × ≥3 seeds, per-problem logs (pass, tokens, retrieved items).
- **G0.3** Re-establish current numbers with variance: student alone, student+memory, teacher.

> **Exit:** Baseline table with CIs reproduces the known null at 3B. If "memory hurts" disappears under proper seeding — that alone is a finding.

### Phase 1 — Diagnose the failure (wk 2–4)
**Goal:** attribute the null to capacity, content, retrieval, or dilution — before fixing anything.

- **G1.1 Uplift audit (TraceLift).** For every memory item: `u = mean pass(with item) − mean pass(without)`, K=3 rollouts on a validation slice. Report the u-distribution.
- **G1.2 Format ablation (Memp).** Rules-only vs few-shots-only vs both. Prediction: rules ≥ both > few-shots on dissimilar problems.
- **G1.3 Procedural-vs-informational probe (OPHSD).** Is the teacher's advantage reasoning structure (promptable/distillable) or knowledge the 3B lacks (not promptable)?
- **G1.4 Dilution check (SelfMem/Memp).** Sweep injected items: 0, 1, 2, 5, 10, all. Look for rise-then-fall.

> **Decision gate:** one-page failure taxonomy with % attribution.
> - ≥30% items with u < 0 → content problem → Phase 2 is the main bet.
> - Advantage mostly informational → capacity problem → weight Phases 3/5.
> - Few-shots hurt, rules flat-positive → format problem → rules-first pipeline.

### Phase 2 — Fix the memory pipeline (wk 4–8)
**Goal:** the best possible ICL memory at 3B — so a later null is about capacity, not pipeline quality.

- **G2.1 Uplift-gated writes.** Only items with measured u > 0 enter frozen memory. *The project's novel mechanism — no paper in the set gates memory writes by causal uplift.*
- **G2.2 SelfMem-style compaction.** Strict write gate (stable reusable procedures only), merge/replace over append, hard cap (≤5 live items per problem).
- **G2.3 Utility-based retrieval (RA-RFT).** Teacher labels exemplar↔problem strategy-relevance; retrieve by that, not embedding similarity.
- **G2.4** Re-run the full loop with the gated+compact+utility pipeline, 3 arms, 3 seeds.

> **Exit:**
> - SUCCESS: GAP ≥ +5 pp at 3B → thesis holds at small scale; write up around uplift-gating.
> - PARTIAL: 0 < GAP < 5 pp → pipeline helps; capacity binding → Phase 3.
> - NULL with a verified-clean pipeline → strong evidence for a capacity floor → Phase 3 is the headline.

### Phase 3 — Capacity band sweep (wk 8–11)
**Goal:** locate the capacity floor for ICL self-improvement.

- **G3.1** Run the Phase-2 pipeline unchanged at ~3B, ~7–8B, ~14B (same family to isolate scale).
- **G3.2** Plot GAP vs scale; identify the threshold band.
- **G3.3** Control: teacher still strictly dominates each student unaided.

> **Exit:** a capacity-threshold curve. Lift at 7B/14B while flat at 3B is publishable either way.

### Phase 4 — Domain leverage: long-horizon (wk 11–15)
**Goal:** test whether memory pays where it eliminates exploration, not in single-shot problems.

- **G4.1** Port the loop to ALFWorld: stream tasks, detect failure, teacher repairs, build memory.
- **G4.2** Replicate Memp's transfer with a twist: their memory came from gold trajectories; ours from **repaired failures**. Matching their +5% / −1.6-step shape with failure-derived memory is itself a contribution.
- **G4.3** Report success rate AND steps; anchor against Memp's published Qwen2.5-14B numbers.

> **Exit:** if long-horizon lifts at a scale where coding is flat → ICL self-improvement is domain-leverage-bound, not dead.

### Phase 5 — The weights lever: prompting vs LoRA (wk 15–20)
**Goal:** when ICL stalls, deliver the same teacher artifacts through weights (SIA lever switch; OPHSD mechanism).

- **G5.1 OPHSD-style distillation:** run student WITH memory to generate rollouts; distill on-policy (reverse-KL or SFT-on-own-assisted-rollouts, LoRA) into the bare student; evaluate with memory removed.
- **G5.2** Head-to-head at each viable scale: prompting-only vs LoRA-only vs LoRA+prompting.
- **G5.3** Test OPHSD's reattachment finding: does re-adding memory after distillation help, do nothing, or hurt?
- **G5.4** Cost accounting: LoRA training cost vs per-query memory-token cost — the practical crossover.

> **Exit:** a decision rule — "below scale X or above reuse-frequency Y, distill; otherwise prompt."

### Phase 6 — Write-up & positioning (wk 20–24)
- **G6.1** Position in the survey's "self-evolving reasoning" layer; nearest neighbors: Memp, TraceLift, RA-RFT, OPHSD.
- **G6.2** Two viable papers regardless of outcome:
  - (a) positive — *Uplift-Gated Teacher Memory: runtime self-improvement without fine-tuning*
  - (b) negative — *The Capacity Floor of ICL Self-Improvement: when teacher-built memory cannot help*
- **G6.3** Release: memory-build code, uplift-gating tool, frozen eval sets, all three arms' logs.

## 4. The full experimental matrix

| Axis | Levels | Source paper |
|---|---|---|
| Memory format | rules-only · few-shots-only · both | Memp (script/traj) |
| Write gating | all-verified · uplift-gated (u>0) | TraceLift |
| Compaction | unbounded · SelfMem write-gate + cap | SelfMem |
| Retrieval | similarity · teacher utility labels | RA-RFT |
| Student scale | ~3B · ~7–8B · ~14B | Memp / RA-RFT |
| Domain | single-shot code · ALFWorld | Memp |
| Delivery | prompt (ICL) · LoRA distill · both | OPHSD / SIA |

Do **not** run the full cross-product (~500 cells). The phases walk a greedy path: fix pipeline at 3B → sweep scale with the winner → sweep domain → sweep delivery. ~25 runs total.

### Milestone summary

| Wk | Milestone | Go / no-go signal |
|---|---|---|
| 2 | M0: Honest baselines with CIs reproduce the null | Harness trusted |
| 4 | M1: Failure taxonomy (% capacity / content / retrieval / dilution) | Pick Phase-2 emphasis |
| 8 | M2: Best-possible ICL memory at 3B evaluated | GAP ≥ +5pp? → write-up track |
| 11 | M3: Capacity-threshold curve (3B/7B/14B) | Floor located |
| 15 | M4: Long-horizon (ALFWorld) result vs Memp anchor | Domain hypothesis settled |
| 20 | M5: Prompting-vs-LoRA decision rule | RQ answered |
| 24 | M6: Paper draft + code release | Ship |

## 5. Kill criteria, risks, pre-registered predictions

### Kill criteria
- **K1:** After Phase 2, if <10% of teacher-repaired items have u > 0 for the 3B student, stop optimizing 3B ICL — the delivery mechanism, not the pipeline, is broken at this scale. Jump to Phases 3/5.
- **K2:** If GAP is null at 14B with the full pipeline on coding AND ALFWorld, declare ICL-memory self-improvement dead for this recipe; pivot to the negative paper + LoRA comparison.
- **K3:** If LoRA distillation also fails at 7B+, the teacher's advantage is informational beyond what its repairs express — the loop needs a better teacher-artifact format (a new project).

### Risks & mitigations

| Risk | Mitigation |
|---|---|
| Eval leakage (memory built from problems related to held-out) | Freeze held-out first; dedupe by statement similarity; per-item provenance |
| Small-n noise mistaken for signal | ≥200 problems, ≥3 seeds, paired bootstrap; pre-registered +5pp bar |
| Teacher contamination (memorized benchmarks) | Prefer post-cutoff/private problems; teacher pass@1 is a ceiling, not a comparison |
| Uplift gating overfits the validation slice | Disjoint validation vs held-out; re-measure u on a second slice |
| Compute cost of K-rollout audits | K=3 (TraceLift's sweet spot); audit at write time only |
| Family confound in scale sweep | Same model family across bands; note exceptions |

### Pre-registered predictions
1. **P1:** ≥30% of current memory items have u ≤ 0 for the 3B student (explains "sometimes hurts").
2. **P2:** Rules-only ≥ combined > few-shots-only on held-out hard (Memp's script finding).
3. **P3:** GAP turns positive somewhere in 7–14B with the gated pipeline (between RA-RFT's 1.7B and Memp's 14B).
4. **P4:** ALFWorld shows lift at a scale where single-shot coding is flat (domain leverage).
5. **P5:** At 3B, LoRA delivery of the same artifacts beats prompting; the gap narrows with scale (OPHSD).

> **The one-sentence pitch:** Teacher-built memory can self-improve a running agent — but only above a capacity floor and with uplift-gated, compact, utility-retrieved content; below the floor, the same artifacts must be delivered through weights.

---

## Appendix A — Status vs work already done (as of adoption, 2026-07-20)

Repo evidence: `docs/FINDINGS_CODING.md`, `runs/*.log`, diagnostics spec + results.

| Plan item | Status | Evidence / delta |
|---|---|---|
| G0.3 baselines w/ variance | **Largely done** | Variance protocol exists (`scripts/variance_check.py`); nemo 4-repeat, qwen 3-repeat; provider-noise floor measured (<0.06 = noise). Missing: 3-seed protocol, bootstrap. |
| G0.1 ≥200 held-out | **Not done** | Current held-out = 29–34 unique. Need bigger import (EvalPlus uncapped + LCB Stage 2). |
| G1.2 format ablation | **Done (small n)** | none/examples/rules/both on 3 students. Result: rules inert (0 discordant — but only 3 rules existed; dilution/format conclusions need a real rules pipeline). Few-shots: 0 (nemo) to −0.265 deterministic (qwen). **P2 directionally confirmed:** rules (0.000) > few-shots (−0.24). |
| G1.1 uplift audit | **Not done — next build** | Per-item u; extends variance_check machinery. Our aggregate result (whole bundle harms qwen deterministically) strongly suggests P1 true. |
| G1.4 dilution check | Not done | Cheap once uplift audit exists. |
| G2.1 uplift-gated writes | Designed | = "rung 2 utility gate" in ALFWorld B1 spec §6/§9 — mechanism already specced, now retargeted per this plan. |
| Injection audit | **Done** | `TelemetryRecord.injection_stats`; zero-injection ruled out as cause. |
| Capacity signal | **Partial** | 3B vs strong coder p≈0.0000; in-band students found (coding: nemo 0.41, qwen3.5-4B 0.55). Proper same-family 3-scale sweep (G3.1) not done. Note: Qwen2.5-Coder family 404s on Prime — family choice must come from live catalog (qwen3/qwen3.5 sizes all alive). |
| G4.1 ALFWorld port | **Env ready** | venv + patched TextWorld (Inform7-free) working, 134 valid_unseen games load, ReAct runner exists (`scripts/alfworld_band_sweep.py`), goal-pinning bug found+fixed. Band search incomplete: gpt-oss 0.10, qwen3.5-4B 0.00, qwen3-8b partial (1✓ in 7 episodes). Note plan targets ~14B-class for Memp anchor — small models flooring is consistent with that. |
| Ops lessons | — | Prime 429s need retry (fixed); temp-0 nondeterminism is deployment-specific (nemo noisy, qwen deterministic); background jobs in this session get killed — long runs need chunking or user-terminal execution. |

**Immediate next steps under this plan:**
1. **G0.1**: expand held-out to ≥200 (uncap EvalPlus import, add LCB importer if short).
2. **G1.1**: build per-item uplift audit (u per memory item, K=3) — tests P1 directly on the archived qwen memory whose harm is deterministic.
3. **G0.2**: upgrade eval runner to 3-arm × 3-seed + paired bootstrap.
