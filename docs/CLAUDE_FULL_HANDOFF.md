# Agent Self-Improvement / “Agent Watch” — Complete Claude Handoff

**Paste this whole file into Claude as session context.** Do not hide nulls.

**Repo:** `/Users/rohanchavan/Desktop/Projects/agent-self-improvement` · `https://github.com/rohanpc0701/agent-self-improvement`  
**Author:** Rohan Chavan (`rohanpc0701`) · **Branch when written:** `feat/q1-self-plan` @ `c788a6d`  
**Sibling settled repo:** `~/Desktop/Projects/reasoning-rsi` (definitive PRBench grid; this repo holds history + Q1)

**Also read (in order):**  
1. `docs/CONTEXT_SAVE_2026-07-30.md` — current truth  
2. `docs/RESULTS_Q1_SELF_PLAN.md` + `docs/prereg/PREREG_Q1_SELF_PLAN.md`  
3. `docs/CODEX_BRAINSTORM_CONTEXT.md` — bridge to reasoning-rsi  

**Doc freshness warning:** Root `CLAUDE.md`, `CLAUDE.local.md`, `README.md`, `harness/CLAUDE.md`, `correction/CLAUDE.md` still describe the **Spider/coding/orchestrator** era. Current tree is **finance-only** after `6935400` (“Remove coding + Spider + old orchestrator framework”). Prefer CONTEXT_SAVE + RESULTS_Q1 as current truth.

---

## 1. One-sentence product + thesis

**Product:** Open-source **runtime self-improvement harness** for verifiable agents: detect accuracy drift → stronger teacher turns failures into memory (few-shots / KG / TraceLift lessons / per-task plans) → **same cheap student** recovers on held-out work — no fine-tuning, no human in the loop.

**Thesis under test (evolved):** Can a stronger teacher build **prompt-appended memory** (or later, **per-task guidance**) that lifts a cheaper student on **held-out** hard domain-reasoning — with improvement attributable to what was learned, not model swap?

**Current settled read (as of 2026-07-30):** Frozen/broadcast teacher memory is **dead** (four delivery forms, three+ domains, two stacks). **Per-task teacher guidance** (HINT/plan) survives. Q1 asked whether that guidance works via plan **STRUCTURE** (student can self-plan for free) vs teacher **KNOWLEDGE** (teacher stays hot-path). **Preregistered verdict: STOP / re-scope** — mechanism unresolved; recovery ratio must not be quoted.

---

## 2. Architecture

### Classic 4-stage loop (original spine — largely deleted from this branch)

```
Harness ──TelemetryRecord──▶ Detector ──DriftEvent──▶ Correction
 ▲                                                        │
 └──── AgentConfig.few_shot_examples (+ optional KG) ─────┘
 all stages ──▶ events.jsonl ──▶ Viewer
```

| Stage | Dir | Consumes | Emits |
|-------|-----|----------|-------|
| Harness | `harness/` | task data | `TelemetryRecord` |
| Detector | `detector/` | telemetry | `DriftEvent` |
| Correction | `correction/` | drift + failures | `CorrectionAction` |
| Viewer | `viewer/` | `events.jsonl` | UI |

**Contracts:** `contracts/schemas.py` — Pydantic; domain-agnostic names (`generated_output`, `domain_id`, …) with legacy SQL aliases. `contracts/eventlog.py` appends typed events.

**Learning = growing text in the prompt**, not weight updates. Teacher is episodic (on drift / train failures), not every query.

### What’s left on `feat/q1-self-plan` (finance-only)

- **Present:** `adapters/finance.py`, `adapters/prbench.py`, `correction/*` (judge, tracelift, graph, etc.), `scripts/finance_*.py`, `scripts/prbench_*.py`, `scripts/q1_*.py`, `fixtures/finance_*`, `fixtures/prbench_*`, `analysis/bootstrap.py`, stubs of `harness/` (agent/evaluator/feed; **no runner**), `detector/` (**tests/__pycache__ only — source deleted**).
- **Gone:** `orchestrator.py`, `viewer/`, coding/spider/gsm8k adapters, detector implementation, `harness/runner.py`, `harness/sandbox.py`.

Current “loop” is **script-driven eval harnesses**, not the classic orchestrator.

---

## 3. Experiment tracks (chronological, honest outcomes)

### A. Original Spider text-to-SQL / drift + few-shots + KG

**Era:** Hackathon → Logan polished demo on `main` / `feat/integration`.  
**Models (demo):** student MiniMax-M2.7 / local Qwen2.5-1.5B; teacher MiniMax-M3.  
**Mechanism:** windowed drift on `execution_accuracy` (last 25; baseline ~40 easy; 5 sustained breaches) → teacher SQL → verify vs gold on SQLite → few-shots (+ anchors) → optional KG `(trap, fix)`.

| Config | WITHOUT | WITH | Δ |
|--------|--------:|-----:|--:|
| Local 1.5B + MiniMax teacher, examples only | 0.100 | 0.333 | **+0.233** |
| Hackathon MiniMax M2.7 + M3 (Logan README / CLAUDE.local) | 0.300 | 0.567 | **+0.267** |

**KG A/B (1.5B):** examples only **0.333** vs examples+rules **0.300** — rules slightly hurt.  
**Dose-response:** 8 capped `failing_run_ids` → Δ **+0.000**; harvesting full degraded window (~24) → **+0.233**.  
**KG wiring:** Mihir’s `graph.py` / `inject.py` / `on_drift.py` existed; **not wired in main orchestrator loop**. Retrieval = **substring** of trigger / schema names in question text — **not vectors**.  
**Status on this branch:** Spider code **removed** (`6935400`). Docs/README still quote these numbers.

**Brutal verdict:** Working **demo of same-distribution recovery**. Not proven held-out self-improvement.

---

### B. Coding adapter / capacity floor

**Findings:** `docs/FINDINGS_CODING.md`, `docs/FABLE_HANDOFF.md`  
**Platform:** Prime. Teacher `minimax/minimax-m2.5`.

| Result | Numbers |
|--------|---------|
| Classic `--full` (same-dist recovery) | WITHOUT **0.273** → WITH **0.455**, Δ **+0.182** |
| Hard-curriculum held-out | WITHOUT **0.111** → WITH **0.111**, Δ **+0.000**; vs teacher **0.111 vs 1.000** |
| Frozen ablation 3B (n=29) | none 0.103 / examples 0.034 / rules 0.103 / both 0.034 |
| Capacity McNemar | qwen3-coder bare **0.931** vs 3B+memory **0.034**, p ≈ **0.0000** |
| Nemo 4-repeat variance | mean Δ(examples−none) **0.000** |
| Qwen3.5-4B 3-repeat | none **0.588** → examples **0.324**, Δ **−0.265 identical every repeat** |
| Rules channel | **inert** on every student (0 discordant) |

**Verdict:** Loop works mechanically; **memory does not transfer** on hard held-out. Capacity necessary ≠ sufficient.

---

### C. GSM8K / TraceLift gate

**Findings:** `docs/FINDINGS_REASONING.md`  
**Student:** `meta-llama/llama-3.2-3b-instruct` @ OpenRouter.

| Stage | Result |
|-------|--------|
| Uplift gate | Kept 3 items (u +0.333, +0.333, +0.167); dropped 0 / −0.333 |
| Curriculum headline | WITHOUT 0.333 → WITH 0.417, Δ **+0.083** (not frozen ablation) |
| Frozen ablation (n=28) | none **0.321** → uplift-memory **0.179**, Δ **−0.143**; **43% zero-injection** |

**Verdict:** Gate **mechanically real**; held-out transfer **failed**.

---

### D. ALFWorld

**PARKED.** Env built; band floors (nemo 0.033, qwen3.5-4b 0.000, gpt-oss-20b 0.100). Superseded by finance pivot. No full LEARN→held-out claim.

---

### E. FinancePro-Bench / RSI-Mem v2

**Plan:** `docs/RSI_MEM_V2_FINANCE.md` · **Findings:** `docs/FINDINGS_FINANCE.md`  
**Data:** HF `Sanscritic/finance-pro-bench` 400 rows. Split seed **42**: 200 / 80 / 120.  
**CTO (2026-07-20):** student `qwen/qwen3.6-27b`, teacher `z-ai/glm-5.2`, OpenRouter, no FT.

| Gate / probe | Numbers |
|--------------|---------|
| Judge reliability (n=26) | pearson_r **0.829**, MAD **4.46** → PASS_SINGLE |
| Headroom | 8b **9.262**; 30b **15.753**; **27b 26.305** (chosen) |
| Band-range (n=17) | r **0.962**, MAD **4.19** → PASS_SINGLE |
| Uplift gate finance | **Admitted: 0** then **bypassed** |
| Boilerplate memory | GAP **−6.1** (n=5) |
| Good memory single-pass | GAP **+5.6** (n=5) |
| Good memory **k=3** | GAP **+0.0** — **+5.6 was noise** |

**Do not claim TraceLift works from FinancePro +5.6.**

---

### F. PRBench Corporate Finance

**This repo (early):** PLAIN **71.9** → MEM **77.2** (**+5.3**, 8W/2T/2L, n=12, ungated, no bootstrap) → TEACHER **90.5**. Later called superseded.

**Definitive settled results in sibling `reasoning-rsi`:**

**Run 1 Fireworks (28×k=3):** gated MEM **+1.39 n.s.**; REFINE **+6.64** sig.  
**Run 2 Prime grid:** HINT **+6.0 p=.001**; TRACE +6.1; REFINE null on Prime; CMEM null.

**Bottom line:** Frozen/broadcast memory **dead**; **per-task HINT/plan** is what survived. Platform flip is first-class.

---

### G. Q1 SELF-PLAN (current)

**Prereg:** `docs/prereg/PREREG_Q1_SELF_PLAN.md`  
**Results:** `docs/RESULTS_Q1_SELF_PLAN.md`  
**Regen:** `python3 scripts/q1_analyze.py`

| Arm | Score | vs A | 95% CI | p |
|-----|------:|-----:|--------|--:|
| A NOPLAN-MATCHED | **36.81** | — | — | — |
| B SELF-PLAN | **40.43** | **+3.62** | [−0.23, +7.73] | 0.067 |
| C TEACHER-PLAN | **42.02** | **+5.21** | **[+0.47, +9.76]** | **0.031** |

- Recovery point 0.695; ratio CI **[−0.00, 1.81]** spans all branches.  
- **Registered branch: STOP / re-scope** — do **not** quote recovery.  
- Teacher-plan effect replicates (+5.21). Self-plan unproven. Mechanism unresolved.

**Next:** Q2-SELF-PLAN (~$35): test `B−A` directly, drop C, n=80.

---

## 4. Memory mechanisms

| Mechanism | How | Status |
|-----------|-----|--------|
| **Few-shots** | Teacher repair → `FewShotExample`; filter by `domain_id`/`db_id`; cap | Demo recovery yes; hard held-out transfer no |
| **KG** | Nodes `schema:{db}:{table[.col]}`, `rule:{db\|global}:{n}`; edges rule→schema | Retrieval = **substring** (trigger or bare name in question). **Not vectors.** Unwired/inert in practice |
| **TraceLift uplift** | `u = with − without` on val | GSM8K: gate ok, transfer fail. Finance: bypassed |
| **Category memory (finance)** | Match category; ≤4 items | Coarse |
| **Contrastive PRBench lessons** | Missed criteria → lesson | Early +5.3; settled gated MEM null |
| **Per-task HINT / plans** | ≤250w plan per question | **What survived** |
| **Self-plan** | Student writes own plan | Q1 unproven (CI touches 0) |

---

## 5. Models & platforms (summary)

Spider: MiniMax. Coding: Prime. GSM8K: OpenRouter. Finance: Prime→OpenRouter. PRBench: OpenRouter Fireworks + Prime grid. Q1: OpenRouter DeepInfra fp8 pin.

**Hard assert:** judge ≠ teacher (≠ student on PRBench).

---

## 6. Wired vs unwired

| Piece | Wired? |
|-------|--------|
| Classic orchestrator / detector / viewer | **Deleted** on this branch |
| KG in main Spider loop | **Never fully wired** |
| TraceLift finance gate | **Bypassed** (admit=0 then ungated) |
| Rubric firewall | **Wired** + tests |
| Q1 harness | **Wired** |
| A2 compute-matched / A3 placebo as primary bars | **Largely unrun** |

---

## 7. Splits / firewalls / arms

- **FinancePro:** 200/80/120 seed 42. Student never sees rubrics; teacher train-only.  
- **PRBench:** 50/15/28 seed 42.  
- **Arms:** A1 alone · A4 memory · A5 teacher (often); Q1 = A restatement / B self-plan / C teacher-plan.  
- Held-out once for claims. Validation only for gating.

---

## 8. File map

| Path | Role |
|------|------|
| `contracts/schemas.py`, `eventlog.py` | Contracts |
| `adapters/finance.py`, `adapters/prbench.py` | Domain adapters + firewall |
| `correction/tracelift.py`, `judge.py`, `prbench_judge.py`, `graph.py`, `inject.py` | Learning / KG / judges |
| `scripts/q1_*.py`, `finance_*.py`, `prbench_*.py` | Eval harnesses |
| `fixtures/finance_*`, `prbench_*` | Data + manifests |
| `docs/CONTEXT_SAVE_2026-07-30.md` | **Start here** |
| `docs/FINDINGS_{CODING,REASONING,FINANCE}.md` | Honest numbers |
| `runs/q1_*` | Q1 cells |

---

## 9. Gotchas

- Stale CLAUDE/README vs finance-only tree.  
- `reasoning.enabled=false` on qwen3.6-27b or empty content.  
- `TEACHER_MAX_TOKENS` must be high (12k–32k) for reasoning teachers.  
- Judge truncation / wrong norm denominator bugs (fixed in Q1 D2/D3).  
- Branch-switching mid-run deleted scripts / killed evals — pin branch.  
- Session env kills long jobs — use VM tmux.  
- Platform flip: never merge Prime+OpenRouter cells.  
- Temp-0 noise ±10–20 pts → need **k≥3**.  
- Author commits: Rohan; prefer no Cursor co-author trailer.

---

## 10. What NOT to claim

- TraceLift “works” from FinancePro +5.6 (k=3 → **+0.0**).  
- This-repo PRBench +5.3 as settled proof (superseded; gated MEM null in reasoning-rsi).  
- Q1 recovery **0.695** as a mechanism result (**STOP**).  
- KG helped coding.  
- Uplift gating was tested on finance.  
- ALFWorld results.  
- README coding +0.182 as hard held-out transfer (it’s same-distribution recovery).  
- Mix platforms in one paired analysis.  
- Validation uplift as held-out evidence.

---

## Standing discipline

≥3 repeats · held-out once · pre-register bars · publish nulls · judge ≠ teacher · provider-pin · loud empty failures · pre-flight before spend.

---

*Generated for Claude handoff from repo state on feat/q1-self-plan. Prefer CONTEXT_SAVE if this file drifts.*
