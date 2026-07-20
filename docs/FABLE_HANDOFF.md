# Handoff for Fable — Agent Self-Improvement

**Author:** Rohan Chavan (`rohanpc0701`)  
**Repo:** https://github.com/rohanpc0701/agent-self-improvement  
**Purpose of this doc:** Full context so you can propose next steps. Do not invent metrics; numbers below are from real runs. Prefer honest null results over optimistic claims.

---

## 1. One-sentence product

Open-source **runtime self-improvement harness** for verifiable agents: detect accuracy drift → stronger teacher turns failures into few-shots + knowledge-graph rules → **same weak student** recovers on held-out work — no fine-tuning, no human in the loop.

**Not:** retry-until-correct, model swap at inference, or a Spider-only demo.

---

## 2. Thesis (what we claim / want to prove)

> After a hard learning phase that populates memory (few-shots + KG) from teacher-verified repairs, a cheap student should improve on **completely new** hard problems — and we should measure that against an **unaided teacher** ceiling, not only against the student’s own zero-shot baseline.

Two related metrics:

| Metric | Question |
|--------|----------|
| **Self-lift** | Student₀ vs Student₊memory on held-out hard (WITHOUT → WITH Δ) |
| **Teacher gap** | Student₊memory vs unaided Teacher on the same held-out hard |

Product north star long-term: problem-agnostic spine → then long-horizon / multi-step agents (trajectory repair, not just final answer).

---

## 3. Architecture (frozen spine)

```
Harness ──TelemetryRecord──▶ Detector ──DriftEvent──▶ Correction
   ▲                                                      │
   └──── few_shot_examples + KG rules (runtime feedback) ─┘
                all stages ──▶ events.jsonl ──▶ Viewer
```

| Dir | Role |
|-----|------|
| `contracts/` | Shared Pydantic schemas + `events.jsonl` I/O |
| `core/` | `TaskAdapter` protocol |
| `adapters/` | `coding`, `spider_sql`, `gsm8k_math` |
| `harness/` | Student client, change-point feed, Spider EX, coding sandbox |
| `detector/` | Windowed drift on `execution_accuracy` |
| `correction/` | Teacher, verify, anchors, KG (`graph.py`, `inject.py`, `distill.py`, `provider.py`) |
| `viewer/` | FastAPI + Chart.js |
| `orchestrator.py` | End-to-end loop |
| `scripts/use_prime_student.sh` | Prime student/teacher coding entrypoint |

**Learning mechanism:** `AgentConfig.few_shot_examples` starts empty and grows via `CorrectionAction`. Optional KG `(trap, fix)` rules injected into the prompt (`AGENT_USE_RULES=1`).

**Teacher is NOT called every hard problem.** Flow is:

```
Student (always) → Detector (windowed) → [only on drift] Teacher batch + write KG
→ later Student runs with examples + KG
```

Not: `hard Q → KG? → Teacher? → Student`.

---

## 4. Contracts (domain-agnostic)

Canonical names (new code should use these):

| Canonical | Legacy alias (still accepted on read) |
|-----------|----------------------------------------|
| `generated_output` | `generated_sql` |
| `correct_output` | `correct_sql` |
| `domain_id` | `db_id` |
| `invalid_output` | `invalid_sql` |
| `gold_output` / `broken_output` (FailingCase) | `gold_sql` / `broken_sql` |

Fixture JSON still uses Spider-shaped keys (`expected_sql`, `db_id`); adapters map at the boundary. For coding, `domain_id` = topic (`dp`, `graphs`, …).

See `contracts/schemas.py`, `.claude/rules/01-contracts.md`.

---

## 5. Coding domain (primary)

- Fixture: `fixtures/coding_subset.json` — **70** problems (35 easy / 32 hard / 3 extra)
- Topics: DP, arrays, graphs, strings, greedy — classic interview/LeetCode-style
- Student prompt: system “return only python fence” + optional few-shots (≤3 same-topic) + optional KG block + problem text
- Score: sandboxed unit tests (`harness/sandbox.py`) → `execution_accuracy` ∈ {0.0, 1.0}
- Feed (shared with Spider idea): easy baseline → hard LEARN (degraded) → hard HELD-OUT (recovery), LEARN⊥HELD-OUT via same-domain split

### Current models (Prime Inference)

| Role | Model | Notes |
|------|--------|------|
| Student | `meta-llama/Llama-3.2-3B-Instruct` | Hot path |
| Teacher | `minimax/minimax-m2.5` | Only on drift / compare arm |
| Distill | same as teacher by default | KG `(trap, fix)` |

Env: `PRIME_API_KEY` in `.env`. Script forces Prime and ignores local Ollama overrides in `.env`.

MiniMax **direct** API previously hit **402 insufficient balance** → teacher silently returned 0 verified examples. Fix: teacher via Prime (`TEACHER_USE_PRIME=1`, `correction/provider.py`). Always log teacher failures loudly.

---

## 6. Measured results (real runs — do not inflate)

### A. Standard `--full` coding (earlier success)

| | Acc |
|---|-----|
| WITHOUT | 0.273 |
| WITH (examples + KG) | **0.455** |
| **Δ** | **+0.182** |

Drift severity 0.320; 15 teacher / 1 gold / 2 anchor; 3 KG rules.

### B. Student+memory vs unaided teacher (`--compare-teacher`)

Same held-out hard, after a prior full run’s memory:

| Arm | Acc |
|-----|-----|
| Student + 18 examples + KG | **0.182** |
| Teacher alone | **1.000** |
| Gap | **−0.818** |

Per-Q: both✓=2, S>T=0, T>S=9.

### C. Hard-curriculum eval (`--hard-curriculum`) — intended product measurement

Pipeline:

1. Easy warmup **40** (detector baseline only — not teaching diet)
2. Hard LEARN **100** instances → drift → teacher few-shots + KG
3. Freeze memory
4. Held-out: WITHOUT / WITH, then **student+KG vs teacher**

Results (one seed):

| Stage | Result |
|-------|--------|
| Correction | 15 teacher / 2 gold / 2 anchor |
| KG | 3 rules |
| Student WITHOUT held-out hard | 0.111 |
| Student WITH memory | 0.111 (Δ **+0.000**) |
| Student+KG vs teacher | **0.111 vs 1.000** (gap **−0.889**) |

**Honest read:** Infrastructure works; **3B + topic-few-shots + short KG did not close the teacher gap** on this hard held-out slice. Earlier +0.182 self-lift shows lift is possible sometimes; curriculum seed did not replicate it.

---

## 7. How to run

```bash
# .env
# PRIME_API_KEY=...

bash scripts/use_prime_student.sh smoke       # student + teacher unit-test check
bash scripts/use_prime_student.sh probe       # cheap WITH/WITHOUT (~22 calls)
bash scripts/use_prime_student.sh full        # classic loop
bash scripts/use_prime_student.sh compare     # needs prior correction in events.jsonl
bash scripts/use_prime_student.sh curriculum  # hard-curriculum eval pipeline

# overrides
PRIME_AGENT_MODEL=... PRIME_TEACHER_MODEL=... N_LEARN=120 N_HELDOUT=40 \
  bash scripts/use_prime_student.sh curriculum
```

Orchestrator flags of note:

- `--adapter coding|spider|gsm8k`
- `--full`, `--fresh`, `--probe`, `--significance`
- `--compare-teacher`
- `--hard-curriculum --n-learn 100 --n-heldout 40`

Artifacts: `events.jsonl`, `correction/graph_store.json`.

Tests: `pytest` (~235+), hermetic, no live API.

**Git hygiene:** Do not add `Co-authored-by: Cursor`. Prefer commit-tree / disabled Cursor attribution. Author: Rohan Chavan `<rohanpc@vt.edu>`.

---

## 8. External research (ICL recovery limits) — summary

Synthesized for this project (weak ~3B code student + stronger teacher):

1. **Reflection gap:** Similar LeetCode few-shots often fail to transfer “insight”; near-direct cues help more than topic-similar demos.
2. **Few verified shots ≫ many** noisy ones; functional correctness can peak at modest shot counts.
3. **Failure-conditioned teaching:** Student fail → teacher refine *that* attempt → keep only unit-test–passing repairs (PersDistill-style) beats generic dumps.
4. **Compact verified concepts/rules** can beat raw few-shots for small models; val-filter rules; avoid large KG dumps.
5. **~3B hard-coding capacity floor** is real (e.g. Qwen2.5-Coder-3B LiveCodeBench Hard ~11% — same order as our 0.111).
6. **CoT / heavy agentic prompting** can *hurt* Pass@1 on small code models — keep prompts simple.
7. **Retrieval:** plan/algorithm similarity + impact filter > crude topic tags.

**ROI-ranked next experiments (from research + our null curriculum):**

| Rank | Experiment |
|------|------------|
| 1 | Failure-conditioned, unit-test–gated ICL bank; retrieve by algorithm/plan, not only topic |
| 2 | Concept/rule distillation from failure clusters; val-filter; inject short rules ± 1–2 exemplars |
| 3 | If ICL plateaus: LoRA / PersD on verified refinements only |

**Also cheap diagnostics before new models:**

- Injection audit: per held-out Q, how many same-topic examples actually entered the prompt?
- Ablate examples-only vs rules-only vs both (`AGENT_USE_RULES=0/1`)
- Stronger student (7–8B), same teacher — separates method failure vs capacity floor
- Online multi-correct during hard LEARN (memory grows mid-curriculum), then freeze + compare teacher

---

## 9. What Rohan wants next (product direction)

### Near term
1. Keep **coding** as the proving ground; **park Spider** as historical unless needed.
2. Treat the harness as **problem-agnostic OSS**: adapter plugin = load → run → score → teacher_verify → KG write/inject.
3. Make the **hard-curriculum + vs-teacher** eval the headline measurement story.
4. Fix **why memory doesn’t transfer** (retrieval + failure-conditioned examples + verified short rules) before adding domains or long-horizon agents.
5. Stay honest: publish nulls; don’t claim teacher-parity until measured.

### Medium term
1. Finish problem-agnostic polish (viewer labels, adapter docs, second thin domain e.g. GSM8K once coding lift is real).
2. Harden stats: multi-seed / McNemar on coding curriculum.

### Longer term (pivot)
**Long-horizon reasoning / multi-step agents:**

| Short-horizon (now) | Long-horizon (future) |
|---------------------|------------------------|
| 1 shot → 1 answer | plan → tools → trajectory → final |
| Score: unit test / EX | Score: task success + step metrics |
| Drift: answer accuracy | Drift: success rate, tool errors, loops |
| Teacher: repair answer | Teacher: repair **trace** / process |
| Few-shots: Q→A | Few-shots/KG: failure *patterns* |

Smallest honest B1: one long-horizon benchmark with automatic success; episode (+ optional step) telemetry; same drift → teacher → held-out WITH/WITHOUT (and vs-teacher) story.

**Do not yet:** rebuild on Spider, jump to unbounded “agents that code all day” without verifiable success, or add CUDA/RL before ICL/memory quality is fixed.

---

## 10. Known pitfalls

- `.env` may still set `AGENT_BASE_URL` to Ollama — `use_prime_student.sh` must force Prime.
- Sandbox/proxy in Cursor can 403 Prime — run with unrestricted network / unset proxy.
- `--fresh` clears `events.jsonl` + `graph_store.json`; compare without a prior learn phase is meaningless.
- Coding few-shot cap is **3** same-topic; topic filter can leave held-out Qs under-served.
- Detector needs ~40 easy baseline for warmup even in hard-curriculum (not for teaching).
- Fixture only has ~35 hard+extra unique problems; “100 hard LEARN” = sampling with replacement (instances, not 100 unique).

---

## 11. Key files to read first

1. `README.md` — current public story  
2. `orchestrator.py` — `run_full_loop`, `run_hard_curriculum_eval`, `compare_teacher_run`  
3. `adapters/coding.py` — student prompt, teacher solve/repair, KG write  
4. `harness/feed.py` — `build_stream`, `build_hard_curriculum_stream`  
5. `harness/sandbox.py` — unit-test scoring  
6. `correction/provider.py` — teacher endpoint resolution (Prime vs MiniMax)  
7. `contracts/schemas.py` — canonical field names  
8. `scripts/use_prime_student.sh` — operator entrypoint  
9. `docs/design.md` — older Spider dose-response / KG A/B detail  

---

## 12. Ask for Fable

Given the above:

1. Propose a **concrete next implementation plan** (smallest diff that can produce a measurable held-out self-lift and/or shrink teacher gap on coding).  
2. Call out what to **measure** (tables/commands) so results aren’t vibes.  
3. Explicitly say what **not** to build yet.  
4. If you recommend model changes, keep student cheap and teacher episodic unless evidence says capacity is the only blocker.

Prefer: failure-conditioned ICL + better retrieval, or a clear ablation plan — over new frameworks or long-horizon work before the short-horizon claim is solid.
