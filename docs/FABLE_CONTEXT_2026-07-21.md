# Full Context for Reasoning About Next Steps — 2026-07-21

Purpose: everything needed to think clearly about where this project goes next.
Written after a long session that ended in an honest null. Prefer honest nulls
over optimistic claims. Numbers here are from real runs.

---

## 1. The thesis under test

A weak **student** model fails on hard work → a stronger **teacher** turns those
failures into **memory** (few-shot examples + rules) → the **same student**, with
that frozen memory in its prompt, does better on **new held-out** problems.
**No fine-tuning, no model swap** — the improvement must come from what the agent
learned (the memory), delivered in-context.

The specific method being tested is **TraceLift-style uplift gating**: only write
a memory item if it *measurably improves* the student on a validation slice
(`u = score_with_item − score_without_item > threshold`). This is our novel
mechanism — no paper in the reference set gates memory writes by causal uplift.

Reference papers shaping the design: Memp (memory helps at scale; rules/scripts
generalize better than examples), TraceLift (verified ≠ useful; gate by uplift),
RA-RFT (retrieve by strategy not similarity), SelfMem (keep memory small),
OPHSD/SIA (if prompting stalls, distill to weights), RALPH→Zenith (memory must
beat compute-matched retries; needs a write-stopping rule).

---

## 2. How we got here (the journey — so you don't re-litigate)

- **Coding domain (v1 proving ground): CLOSED.** Text-to-SQL → coding adapter.
  Result: memory doesn't reliably help a 3B student; capacity floor real; the
  same memory bundle was 0.000 effect on mistral-nemo (4 repeats) and a
  *deterministic −26.5 pts* on Qwen3.5-4B (3 repeats). KG rules inert. This is
  where the **variance protocol** was born: temp-0 provider nondeterminism means
  deltas below the noise floor need ≥3 repeats before you believe them.
- **GSM8K reasoning trial:** built the actual uplift gate (`correction/tracelift.py`).
  Gate worked mechanically (admitted +u items, dropped −u). But held-out transfer
  still failed — curriculum +0.083 then frozen ablation showed *harm*; 43% of
  held-out questions got zero memory injected (topic-retrieval miss). Lesson:
  gating is necessary but not sufficient; retrieval + student capacity matter.
- **ALFWorld long-horizon: PARKED.** Env built and working (venv, patched
  Inform7-free TextWorld, 134 valid_unseen games load, ReAct runner). Band sweep
  found small models floor (nemo 0.033, qwen3.5-4b 0.000, gpt-oss-20b 0.100).
  Superseded by the finance pivot before a full run.
- **RSI-Mem v2 → FinancePro-Bench (current).** Rubric-graded expert finance
  reasoning, 400 questions, 33 categories. Then the **CTO narrowed it**
  (2026-07-20): one focused test — GLM 5.2 teacher, Qwen 3.6 27B student,
  does TraceLift improve? No fine-tuning.
- **Platform: migrated Prime → OpenRouter** (CTO requirement). Same model slugs.

---

## 3. Current setup (fixed by CTO)

- **Student:** `qwen/qwen3.6-27b` (reasoning DISABLED via OpenRouter
  `reasoning.enabled=false` — critical fix, 6× faster, real content instead of
  empty). Mid-band on finance (~26/100 unaided).
- **Teacher:** `z-ai/glm-5.2` (heavy reasoner; needs ≥4000 max_tokens or content
  comes back empty). Repairs failures + distills memory.
- **Judge:** `openai/gpt-5.2` (must differ from teacher — self-preference bias;
  hard-asserted in code). Grades against official rubrics.
- **Platform:** OpenRouter only. Entrypoint `scripts/use_openrouter_finance.sh`.
- **Data:** 200 train / 80 validation / 120 held-out, category-stratified,
  seed 42, frozen (`fixtures/finance_manifest.json`). Rubric firewall enforced
  in code + tests (student never sees rubrics; teacher only train-stream).

### CTO constraints (hard)
- **No fine-tuning** (rules out LoRA/SFT/RL for now).
- One implementation at a time.
- OpenRouter, GLM 5.2 teacher, Qwen 3.6 27B student.

---

## 4. The evidence base (all measured, honest)

### Coding (3 students)
| student | band | memory effect |
|---|---|---|
| Llama-3.2-3B | floor 0.10 | −0.07 ns |
| mistral-nemo | in-band 0.53 | 0.000 (4-repeat) |
| Qwen3.5-4B | in-band 0.59 | **−0.265 deterministic** (3-repeat) |

### GSM8K
Uplift gate worked; held-out transfer failed (curriculum +0.083 → frozen
ablation harm; 43% zero-injection).

### Finance TraceLift (the headline result, `docs/FINDINGS_FINANCE.md` §G-H)
Eval on the 7 Credit/Trading held-out questions where category-memory injects.

| memory | single-pass GAP (A4−A1) | k=3 averaged GAP |
|---|---:|---:|
| boilerplate distillation | −6.1 (hurt) | — |
| good distillation | **+5.6** | **+0.0** (n=6) |

The single-pass +5.6 was **noise**: per-question k=3 deltas were
−0.8, +6.9, +2.9, −7.9, −13.2, +12.3 — the biggest single-pass driver
(fpb-00380, +17.9) reversed to −13.2 under repeats. **Averaged: clean null.**

### The through-line
**Three domains agree: teacher-built in-context memory does not reliably improve
this class of student once noise is averaged out.** The pipeline runs; the
memory doesn't transfer.

---

## 5. What actually works vs what's fragile (infra reality)

**Works (verified):**
- Full pipeline mechanics: student gen → GLM repair → distill → inject → judge.
- **Distillation-quality fix** (`adapters/finance.py::_teacher_distill`, commit
  4d81ef6): teacher writes transferable content directly (ASC/IFRS-grounded,
  leak-safe) instead of the old extract-then-scrub that produced generic
  boilerplate. Boilerplate memory HURT (−6.1); good memory is at least neutral.
- **Judge TOTAL-fallback** (reconstruct total from item breakdown when the judge
  omits the TOTAL line — gpt-5.2 does this intermittently). NOTE: this fix has
  been reverted at least once by a branch switch — CHECK it's present.
- Reasoning-disable for qwen (the single biggest perf unlock).
- Judge reliability validated on Prime (MAD ~4.2) — NOT re-validated on OpenRouter.

**Fragile / broken (cost us most of the session):**
- **The TraceLift build harness (`scripts/finance_tracelift.py`)** has stacked
  bugs: state doesn't persist between chunks (restarts from done=0), the uplift
  gate "considers 0 candidates" even when candidates exist, bare-baseline
  recomputes every chunk. We ended up **bypassing the gate** and building
  ungated memory directly. THE GATE HAS NEVER ACTUALLY RUN TO COMPLETION on
  finance.
- **Judge flakiness during gating**: empty outputs / missing TOTAL / some
  questions have rubrics without `Item R*(max N)` (unscoreable — e.g. fpb-00108).
- **Branch thrashing**: this session's working dir kept getting switched between
  `feat/finance-tracelift` and `feat/finance-phase0` (Cursor git ops), deleting
  scripts mid-run. Work is committed on `feat/finance-tracelift`.
- **This session's environment kills long background jobs** (~10 kills, even
  nohup). Long runs must go on a stable machine / Cursor's environment.
- **Measurement noise is brutal**: ±10-20 normalized pts per question at temp 0.
  A real +4 effect is below the noise floor of a single pass.

---

## 6. The untested levers (this is where thinking should focus)

1. **The uplift gate has never actually been tested end-to-end.** Every finance
   result used UNGATED memory (all candidates injected). The whole TraceLift
   thesis is "keep only items that measurably help." Ungated-null ≠ gated-null.
   A working gate might filter 10 items → the 2 that help. **This is the single
   biggest hole.** Fixing the harness (or rebuilding the gate cleanly, direct
   like the bypass scripts in `/tmp`) is high-value.
2. **Measurement can't see a +4 yet.** n=6, no CI, single-pass noise ±15. Need:
   bigger held-out (all categories, ~40+ Qs), ≥3 repeats + paired bootstrap
   (`analysis/bootstrap.py` exists), 2-pass judge. Without this, no method result
   is provable. **Neither this nor the gate needs RL or fine-tuning** — both are
   frozen-student inference + bookkeeping.
3. **Retrieval is category-only (coarse).** RA-RFT says retrieve by
   strategy/predicted-trap, not bucket. The trap registry was never used as a
   *retrieval key*.
4. **Procedure vs knowledge (OPHSD diagnostic).** Is the teacher's edge
   promptable reasoning procedure, or finance knowledge the 27B lacks? Compare
   uplift on questions that quote the governing standard vs those that don't.
   **If it's a knowledge gap, in-context memory fundamentally can't fix it** —
   and that's the evidence-based case for weights (which CTO ruled out).
5. **Compute-matched retries baseline (RALPH).** Even if memory helps, it must
   beat spending the same tokens on best-of-k fresh attempts. Never measured.

---

## 7. The honest strategic picture

- Three nulls + "the research points to weights" is starting to say the real
  answer for a ~27B student on expert reasoning is **LoRA/SFT distillation** of
  the teacher's repairs — which the CTO ruled out for this test.
- BUT the in-context path is **not yet fairly tested**: the gate never ran, the
  measurement can't resolve a +4, retrieval is coarse. A clean gated + properly-
  measured run is the honest thing to do before declaring in-context dead.
- Recommendation on the table: do measurement-fix + working-gate together as one
  milestone (both inside "no fine-tuning"), re-run gated A1-vs-A4 on a real
  held-out with CIs. If null → in-context is genuinely done for this student, and
  the procedure-vs-knowledge diagnostic says whether to reopen the weights option.

---

## 8. Key files
- `docs/RSI_MEM_V2_FINANCE.md` — master plan (v2.1, arms, kill criteria, CTO directive).
- `docs/FINDINGS_FINANCE.md` — all finance numbers (§A judge, §C headroom, §G-H TraceLift result).
- `docs/FINDINGS_CODING.md`, `docs/FINDINGS_REASONING.md` — coding + GSM8K nulls.
- `docs/updates/2026-07-21-tracelift-finance-cto.md` — the CTO summary (GAP +0.0).
- `adapters/finance.py` — student gen, teacher repair, `_teacher_distill`, injection, rubric firewall.
- `correction/tracelift.py` — uplift gate (`estimate_uplift`, `select_uplift_memory`).
- `correction/judge.py` — rubric judge + TOTAL fallback (verify present).
- `scripts/finance_tracelift.py` — build harness (BUGGY — state/gate).
- `scripts/finance_eval.py` — A1/A4/A5 eval harness.
- `scripts/use_openrouter_finance.sh` — OpenRouter entrypoint.
- `runs/finance_memory_good.json` — the 10 good-distillation memory items.
- Branch: `feat/finance-tracelift` (all current work; keep off other branches).

## 9. Operational gotchas
- qwen3.6-27b: set `reasoning.enabled=false` or it burns 18k reasoning tokens → empty.
- GLM 5.2: needs ≥4000 max_tokens or empty content.
- Judge ≠ teacher (asserted). gpt-5.2 intermittently drops the TOTAL line → need the fallback.
- Long jobs: run on a stable machine; this session's env kills them.
- Keep the branch pinned; do not switch mid-run.
- `.env` has `OPENROUTER_API_KEY` (+ `PRIME_API_KEY`/`PRIME_TEAM_ID`, now unused). Never commit.
