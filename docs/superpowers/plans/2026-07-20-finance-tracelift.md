# FinancePro TraceLift Test — Work Order (Cursor)

> **CTO directive (2026-07-20):** one focused test. Student `qwen/qwen3.6-27b`,
> teacher `z-ai/glm-5.2`, judge `openai/gpt-5.2`, **OpenRouter only**, method =
> **TraceLift (uplift-gated ICL memory)**, **NO fine-tuning**. Question: does
> TraceLift memory improve the student on held-out finance questions?
>
> Branch `feat/finance-tracelift` off latest main. TDD, commit per task step,
> no co-author trailers. Entrypoint for all live runs:
> `bash scripts/use_openrouter_finance.sh <args>` (wires all three roles to
> OpenRouter). Master plan §CTO directive: `docs/RSI_MEM_V2_FINANCE.md`.

## Deliverable

A held-out comparison table:
- **A1** student alone
- **A4** student + frozen TraceLift memory  ← the claim
- **A5** GLM teacher alone (ceiling)
- (**A2** compute-matched retries — build if time; the honest bar, but A4-vs-A1 is the CTO ask)

on the 120 held-out questions, normalized rubric score, per-category + trap-hit
breakdown, paired bootstrap CI. Numbers verbatim → `docs/FINDINGS_FINANCE.md`.

## What already exists (reuse, don't rebuild)

- `correction/tracelift.py` — `estimate_uplift`, `select_uplift_memory`, `build_val_slice`, `uplift_enabled`, `memory_max_total`. Adapter-generic.
- `adapters/finance.py` — loader, manifest, rubric ACL (`rubric_for(role=...)`), student `run_item`, empty-content retry for reasoning models.
- `correction/judge.py` — rubric judge, JUDGE_PASSES gating, judge≠teacher assert.
- `scripts/finance_baselines.py` — A1/A5 eval harness, chunked+resume.
- `analysis/bootstrap.py` — paired bootstrap.

## Global constraints

- No `contracts/schemas.py` changes.
- Hermetic pytest (mock API); suite green.
- **[LIVE]** chunk ≤10 min, incremental JSONL, `--resume`.
- Teacher GLM 5.2 needs `max_tokens ≥ 4000` (reasoner truncates mid-thinking →
  empty content). `TEACHER_MAX_TOKENS=4000` is exported by the wrapper — make
  the teacher call read it.
- **Platform consistency:** discard Prime-era answer files
  (`runs/finance_heldout_*`, `runs/finance_headroom_*`). All arms regenerate on
  OpenRouter so no comparison mixes platforms.

---

### Task A — Finance teacher-repair → candidate memory items

**Files:** extend `adapters/finance.py`; test `tests/test_finance_memory.py`.

- `teacher_repair(qid, student_answer) -> str`: GLM 5.2, sees the question +
  the **train-stream rubric** (allowed via `rubric_for(role="teacher")` — assert
  qid ∈ train), produces a corrected answer. `max_tokens` from `TEACHER_MAX_TOKENS`.
- `distill_memory_item(qid, repaired) -> FewShotExample`: compress the repair
  into a category playbook / trap entry / ≤300-token skeleton (per plan §3),
  `domain_id = category`, `source = "tracelift"`. **Leakage guard:** strip named
  entities from the source question (reuse/extend the entity audit); a test
  asserts no held-out/validation question text leaks in.
- Commit: `feat: finance teacher-repair + memory-item distillation`

### Task B — Memory injection into the student prompt

**Files:** extend `adapters/finance.py` (`build_user_prompt`), tests.

- Inject ≤4 items (1 playbook + ≤2 traps + ≤1 skeleton), retrieved by
  **category match** (not embedding similarity — RA-RFT adapted), gated by
  `AGENT_USE_EXAMPLES`. Populate `injection_stats`.
- Student never sees rubrics (firewall test still passes).
- Commit: `feat: finance memory retrieval + injection (category-keyed)`

### Task C — TraceLift build loop (train-stream → uplift-gated frozen store)

**Files:** create `scripts/finance_tracelift.py`; tests.

- Pipeline: run student on TRAIN-STREAM → collect failures (judged low) →
  `teacher_repair` → `distill_memory_item` → **uplift-gate** each candidate via
  `tracelift.estimate_uplift` on the 80 VALIDATION questions (u = mean score
  with-item − without; keep u > +1 normalized pt, K=2) → compact (cap per plan)
  → freeze to a memory store file.
- **Stopping rule (plan §3):** freeze when window(15) mean uplift < +0.5 or
  admission rate < 20%. Log the saturation curve.
- [LIVE] chunked + resume. Report: #candidates, #admitted, u-distribution,
  stop iteration → `docs/FINDINGS_FINANCE.md`.
- Commit: `feat: finance TraceLift build loop with uplift gate + stopping rule`

### Task D — Held-out eval A1 / A4 (+A5, +A2 if time)

**Files:** extend `scripts/finance_baselines.py` (memory arm) or new
`scripts/finance_eval.py`; tests.

- A1 student-alone + A4 student+frozen-memory on 120 held-out, judged
  (JUDGE_PASSES per gate). A5 teacher = GLM 5.2. [LIVE] chunked+resume.
- Report GAP_alone = A4 − A1 with paired bootstrap CI + p; per-category; trap-hit
  rate A1 vs A4 (does memory cut trap penalties?). Verbatim → FINDINGS_FINANCE.md §E.
- Commit: `feat: held-out A1/A4 TraceLift eval + results`

---

## Reporting

Checkboxes + execution log below; pytest in commit bodies; [LIVE] numbers
verbatim into `docs/FINDINGS_FINANCE.md`; ambiguity → Questions section.
Commit per task step.

## Execution log
- Task A:
- Task B:
- Task C:
- Task D:

## Questions
(none yet)
