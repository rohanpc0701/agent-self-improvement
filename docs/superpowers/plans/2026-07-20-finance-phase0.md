# FinancePro-Bench Phase 0 Work Order (for Cursor execution)

> **Executor:** Cursor agent. Branch `feat/finance-phase0`. TDD per task; commit
> after every task step (no co-author trailers). Tick checkboxes + fill the
> execution log here. Master plan: `docs/RSI_MEM_V2_FINANCE.md` — its §1–§2
> (splits, rubric access policy, judge protocol) BIND every task below.

## First: close out v1

- [ ] On `feat/rsi-mem-phase0`: commit whatever exists as
  `wip: coding pool expansion (parked at v2 pivot)`, push nothing else, stop
  work there. The coding fixture/probe results stay — harmless, possibly useful
  for Phase 4b later.

## Global constraints

- No changes to `contracts/schemas.py`.
- Hermetic pytest (mock all API calls + dataset rows); suite stays green.
- **[LIVE]** steps: `PRIME_API_KEY` in `.env`, cost money, chunk ≤10 min,
  incremental JSONL + `--resume` everywhere.
- Models (Prime): judge default `openai/gpt-5.2`, teacher default
  `minimax/minimax-m3` — **judge ≠ teacher, hard assert in code.** Student
  candidates for G0.4 headroom probe: `qwen/qwen3-8b`,
  `qwen/qwen3-30b-a3b-instruct-2507`, `qwen/qwen3.6-27b` (all smoke-tested
  alive; user prefers Qwen family).
- Long questions (3–10k chars): student `max_tokens=2048`, judge
  `max_tokens=2048`, temperature 0 both.

---

### Task 1 — Dataset + adapter + frozen splits (G0.1)

**Files:** create `adapters/finance.py`, `scripts/finance_freeze_splits.py`,
`fixtures/finance_manifest.json`; tests `tests/test_finance_adapter.py`,
`tests/test_finance_splits.py`. Modify `adapters/__init__.py` (register).

1. [ ] Loader: pull `Sanscritic/finance-pro-bench` via HF `datasets` once,
   cache to `fixtures/finance_pro_bench.json` (id, category, question, rubric).
   Commit the cached file (CC-BY-4.0 — include attribution line in the file
   header or a `fixtures/FINANCE_LICENSE` note).
2. [ ] `scripts/finance_freeze_splits.py`: category-stratified 200/80/120
   (train-stream/validation/held-out), seed 42, writes
   `fixtures/finance_manifest.json` `{frozen_at, seed, dataset_sha256,
   train_ids, validation_ids, heldout_ids}`; `--check` verifies disjointness +
   sha. Every category must appear in train AND held-out where size permits
   (categories with <3 questions: train only — log them).
3. [ ] `adapters/finance.py` (TaskAdapter protocol, mirror `adapters/coding.py`
   structure): `load_questions`, `build_feed` (train-stream order, seeded),
   `run_item` (student answer → judged score → `TelemetryRecord` with
   `execution_accuracy = normalized_score/100`, `domain_id = category`,
   `injection_stats` populated). Judge call lives in Task 2's module — adapter
   imports it.
4. [ ] **Rubric firewall in code, not convention:** the student prompt builder
   takes only `question`; a test asserts the rubric string can never reach a
   student or teacher prompt for ids in `heldout_ids`/`validation_ids`
   (teacher + train-stream rubric is allowed).
5. [ ] Commit: `feat: finance adapter, dataset cache, frozen 200/80/120 splits`

**Acceptance:** `--check` exits 0; pytest green; manifest + dataset cache
committed at matching sha.

### Task 2 — Judge harness + reliability check (G0.2)

**Files:** create `correction/judge.py`, `scripts/judge_reliability.py`;
tests `tests/test_judge.py`.

1. [ ] `judge.py`: `grade(question, rubric, answer, model=JUDGE_MODEL, passes=1)
   -> {"total": float, "max": float, "normalized": float, "items": {"R1": pts, ...},
   "traps_hit": ["T2", ...], "bonuses": [...], "raw": str}` — prompt per plan §2
   (rubric verbatim, required output format), strict output parser with one
   repair-retry on parse failure, then hard error (no silent zeros). Uses
   `_chat_with_retry`. Assert `JUDGE_MODEL != TEACHER_MODEL` at import.
2. [ ] Parser tests on synthetic judge outputs: clean, malformed-then-repaired,
   trap lines, bonus lines, missing TOTAL (→ error).
3. [ ] **[LIVE]** `scripts/judge_reliability.py`: 40 stratified questions ×
   1 student answer each (generate once with `qwen/qwen3-8b`, cache to JSONL)
   → judge twice in fresh contexts → report test–retest Pearson r + MAD
   (normalized). Output table into `docs/FINDINGS_FINANCE.md` (new file).
   **Gate: MAD ≤ 5 → single pass; 5 < MAD ≤ 8 → set `JUDGE_PASSES=2` default;
   MAD > 8 → STOP, write to Questions section (K1 territory).**
4. [ ] Select 15 of the 40 for hand-audit: dump `(question, answer, judge
   breakdown)` to `runs/judge_audit_sample.md` for Rohan to eyeball — flag in
   the execution log when ready.
5. [ ] Commit: `feat: rubric judge harness + reliability check results`

**Acceptance:** reliability numbers in `docs/FINDINGS_FINANCE.md`, verbatim;
gate decision recorded; audit sample ready for human review.

### Task 3 — Headroom probe + baselines (G0.3, G0.4)

**Files:** create `scripts/finance_baselines.py`; extend
`analysis/bootstrap.py` if not yet built (paired bootstrap per v1 work-order
Task 3 spec — build it here if v1 never finished it); tests
`tests/test_finance_baselines.py`.

1. [ ] **[LIVE] Headroom probe (G0.4):** 3 student candidates × 20 stratified
   VALIDATION questions, bare prompt → judged. Pick the smallest model scoring
   15–40 normalized as THE student; record the table. If all <15 → try
   `qwen/qwen3.5-35b-a3b`; if all >40 → add smaller qwen. Write outcome to
   `docs/FINDINGS_FINANCE.md`.
2. [ ] **[LIVE] Baselines on HELD-OUT (120 Qs), chunked:** student-alone and
   teacher-alone arms, single pass each (both temp-0; note deterministic-or-not
   per model as measured), judged with the gated `JUDGE_PASSES`. Per-category
   breakdown + overall mean ± bootstrap CI into `docs/FINDINGS_FINANCE.md`.
   THIS IS THE ONLY HELD-OUT TOUCH ALLOWED in Phase 0 — no memory arms yet.
3. [ ] Tests: baseline script plumbing with mocked judge/adapter; bootstrap
   math (if newly built).
4. [ ] Commit: `feat: headroom probe + held-out baselines (student, teacher)`

**Acceptance:** `docs/FINDINGS_FINANCE.md` has: judge reliability table, student
band table + choice, held-out baseline table (student vs teacher, per-category),
all verbatim from logs under `runs/`.

---

## Reporting back

Same protocol as v1 work order: checkboxes, execution log lines, pytest output
in final commit body, [LIVE] numbers verbatim into `docs/FINDINGS_FINANCE.md`,
ambiguities → Questions section, do NOT improvise around plan §1–§2 rules.

## Execution log (Cursor fills in)

- v1 closeout:
- Task 1:
- Task 2:
- Task 3:

## Questions

(none yet)
