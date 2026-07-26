# RSI-Mem Phase 0–1 Work Order (for Cursor execution)

> **Executor:** Cursor agent. Work on branch `feat/rsi-mem-phase0`. TDD per task:
> failing test → implement → `python3 -m pytest -q` green → commit (NO
> Co-Authored-By trailers of any kind). Tick the checkboxes in this file as you
> go. All commands from repo root. Read `docs/RSI_MEM_PLAN.md` §2 (honesty
> rules) before starting — they bind every task here.

**Goal:** plan items G0.1 (frozen ≥200 held-out), G1.1 (per-item uplift audit),
G0.2 (3-arm × 3-seed runner with paired bootstrap).

**Context files:** `docs/RSI_MEM_PLAN.md` (master plan + status appendix),
`docs/FINDINGS_CODING.md` (what's already measured), `scripts/import_problems.py`
+ `scripts/import_problems_probe.py` (existing import), `scripts/variance_check.py`
(machinery to extend), `orchestrator.py` (`run_ablation_eval`, `_mcnemar_report`).

**Global constraints**
- `contracts/schemas.py`: no changes in this work order.
- Tests hermetic — no live API in pytest (~260 pass currently; stay green).
- Live-API steps are marked **[LIVE]**; they need `PRIME_API_KEY` in `.env` and
  cost money — run them once, save logs under `runs/`, never in CI.
- Student for uplift audit: `Qwen/Qwen3.5-4B` (deterministic on Prime — cleanest
  signal). Archived memory: `runs/qwen35-4b_artifacts/events.jsonl`.
- Long live runs: chunk to ≤10 minutes per invocation (background jobs in some
  sessions get killed); every script must write results incrementally to disk
  (JSONL append), so a killed run resumes instead of restarting.

---

### Task 1 — G0.1: Expand and FREEZE the held-out set (≥200 hard)

**Files:** modify `scripts/import_problems.py`, `scripts/import_problems_probe.py`;
create `scripts/freeze_heldout.py`, `fixtures/heldout_manifest.json`;
test `tests/test_import_problems.py`, `tests/test_freeze_heldout.py`.

1. [ ] Uncap the import: `--max-keep` default → 400; fetch stage keeps ALL
   sandbox-validated candidates (MBPP+ ~399 + HumanEval+ ~164 → several hundred).
2. [ ] **[LIVE]** Re-run `fetch`, then `probe` with `--k 2 --temperature 0.7`,
   keep pass-rate ≤ 0.5, `--max-keep 400`. **Chunk it**: add `--offset/--limit`
   args so each invocation probes ≤100 candidates and appends results to
   `runs/probe_results.jsonl` (resume = skip ids already present). Target ≥240
   hard total in the fixture (existing 75 + new). If the EvalPlus pool tops out
   short of 240, stop and report the number — LCB import becomes a follow-up
   decision, do NOT improvise a new dataset.
3. [ ] `scripts/freeze_heldout.py`: deterministic split of the hard pool —
   **200 held-out / rest LEARN** (topic-stratified, seed 42), writes
   `fixtures/heldout_manifest.json`:
   `{"frozen_at": iso8601, "seed": 42, "fixture_sha256": ..., "heldout_ids": [...], "learn_ids": [...], "validation_ids": [...]}`
   — `validation_ids` = 30 problems carved from LEARN (never held-out) for
   uplift measurement (Task 2). Disjointness: held-out ∩ learn ∩ validation = ∅.
4. [ ] Leakage guard: near-duplicate check between held-out and learn questions
   (normalized-token Jaccard > 0.8 → drop the LEARN copy; log what was dropped).
5. [ ] Wire the harness: `harness/feed.py` / coding adapter read the manifest
   when `HELDOUT_MANIFEST=fixtures/heldout_manifest.json` is set, instead of
   re-splitting. Tests: manifest round-trip, disjointness, determinism,
   Jaccard dedupe on synthetic near-dupes, feed respects manifest.
6. [ ] Commit(s): `feat: expand hard pool and freeze 200-problem held-out manifest`

**Acceptance:** `python3 scripts/freeze_heldout.py --check` exits 0, prints
pool sizes; pytest green; manifest committed WITH the fixture at the same sha.

---

### Task 2 — G1.1: Per-item uplift audit (tests prediction P1)

**Files:** create `scripts/uplift_audit.py`, `tests/test_uplift_audit.py`.

Definition: for each memory item i in a frozen bundle,
`u_i = mean_pass(validation, memory = {i}) − mean_pass(validation, memory = ∅)`
at temperature 0, on the 30 `validation_ids` (NEVER the held-out set). Also
measure `u_full` for the whole bundle and `u_loo(i)` (leave-one-out: full minus
item i) for the top-3 most negative items.

1. [ ] Pure core, hermetic-testable:
   `audit_plan(bundle, validation_ids) -> list[RunSpec]` and
   `summarize(results) -> {"per_item": {...}, "u_full": float, "n_pos": int, "n_neg": int, "p1_verdict": str}`
   — P1 verdict: fraction of items with u ≤ 0; ≥30% → "P1 CONFIRMED".
2. [ ] Runner reuses `variance_check.run_arm` machinery (import, don't copy);
   incremental JSONL output `runs/uplift_audit_<model>.jsonl`; `--resume` skips
   completed (item, question) cells; chunkable via `--items 0-5`.
3. [ ] Tests: audit_plan cell count = (|bundle|+1) × |validation|; summarize
   math on synthetic results; resume logic (pre-seeded JSONL → only missing
   cells planned).
4. [ ] **[LIVE]** Run on the archived qwen bundle (11 items):
   `python3 scripts/uplift_audit.py --model Qwen/Qwen3.5-4B --events runs/qwen35-4b_artifacts/events.jsonl --manifest fixtures/heldout_manifest.json`
   (12 arms × 30 validation Qs = 360 calls; chunk by `--items`). Save the
   summary table into `docs/FINDINGS_CODING.md` under a new section
   "G. Per-item uplift audit" — verbatim numbers, include the u ≤ 0 fraction
   and whether P1 held.
5. [ ] Commit: `feat: per-item uplift audit (G1.1) + qwen bundle results`

**Acceptance:** summary prints per-item u sorted ascending; P1 verdict line;
findings doc updated; pytest green.

---

### Task 3 — G0.2: Three-arm, multi-seed eval runner + paired bootstrap

**Files:** create `scripts/eval_runner.py`, `analysis/bootstrap.py`,
tests `tests/test_bootstrap.py`, `tests/test_eval_runner.py`.

1. [ ] `analysis/bootstrap.py`: `paired_bootstrap(a: list[float], b: list[float], n_boot=10_000, seed=0) -> {"delta": float, "ci_low": float, "ci_high": float, "p_value": float}`
   (resample problem indices; two-sided p for Δ≠0). Hermetic tests against
   known synthetic cases (all-equal → p≈1; disjoint → p≈0).
2. [ ] `scripts/eval_runner.py`: three arms (student-alone / student+memory /
   teacher-alone) × N seeds over the manifest held-out set.
   **Seed semantics (decide once, document in the script docstring):** a seed
   controls the LEARN-stream sampling + teacher generation that BUILDS the
   memory bundle (3 independent memory builds); held-out set and decoding
   (temp 0) stay fixed. Student-alone and teacher-alone run once (deterministic)
   and are reused across seeds.
3. [ ] Output: `runs/eval_<timestamp>/` with per-problem JSONL per arm per seed
   + `summary.json` (per-seed GAP, mean GAP, paired-bootstrap CI/p pooled per
   seed and across seeds, teacher-gap-closed %, tokens per arm). Incremental
   writes + `--resume`.
4. [ ] Tests: seed plumbing (mocked adapter → distinct bundles per seed, shared
   baseline arms), summary math, resume.
5. [ ] Commit: `feat: 3-arm multi-seed eval runner with paired bootstrap (G0.2)`

**Acceptance:** `python3 scripts/eval_runner.py --dry-run --seeds 3` prints the
run matrix without API calls; pytest green. (The full [LIVE] Phase-2 evaluation
run is NOT part of this work order — it waits for the gated pipeline.)

---

## Reporting back (every task)

- Update checkboxes here + one-line result per task at the bottom of this file.
- `python3 -m pytest -q` output pasted into the final commit message body.
- [LIVE] runs: log files under `runs/`, key numbers copied verbatim into
  `docs/FINDINGS_CODING.md`.
- Anything ambiguous or broken: STOP and leave a `## Questions` note at the
  bottom of this file rather than improvising around the honesty rules.

## Execution log (Cursor fills in)

- Task 1:
- Task 2:
- Task 3:

## Questions

(none yet)
