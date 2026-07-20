# Cursor → Fable session handoff (2026-07-20)

**Branch:** `feat/rsi-mem-phase0`  
**Executor:** Cursor (Rohan)  
**Upstream plans Fable already landed on this branch:**
- `docs/RSI_MEM_PLAN.md` (coding/RSI-Mem v1, 24-week)
- `docs/superpowers/plans/2026-07-20-rsi-mem-phase0-1.md` (coding G0.1 / G1.1 / G0.2 work order)
- `docs/RSI_MEM_V2_FINANCE.md` + `docs/superpowers/plans/2026-07-20-finance-phase0.md` (commit `f2713e4`)

**Product brief (older, still useful):** `docs/FABLE_HANDOFF.md`

---

## 1. What Cursor was executing

Fable’s coding work order: **RSI-Mem Phase 0–1** (`2026-07-20-rsi-mem-phase0-1.md`)

| Task | Goal | Cursor status |
|------|------|----------------|
| **Task 1 G0.1** | Expand hard pool ≥240, freeze 200/learn/30 validation manifest, wire `HELDOUT_MANIFEST` | **Partial — not committed** |
| **Task 2 G1.1** | Per-item uplift audit + live Qwen run | Hermetic core done; **LIVE blocked** on missing manifest |
| **Task 3 G0.2** | 3-arm multi-seed eval + paired bootstrap | Hermetic + `--dry-run` done; **not committed** |

Then Rohan pasted **RSI-Mem v2 (FinancePro-Bench)**. Cursor **stopped the coding probe** and asked A/B/C pivot strength. **No answer yet** when this handoff was written.

---

## 2. Uncommitted work on disk (coding Phase 0–1)

Do **not** lose these; none of this is in a commit yet:

### Task 1 (G0.1) — mostly ready except LIVE freeze
- `scripts/import_problems_probe.py` — chunked `--offset/--limit`, resume via `runs/probe_results.jsonl`, `--no-merge`, `max_keep=400`
- `scripts/import_problems.py` — CLI args wired
- `scripts/freeze_heldout.py` — topic-stratified 200/learn/30, Jaccard near-dupe drop, `--check`
- `harness/feed.py` — `HELDOUT_MANIFEST` loads frozen learn/heldout (also continuous stream)
- `harness/agent.py` — `AGENT_TIMEOUT_S` (default 90) to stop infinite hangs on Prime
- `tests/test_freeze_heldout.py` — **6 passed**
- `.gitignore` — `runs/` already ignored

**LIVE probe (paused intentionally for v2 pivot):**
- File: `runs/probe_results.jsonl` (**228 / 448** candidates)
- Hard candidates (pass-rate ≤ 0.5): **~183**
- Fixture still **75 hard** (merge never ran → no `fixtures/heldout_manifest.json`)
- Resume: `python3 -m scripts.import_problems probe --k 2 --temperature 0.7 --max-keep 400 --offset N --limit 50 --no-merge` then final merge without `--no-merge`
- Student: Prime `meta-llama/Llama-3.2-3B-Instruct`, k=2 @ 0.7

**Blocker for freeze:** need hard pool ≥ 230 (200 held-out + 30 validation). Probe yield is high (~80% hard); finishing remaining ~220 probes then merge should clear ≥240.

### Task 2 (G1.1) — hermetic only
- `scripts/uplift_audit.py` — `audit_plan`, `summarize`, resume JSONL, `--items`, `--dry-run`
- `tests/test_uplift_audit.py`
- LIVE not run (needs manifest + `Qwen/Qwen3.5-4B` on archived `runs/qwen35-4b_artifacts/events.jsonl`)

### Task 3 (G0.2) — hermetic only
- `analysis/bootstrap.py` — `paired_bootstrap`
- `scripts/eval_runner.py` — matrix + summary + `--dry-run --seeds 3` (live eval deliberately exits: waits gated pipeline)
- `tests/test_bootstrap.py`, `tests/test_eval_runner.py`
- Combined with freeze/uplift tests: **19 passed** last check; dry-run OK

**Author/commit hygiene Rohan wants:** Rohan Chavan `<rohanpc@vt.edu>`, **no** Co-Authored-By trailers (use `commit-tree` if needed).

---

## 3. Honest results already on `main` / prior commits (don’t re-litigate)

From earlier sessions (see `docs/FINDINGS_CODING.md`, `docs/FINDINGS_REASONING.md`):

- Coding ablation: 3B memory **doesn’t help**; injection works; capacity (`qwen/qwen3-coder`) ≫ 3B+memory
- Nemo variance: mean Δ≈0; small protocol deltas need ≥3 repeats
- GSM8K uplift-gate: Llama-3.2-3B in band; curriculum +0.083 then frozen ablation **harm** (examples 0.179 vs none 0.321); topic filter zero_inj high
- KG exists but often not on critical path of polished demo

---

## 4. RSI-Mem v2 (FinancePro) — Fable already committed the plan

Commit `f2713e4`:
- Master: `docs/RSI_MEM_V2_FINANCE.md`
- Work order: `docs/superpowers/plans/2026-07-20-finance-phase0.md`

**North star (v2):** `student + frozen memory > student alone` on FinancePro-Bench held-out, normalized rubric score; memory from train-stream failures only.

**Binding constraints Cursor has internalized:**
- 400 Q → freeze **200 train / 80 val / 120 held-out** stratified before anything else
- Rubric LLM judge + reliability gate (MAD ≤ ~5) is an **exit blocker**
- Placebo (style-only) arm required
- Memory = category playbooks + **trap registry** + skeleton exemplars (not raw long few-shots)
- Deep reasoning ≠ long-horizon interactive (ALFWorld is a separate claim)

**Cursor ask to Rohan (unanswered):** pivot strength  
- A) Finance primary; coding frozen on disk  
- B) Parallel (finish coding freeze + start Finance Phase 0)  
- C) Commit coding hermetic WIP, new branch `feat/financepro-phase0`  
Cursor recommended **A or C**.

---

## 5. What Fable should decide / do next

1. **Confirm pivot (A/B/C)** with Rohan.
2. If coding is abandoned as primary: either leave WIP uncommitted or have Cursor commit “WIP: coding G0.1 tooling + probe checkpoint note” so hermetic scripts aren’t lost — **do not** claim a frozen 200 held-out (manifest doesn’t exist).
3. If Finance is next: execute `2026-07-20-finance-phase0.md` starting **G0.1 split freeze** — Cursor is ready to implement against Fable’s work order (same tandem: Fable plans, Cursor executes).
4. Do **not** invent Finance baselines until judge reliability (G0.2) passes.

---

## 6. Ops notes for whoever resumes

- Repo root imports; `.env` has `PRIME_API_KEY` / `OPENROUTER_API_KEY` (never commit).
- `pip3`; prefer Prime for student probes.
- Subagents hit **Cursor model API limits** repeatedly; long LIVE jobs were run as **background shells** instead.
- Probe hung once without timeout → fixed via `AGENT_TIMEOUT_S=90`.
- Terminal cwd sometimes drifted to `~/`; always `cd` to repo absolute path.

---

## 7. One-paragraph status for Fable

Cursor executed most of the **coding** Phase 0–1 work order on `feat/rsi-mem-phase0`: chunked probe + freeze tooling + `HELDOUT_MANIFEST` feed wiring + uplift audit + bootstrap/eval_runner hermetic tests are on disk but **uncommitted**; LIVE hard-pool probe paused at **228/448 (~183 hard)** with **no merge and no held-out manifest**. Fable has since landed **RSI-Mem v2 FinancePro** docs/work order (`f2713e4`). Cursor stopped coding API burn and is waiting on Rohan’s pivot decision before either finishing the coding freeze or starting Finance Phase 0 G0.1.
