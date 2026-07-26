# Repo structure — live path vs legacy

Two eras share this tree. The **live** era is teacher-built frozen memory on rubric-graded reasoning
benchmarks (PRBench, FinancePro). The **legacy** era is the retired Spider text-to-SQL drift-detection
loop; its top-level stages (`detector/`, `viewer/`, `orchestrator.py`) were deleted in `6935400`, but
some modules survive because tests still cover them.

Before importing anything from `correction/` or `harness/`, check which column it's in.

## Live path

| Path | Role |
|---|---|
| `adapters/prbench.py` | **active track.** Prompt assembly (system + memory + turns), rubric ACL, teacher hints, contrastive memory items, self-refine loop, teacher-alone ceiling |
| `adapters/finance.py` | FinancePro track + the shared helpers PRBench reuses: `select_category_memory`, `memory_kind_of`, `extract_named_entities` / `strip_named_entities`, `_token_trim`, `_teacher_distill` |
| `correction/prbench_judge.py` | weighted-criteria judge → normalized 0–100 + the `missed` list that drives memory building |
| `correction/judge.py` | FinancePro rubric judge (`Item R*(max N)` format) |
| `correction/provider.py` | teacher/distill client resolution (OpenRouter / Prime / MiniMax) |
| `correction/tracelift.py` | uplift gate: keep a candidate only if it measurably helps the frozen student. Wired for the finance adapter; **not** for PRBench (needs `run_item`) |
| `harness/agent.py` | student client: OpenAI-compatible calls, retry/backoff, OpenRouter provider pin + fallback accounting |
| `analysis/bootstrap.py` | `mean_bootstrap`, `paired_bootstrap` — CI and p-value for a delta |
| `contracts/schemas.py` | `FewShotExample`, `AgentConfig` (+ retired records, see rules/01) |
| `core/adapter.py` | `TaskAdapter` protocol — the seam a new benchmark implements |
| `scripts/prbench_*.py` | prepare data → freeze splits → build memory → eval (`memory_eval`, `planner_eval`) |
| `scripts/finance_*.py` | same shape for FinancePro, plus the gated build loop (`finance_tracelift.py`) |
| `scripts/eval_runner.py`, `scripts/freeze_heldout.py` | multi-seed eval + held-out freezing helpers |
| `fixtures/` | cached datasets, licenses, frozen split manifests — **committed on purpose** |

## Legacy (retired drift-detection era — do not build on)

| Path | Why it's still here |
|---|---|
| `harness/feed.py` | `FeedItem` is still the input type for `correction/tracelift.py` and the finance adapter's `run_item` |
| `harness/evaluator.py` | SQL execution accuracy; imported only by `correction/learner.py` |
| `correction/learner.py`, `teacher.py` | few-shot construction + anchoring from the SQL era; reached via `core/adapter.py` and the finance adapter's legacy surface |
| `correction/graph.py`, `store.py`, `inject.py` | knowledge-graph `(trap, fix)` rules. `inject.py` is still imported by `harness/agent.py`; the graph itself is off the live path |
| `correction/correction.py`, `on_drift.py`, `memory.py`, `repair.py`, `distill.py`, `contracts.py` | drift→correction plumbing. No live importers; kept for their tests and history |
| `contracts/eventlog.py` | append-only `events.jsonl` helper — reads historical logs only |
| `fixtures/generate_mocks.py`, `fixtures/mock_*.jsonl` | mock telemetry/drift/event fixtures for the retired stages' tests |
| `events.jsonl`, `events.jsonl.bak*` | historical run logs (gitignored) |

Deleting legacy modules means deleting their tests too — do it deliberately, in its own PR, not as a
drive-by.

## Conventions

- **Run from the repo root.** Imports assume it (`from contracts.schemas import …`); scripts insert
  the root on `sys.path` themselves.
- `runs/` is gitignored. Nothing there is a durable record — see `.claude/rules/03-research-integrity.md`.
- Per-stage notes live in `harness/CLAUDE.md` and `correction/CLAUDE.md`.
- Private session notes go in `CLAUDE.local.md` (gitignored).

## Start here

1. `make install`
2. `make test` — hermetic; one known failure is documented in `CLAUDE.md` "Known gaps".
3. Read `CLAUDE.md`, then the newest file in `docs/updates/`.
