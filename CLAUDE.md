# CLAUDE.md — Reasoning RSI

> Project context for AI coding assistants working in this repo.
> Current as of 2026-07-25. If this file disagrees with the code, the code wins — fix this file.

## What we're building (one sentence)
**Runtime self-improvement for reasoning models without training:** a stronger *teacher* reads a
cheaper *student*'s graded failures, writes transferable lessons, and those lessons — frozen as
plain text prepended to the student's prompt — lift the student on **held-out** expert-reasoning
tasks. No fine-tuning, no RL, no weight access, no human in the loop.

The only thing that changes about the student is text in its context. That is the whole claim.

## The loop
```
BUILD (train split)                             EVAL (held-out split)
  student answers task                            PLAIN : student alone            → judge
  judge grades vs rubric → missed criteria        MEM   : student + frozen memory  → judge
  teacher(question, student answer, missed)       TEACHER: teacher alone (ceiling) → judge
    → ONE transferable lesson (entity-scrubbed)
  freeze lowest-scoring tasks' lessons          Δ = MEM − PLAIN, per task, averaged over k reps
    → runs/*_memory.json
```
**"Gaps-only" is the design that won:** the teacher needs only to see *where the student failed*,
not to solve the task. Adding the teacher's own worked answer as a third input did not help
(+2.5 vs +5.3; see `docs/updates/2026-07-21-prbench-corpfin-cto.md` §4a).

## Roles (three models, never collapsed)
| Role | PRBench track | FinancePro track | Job |
|---|---|---|---|
| Student | `deepseek/deepseek-v4-pro` (Fireworks-pinned) | `qwen/qwen3.6-27b` | The model being improved. Weights frozen. Never sees a rubric. |
| Teacher | `anthropic/claude-fable-5` | `z-ai/glm-5.2` | Reads graded failures, writes lessons. Train split only. |
| Judge | `openai/gpt-5.2` | `openai/gpt-5.2` | Grades against the official rubric → normalized 0–100. |

Judge ≠ teacher ≠ student, asserted in code (`correction/prbench_judge.grade`,
`correction/judge`). Everything runs on **OpenRouter**.

## Benchmarks
| Benchmark | Data | Split (seed 42) | Status |
|---|---|---|---|
| **PRBench** Corporate Finance (Scale AI, arXiv 2511.11562) | `fixtures/prbench_corpfin.json` — 93 tasks, weighted criteria, single + multi-turn | 50 train / 15 val / 28 held-out | **active** — `+5.3` on 12 held-out × k=3, ungated, no CI yet |
| **FinancePro-Bench** (`Sanscritic/finance-pro-bench`) | `fixtures/finance_pro_bench.json` — 400 questions, 33 categories | 200 / 80 / 120 | measured null (`+0.0` at k=3) — see `docs/FINDINGS_FINANCE.md` §H |

Earlier tracks (Spider text-to-SQL drift detection, coding, GSM8K) are **retired**; their code was
removed in `6935400` and their docs live in `docs/archive/`.

## Where things live
| Dir | Role |
|---|---|
| `adapters/` | `prbench.py`, `finance.py` — prompt assembly, rubric firewall, teacher calls, memory items |
| `correction/` | `prbench_judge.py`, `judge.py` (judges), `provider.py` (teacher client), `tracelift.py` (uplift gate) |
| `harness/agent.py` | student client: OpenAI-compatible calls, retry/backoff, **provider pin** |
| `scripts/` | the runnable experiments — freeze splits, build memory, eval arms |
| `analysis/bootstrap.py` | paired bootstrap CI + p-value (use it before quoting any delta) |
| `contracts/schemas.py` | shared Pydantic records (`FewShotExample`, `AgentConfig`, …) |
| `fixtures/` | cached datasets + frozen split manifests (committed on purpose) |
| `runs/` | **gitignored** — every result artifact is ephemeral. See rules/03 |
| `docs/archive/` | retired Spider / hackathon / v1-coding docs. Historical only |

Legacy modules from the drift-detection era still sit in `correction/` (`graph.py`, `on_drift.py`,
`memory.py`, `repair.py`, `distill.py`, `correction.py`, `learner.py`, `teacher.py`, `store.py`,
`contracts.py`) and `harness/` (`feed.py`, `evaluator.py`). They are not on the live path — see
`STRUCTURE.md` for the live/legacy map before importing anything.

## Shared rules
@.claude/rules/00-architecture.md
@.claude/rules/01-contracts.md
@.claude/rules/02-tech-decisions.md
@.claude/rules/03-research-integrity.md
@.claude/rules/04-workflow.md

## First thing, every session
1. Read the newest file in `docs/updates/` — that's the current result of record.
2. Check `runs/` for the artifacts you're about to rely on. **They may not exist** (gitignored,
   and past runs have been overwritten or lost). Never trust a number you cannot recompute.
3. `make test` before and after. One test is known-failing — see below.
4. Nothing single-pass is trusted: temp-0 noise on this stack is ±10–20 normalized points, so
   any delta needs **k ≥ 3** repeats averaged per task (rules/02).

## Run from repo root
```bash
make install                                        # pip3 install -r requirements.txt
make test                                           # pytest, hermetic, no API keys

# PRBench (active track) — OPENROUTER_API_KEY required for anything live
python scripts/prbench_freeze_splits.py --check     # verify split + dataset sha
python scripts/prbench_build_memory.py --n-train 12 --max-items 10
python scripts/prbench_memory_eval.py --k 3 --arms PLAIN,MEM,TEACHER
python scripts/prbench_memory_eval.py --summarize   # re-aggregate from cache, no API
python scripts/prbench_planner_eval.py --k 3        # A1/A2/A4-hints/A5 arms
```

## Known gaps (do not re-discover these)
Audited 2026-07-25; details and repro in `docs/updates/2026-07-21-prbench-corpfin-cto.md`
§"Corrections". Short version:
- **≤1 lesson actually reaches the prompt.** `select_category_memory` caps at 1 playbook, and every
  built lesson is tagged `[FINANCE_PLAYBOOK]`, so a 10-lesson store injects one item.
- **The `+5.3` score cells are gone** (`runs/prbench_memory_scores.jsonl` deleted; `runs/` is
  gitignored). The headline is not currently recomputable.
- **Deltas in the eval scripts are unpaired** (positional `[:n]` truncation), and
  `analysis/bootstrap.paired_bootstrap` has never been applied to a PRBench result.
- **19 detrimental rubric criteria are stored with positive weight** (`scripts/prepare_prbench.py`
  takes the first non-None weight field) — traps currently score as rewards.
- **The judge sees only the final turn**; 34 of 93 tasks are multi-turn.
- **The uplift gate is not wired for PRBench** — `correction/tracelift.py` needs `adapter.run_item`,
  which `adapters/prbench.py` does not implement. "Turn the gate on" is new code, not a flag.
- **One test fails in a full-suite run** (`tests/test_finance_memory.py::
  test_firewall_still_blocks_rubric_in_memory`): `scripts/finance_baselines.py` mutates
  `os.environ["AGENT_USE_EXAMPLES"]="0"` globally, so the rubric-firewall assertion goes vacuous.
  Passes in isolation. CI is red until it's fixed.
