# LOCKED RULE: Architecture & data flow

**Do not change without agreement.** This defines how a run is wired.

## The pipeline
```
BUILD (train split)                                    EVAL (held-out split)
 ┌─────────────────────────────────────────┐            ┌──────────────────────────────┐
 │ student answers task                    │            │ PLAIN   student alone        │
 │ judge grades vs rubric → missed criteria│  frozen    │ MEM     student + memory     │
 │ teacher(task, answer, missed) → lesson  │  memory →  │ REFINE  self-critique×3      │
 │ entity-scrub → runs/<track>_memory.json │            │ TEACHER teacher alone        │
 └─────────────────────────────────────────┘            └──────────────────────────────┘
                                                          all arms → judge → normalized 0–100
                                                          Δ = MEM − PLAIN, per task, k reps
```

## The three seams (the only coordination points)
1. **Judge → teacher:** the list of *missed rubric criteria* for a train task. That list — not the
   task, not a gold answer — is what the teacher reasons about. ("Gaps-only.")
2. **Teacher → memory:** one `FewShotExample` per admitted lesson, written to
   `runs/<track>_memory.json`. Entity-scrubbed, kind-prefixed, category-keyed.
3. **Memory → student:** `select_category_memory` picks items by category and the adapter prepends
   them to the **system message**. The student's task turns are never rewritten.

## The frozen-memory principle (non-negotiable)
Memory is **built once from train-split failures, then frozen** before held-out evaluation. No
teacher call, no rubric, and no judge feedback may touch a held-out question in any arm except the
`TEACHER` ceiling arm (which answers directly and is labelled as the ceiling).

If a lesson could not have been written before the held-out question was seen, the result is void.

## The rubric firewall (enforced in code)
| Role | Train rubrics | Validation / held-out rubrics |
|---|---|---|
| Student | never | never |
| Teacher | yes | never |
| Judge | yes | yes |

Enforced by `adapters/prbench.rubric_for` and `adapters/finance.rubric_for`, which raise
`PermissionError`; plus a stem-substring check (`_assert_no_rubric_smuggle`) so rubric text cannot
reach the student inside a memory item. Both are covered by tests — keep them passing.

## Artifacts
Every `(arm, task, rep)` cell is appended to a resumable JSONL under `runs/` and keyed
`"{arm}|{tid}|{rep}"`, so a killed run resumes and a summary can be recomputed with `--summarize`
(no API calls). One append-only log per experiment; do not invent a second format.

`runs/` is **gitignored** — see rules/03 for what that obliges you to do.

## No orchestrator
There is no single entrypoint and no live loop. Each experiment is one script in `scripts/`, run
explicitly. The retired `orchestrator.py` (Spider drift-detection era) was deleted in `6935400`;
do not resurrect it.
