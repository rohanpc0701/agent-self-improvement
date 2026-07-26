# Archive — retired eras

**History only.** Nothing here describes how the repo works today. For that, read the repo-root
`CLAUDE.md`, then the newest file in `docs/updates/`.

Archived 2026-07-25, when the project's docs were brought in line with the reasoning-RSI track
(teacher-built frozen memory on rubric-graded benchmarks). These files are kept because they record
decisions and results that later work is built on — not because they are still true.

## Era 1 — Spider text-to-SQL drift detection (hackathon)

The original project: watch a text-to-SQL agent on Spider, detect windowed accuracy drift, repair
failures with a teacher, feed learned few-shots back. Four stages (`harness` → `detector` →
`correction` → `viewer`) wired by `orchestrator.py`. **That code was deleted in `6935400`**; the
`detector/`, `viewer/` directories and `orchestrator.py` no longer exist.

| File | What it was |
|---|---|
| `design-spider-drift.md` | design rationale for the four-stage loop, drift math, KG A/B, McNemar |
| `plan-correction-kg-hackathon.md` | build order for the knowledge-graph correction module |
| `004-failure-mode-diag.md` | failure-mode diagnostics plan (valid-but-wrong vs invalid SQL) |
| `006-first-integration-checkpoint.md` | the four-person integration checkpoint work order |
| `demo-spider-recovery.gif` | the recovery curve from the Spider-era viewer |

The hackathon compliance rule that governed this era was replaced by
`.claude/rules/03-research-integrity.md`.

## Era 2 — RSI-Mem v1 (coding domain)

Same mechanism, unit-test-verified Python instead of SQL. Produced a measured hard-bucket lift
(0.273 → 0.455) that did not survive as a general claim; superseded by the finance/reasoning track.
Its null is part of the evidence trail in `docs/FINDINGS_CODING.md`.

| File | What it was |
|---|---|
| `RSI_MEM_PLAN-v1-coding.md` | the 24-week v1 plan (explicitly superseded by `docs/RSI_MEM_V2_FINANCE.md`) |
| `superpowers/plans/2026-07-19-diagnostics-ablation.md` | frozen-memory ablation (examples / rules / both / none) + capacity probe |
| `superpowers/specs/2026-07-18-diagnostics-ablation-design.md` | its design doc |
| `superpowers/plans/2026-07-20-rsi-mem-phase0-1.md` | coding G0.1 / G1.1 / G0.2 work order |
| `superpowers/specs/2026-07-19-alfworld-longhorizon-b1-design.md` | ALFWorld long-horizon arm — approved, never built; replaced by Phase 4b in the v2 plan |

## Era 3 — executed work orders and session handoffs (finance track)

These plans were carried out; their results live in `docs/FINDINGS_FINANCE.md` and `docs/updates/`.
The handoff docs are dated context dumps written for a single session — useful for archaeology, stale
as instructions.

| File | What it was |
|---|---|
| `superpowers/plans/2026-07-20-finance-phase0.md` | FinancePro Phase 0: splits, judge harness, headroom probe |
| `superpowers/plans/2026-07-20-finance-tracelift.md` | the uplift-gated TraceLift build loop |
| `CURSOR_SESSION_HANDOFF_2026-07-20.md` | Cursor → Fable handoff on `feat/rsi-mem-phase0` |
| `FABLE_HANDOFF-2026-07-19.md` | full-context handoff, coding/Prime era |
| `FABLE_CONTEXT_2026-07-21.md` | full-context reasoning doc written after the FinancePro null |

## What is *not* archived

`docs/FINDINGS_*.md` stay in place — including the coding and reasoning ones. They are the running
evidence log, nulls included, and the strategy ladder still reasons from them.
