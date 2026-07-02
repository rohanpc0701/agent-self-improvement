# CLAUDE.md — Agent Self-Improvement

> Project context for AI coding assistants working in this repo.

## What we're building (one sentence)
A self-improvement layer for AI agents: it watches a **text-to-SQL agent** evaluated on **Spider**, detects when accuracy **drifts** as queries get harder, and makes the agent **learn from its own failures** (teacher-generated few-shot examples + knowledge-graph rules) to recover — no human in the loop.

## The loop (4 stages)
```
 HARNESS --TelemetryRecord--> DETECTOR --DriftEvent--> CORRECTION
     ^                                                      |
     |  AgentConfig.few_shot_examples (starts empty; correction appends) |
     +----------------------------------------------------------------------+
              all stages append typed events -> events.jsonl --> VIEWER
```
The growing `few_shot_examples` list **is** the agent learning. That feedback path is the spine of the project.

## Architecture
| Stage | Dir | Consumes | Emits |
|-------|-----|----------|-------|
| Harness / telemetry | `harness/` | Spider data | `TelemetryRecord` |
| Detector | `detector/` | `TelemetryRecord` | `DriftEvent` |
| Correction / learning | `correction/` | `DriftEvent` + failing cases | `CorrectionAction` |
| Viewer | `viewer/` | `events.jsonl` | (UI) |

Each directory has its own `CLAUDE.md` with the stage contract + implementation notes.

## Shared rules
@.claude/rules/00-architecture.md
@.claude/rules/01-contracts.md
@.claude/rules/02-tech-decisions.md
@.claude/rules/03-compliance.md
@.claude/rules/04-workflow.md

## First thing, every session
1. Read the relevant stage's `CLAUDE.md`.
2. Build against **mock fixtures** (`python fixtures/generate_mocks.py`) — no live API required for tests.
3. Treat `contracts/schemas.py` as frozen unless you intentionally version the contract.

## Run from repo root
All imports assume repo root on path: `from contracts.schemas import TelemetryRecord`. Run commands from the repo root (or `pip install -e .`).
