# Design decisions

Interview-oriented notes on why this system is built the way it is.

## Problem

A production LLM agent's accuracy drifts when the input distribution shifts (harder queries, new schemas). Retraining is slow and expensive. This project asks: can the agent **detect drift statistically** and **recover in-context** from its own verified failures — with no human in the loop?

## Architecture

```
Harness → TelemetryRecord → Detector → DriftEvent → Correction → CorrectionAction
   ↑                                                              |
   └──────────── few_shot_examples (feedback spine) ──────────────┘
                    all stages → events.jsonl → Viewer
```

Stages communicate only through frozen Pydantic contracts ([`contracts/schemas.py`](../contracts/schemas.py)) and an append-only event log. Each stage can be developed and tested in isolation against mocks.

## Key decisions

| Decision | Choice | Rationale |
|---|---|---|
| Drift signal | Windowed mean of `execution_accuracy` | Single-query failures are noise; sustained drop is signal |
| Baseline | First ~40 easy-only runs | Medium questions make a weak base noisy; easy-only stabilizes baseline |
| Learning | In-context few-shots, not fine-tuning | Immediate feedback; no training pipeline; demo-friendly |
| Teacher | Stronger model repairs failures | Student stays weak; teacher only on drift (cheap hot path) |
| Verification | Execute teacher SQL vs gold | Teacher can be wrong; execution is ground truth |
| Fallback | Gold SQL when teacher misses | Guarantees at least one correct example per failure |
| Measurement | LEARN / HELD-OUT split by `db_id` | Examples never leak into benchmark questions |
| Baseline arm | `use_rules=False`, no examples | Contamination-free with/without comparison |
| Student economics | Local Ollama + cloud teacher | 1.5B free on hot path; M3 only when drift fires |

## Dose-response finding (important)

**Run 1:** Correction used only the drift event's 8 capped `failing_run_ids` (with duplicates from sampling with replacement). Result: **+0.000** hard-bucket recovery.

**Run 2:** Orchestrator harvests the full degraded window via [`_harvest_failing_cases`](../orchestrator.py) — up to 24 unique failures, round-robined across schemas. Result: **+0.233** hard-bucket recovery.

**Lesson:** The correction dose that heals a cloud student may be too dilute for a 1.5B local student. Small models need **denser same-schema examples** (~6 per prompt), not 1–2 scattered across DBs.

## Knowledge graph A/B

| Channel | Hard recovery |
|---|---|
| Few-shot examples only (`AGENT_USE_RULES=0`) | 0.333 |
| Examples + abstract rule text (`AGENT_USE_RULES=1`) | 0.300 |

At 1.5B scale, prose `(trap, fix)` rules slightly hurt vs SQL patterns alone. The graph remains useful as persistent memory and for larger models; default off for small students.

## What we explicitly did not do

- **ML drift classifier** — windowed statistics are explainable and need no labels
- **Fine-tuning** — contradicts the in-context learning story and adds ops burden
- **Multi-hop ReAct on every query** — increases cost and muddies the drift narrative

## Extensibility

The detector and correction stages only consume `TelemetryRecord.execution_accuracy`. Any task with automatic verification can plug in via a `TaskAdapter` ([`core/adapter.py`](../core/adapter.py)): Spider SQL today, GSM8K math tomorrow.
