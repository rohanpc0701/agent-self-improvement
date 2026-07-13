# LOCKED RULE: Data contracts (canonical names + legacy aliases)

**`contracts/schemas.py` is the single shared dependency of all four stages.**

Canonical field names are **domain-agnostic** (`generated_output`, `correct_output`,
`domain_id`, `invalid_output`). Legacy SQL-shaped keys (`generated_sql`, `correct_sql`,
`db_id`, `invalid_sql`) remain accepted on **input** via Pydantic validation aliases so
historical `events.jsonl` and fixtures still load.

- Prefer the canonical names in new code.
- Do not redefine record shapes locally — import from `contracts.schemas`.
- If a field must change again: announce, change once, everyone re-pulls.

## The records
- `AgentConfig` — `few_shot_examples` starts empty and grows via correction.
- `TelemetryRecord` — one agent run (channels + question/output for the viewer).
- `DriftEvent` — windowed drift + `failure_mode` + `failing_run_ids`.
- `CorrectionAction` — `new_few_shot_examples` the agent should learn from.

## Why `difficulty` is on every record
Stratifying accuracy by difficulty makes the improvement claim defensible.

## Event log helper
Use `contracts/eventlog.py` (`append_event`, `read_events`, `tail_events`).
