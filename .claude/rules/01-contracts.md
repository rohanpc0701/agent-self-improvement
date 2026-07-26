# LOCKED RULE: Data contracts (canonical names + legacy aliases)

**`contracts/schemas.py` is the shared dependency. Do not redefine record shapes locally.**

Canonical field names are **domain-agnostic** (`generated_output`, `correct_output`, `domain_id`).
Legacy SQL-shaped keys (`generated_sql`, `correct_sql`, `db_id`) are still accepted on **input** via
Pydantic validation aliases so historical fixtures and logs load. Prefer canonical names in new code.
If a field must change: announce, change once, re-pull.

## The record that matters now
`FewShotExample` — **one memory item**. Every field is load-bearing:

| Field | Meaning on the live path |
|---|---|
| `question` | the item's **header**, not a question: `[FINANCE_PLAYBOOK] <category>` (or `[FINANCE_TRAP]` / `[FINANCE_SKELETON]`). `memory_kind_of()` parses this prefix, and `select_category_memory()` uses the kind to decide what gets injected. Change the prefix and retrieval silently changes. |
| `correct_output` | the lesson body (entity-scrubbed, token-trimmed) |
| `domain_id` | the retrieval key — the benchmark **category** (e.g. `Corporate Finance`), not a database id |
| `source` | provenance: `teacher` \| `tracelift` \| `uplift` \| `gold` \| `anchor` |

`AgentConfig` — `model` + `few_shot_examples`. The memory list is passed explicitly to
`generate_answer(..., memory=...)` on the eval path; `config.few_shot_examples` is the fallback.

## Records kept for the retired drift-detection era
`TelemetryRecord`, `DriftEvent`, `CorrectionAction`, `Difficulty`, `FailureMode` and
`contracts/eventlog.py` (`append_event`, `read_events`, `tail_events`) are **not on the live path**.
They still load historical `events.jsonl` and are imported by legacy modules and their tests. Leave
them alone; don't build new work on them.

## Rubric text is not a contract field
Rubrics live in the dataset fixtures and are reached only through `rubric_for(qid, role=...)`.
Never pass a rubric into a schema object, a memory item, or a student prompt (rules/00).
