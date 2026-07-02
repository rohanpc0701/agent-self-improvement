# Agent Self-Improvement

[![CI](https://github.com/rohanpc0701/agent-self-improvement/actions/workflows/ci.yml/badge.svg)](https://github.com/rohanpc0701/agent-self-improvement/actions/workflows/ci.yml)

**Drift detection and automated self-improvement for AI agents** — task-agnostic middleware
with adapters for text-to-SQL (Spider) and grade-school math (GSM8K).

![Recovery curve: drift detection and self-correction replay](docs/demo.gif)

A text-to-SQL agent runs in "production" against the [Spider](https://yale-lily.github.io/spider)
benchmark. When its accuracy **drifts** as queries get harder, the system detects the
degradation, makes the agent **learn from its own failures** — a teacher model turns the
agent's mistakes into few-shot examples and knowledge-graph rules — and the agent **recovers
on questions it has never seen**. No human in the loop. The same detector → correction →
viewer pipeline runs on **GSM8K math** via `--adapter gsm8k`, so the pitch is middleware,
not a single-benchmark script.

> Started at **The Self-Improvement Stack** track of the AI Engineer World's Fair Hackathon
> (June 2026) and extended since: local open-source student support, knowledge-graph correction,
> enriched failure harvesting, continuous multi-drift mode, and a second task adapter.

**Resume bullet:** Built self-improvement middleware for verifiable agent tasks — local
1.5B student + cloud teacher, +23pt held-out hard-bucket recovery on Spider, contamination-free
eval, 219-test CI, dual SQL/math adapters.

See [`docs/design.md`](docs/design.md) for architecture decisions, dose-response findings, and
measurement guardrails.

---

## Results

### Dual-benchmark summary (local Qwen2.5-1.5B student)

| Domain | Adapter | Hard-bucket WITHOUT | WITH correction | Δ |
|---|---|---|---|---|
| Text-to-SQL (Spider) | `--adapter spider` | 0.100 | **0.333** | **+0.233** |
| Grade-school math (GSM8K) | `--adapter gsm8k` | 0.150 | **0.350** | **+0.200** |

Spider uses 24 teacher-verified same-schema SQL examples (`AGENT_USE_RULES=0`). GSM8K uses
gold-answer few-shots from the LEARN split — rules are SQL-specific and skipped for math.
Effect stable across seeds on Spider probe: run `python3 scripts/multi_seed_eval.py` for
mean ± std over seeds `[42, 7, 99]`.

### Local open-source student (Qwen2.5-1.5B via Ollama) + cloud teacher — Spider detail

The flagship configuration: a **1.5B-parameter student runs locally and free** on the hot path;
the frontier teacher (MiniMax-M3) is called only when drift fires. The student **more than
tripled its accuracy on hard held-out queries** it had never seen as examples:

| Hard-bucket execution accuracy (same 30 held-out questions, same eval) | Accuracy |
|---|---|
| Student (Qwen2.5-1.5B) — no examples | 0.100 |
| **Student after self-correction — 24 teacher-repaired failures as examples** | **0.333** |
| **Improvement** | **+0.233** |

The extra-hard bucket improved too (0.091 → 0.273), and recovery was broad across schemas —
not one lucky database.

**A dose-response finding along the way:** injecting only the drift event's 8 capped
(and partially duplicated) failing cases recovered **+0.000**; harvesting the full degraded
window — 24 deduplicated failures, round-robined so every schema gets coverage — recovered
**+0.233**. The correction dose that heals a 40B-class model is too dilute for a 1.5B one.
That diagnosis came entirely from the event log the system writes about itself.

**Knowledge-graph rules A/B (same run, same examples):**

| Recovery channel (hard bucket) | Recovered accuracy | Δ vs base |
|---|---|---|
| Few-shot examples only (`AGENT_USE_RULES=0`) | **0.333** | **+0.233** |
| Examples + KG rule text in prompt (`AGENT_USE_RULES=1`) | 0.300 | +0.200 |

At 1.5B scale, abstract `(trap, fix)` rule text slightly *hurts* vs examples alone — the
small student copies SQL patterns better than it follows prose corrections. The graph is
still valuable as persistent memory and for larger models; toggle with `AGENT_USE_RULES`.

### Original hackathon configuration (MiniMax M2.7 student)

| Hard-bucket execution accuracy (same 30 held-out questions, same eval) | Accuracy |
|---|---|
| Base agent — no examples | 0.300 |
| Teacher model (MiniMax-M3) — the stronger model used to generate examples | 0.400 |
| **Base agent after self-correction — learned few-shot examples** | **0.567** |
| **Improvement over base** | **+0.267** |

The self-improved base agent **exceeded its own teacher** on the same questions — because
5 of the 10 injected examples are execution-verified gold SQL that the teacher itself couldn't
produce, scaffolding the weaker model beyond the stronger one's unaided performance.

<img width="1045" height="678" alt="Self Improving Agent" src="https://github.com/user-attachments/assets/83d13a2b-cb72-49c9-a72e-bb87b0264062" />

*Hackathon screenshot above; animated replay from recorded telemetry: [`docs/demo.gif`](docs/demo.gif).*

In one unattended run the detector fired automatically at the change-point
(`severity=0.295`, windowed accuracy `0.48` vs `0.775` baseline), correction synthesized 10
few-shot examples (3 teacher-verified, 5 gold-fallback, 2 anti-forgetting anchors), and the
agent recovered. The held-out questions are drawn from a pool disjoint from the examples'
source, so this is out-of-sample generalization — not memorization.

**Is it chance?** We re-evaluate the same 30 hard questions twice — each with and without the
learned examples — and run a paired **McNemar exact test**: **p = 0.016** (two-sided), paired
Δ **+0.233**, 95% CI **[+0.08, +0.39]**. Of the 7 questions whose outcome changed, **7 improved
and 0 regressed** — the anti-forgetting anchors held: no question the agent already answered
correctly broke after injection. (Absolute accuracy shifts ~2–3 questions run-to-run because the
model is not perfectly deterministic even at `temperature=0`; the demo run above measured +0.267,
the paired test +0.233 — the *effect* is stable, so the headline claim is the paired delta with
its interval, not a single point estimate.) Reproduce with `python orchestrator.py --significance`.

---

## The idea: a feedback spine

The agent's state is one `AgentConfig`, and the only thing that changes about it is a list:

```python
class AgentConfig(BaseModel):
    config_id: str
    model: str
    few_shot_examples: list[FewShotExample] = []   # starts EMPTY, grows via correction
```

`few_shot_examples` **starts empty**. On drift, correction appends learned examples; the
harness reads them on the next runs. Recovery happens because the agent *learned* — nothing
is reverted, no bigger model is swapped in. **That growing list is the self-improvement.**

---

## Architecture

Four typed stages, connected only through frozen [Pydantic contracts](contracts/schemas.py)
and a single append-only event log:

```
  Harness ──TelemetryRecord──▶ Detector ──DriftEvent──▶ Correction
     ▲                                                       │
     └────────── learned few-shot examples (feedback) ───────┘
                  every stage ──▶ events.jsonl ──▶ Viewer
```

| Stage | Consumes | Emits | What it does |
|-------|----------|-------|--------------|
| **[Harness](harness/)** | Spider data + `AgentConfig` | `TelemetryRecord` | Runs the agent over a change-point query stream; execution-based eval |
| **[Detector](detector/)** | `TelemetryRecord` | `DriftEvent` | Windowed statistical drift detection; classifies the failure mode |
| **[Correction](correction/)** | `DriftEvent` + failing cases | `CorrectionAction` | Teacher generates corrected SQL → verified → few-shot examples |
| **[Viewer](viewer/)** | `events.jsonl` | (web UI) | Thin live view: recovery curve + SQL example panel |

The seams are the only coordination points; each stage is built and tested in isolation
against mock fixtures, then connected by [`orchestrator.py`](orchestrator.py).

---

## How the loop works

**1. The feed is a change-point on a stratified stream** ([`harness/feed.py`](harness/feed.py)).

- *Baseline* — easy Spider questions, base agent → stable high accuracy (~0.78).
- *Change-point* — the input distribution shifts to **hard / extra-hard** questions
  (models real-world complexity creep).
- *Degraded* — hard questions, base agent → accuracy settles lower. **The transition is the drift.**
- *Recovery* — same hard distribution, now with learned examples → accuracy climbs.

Hard/extra questions are split **by database** into a disjoint **LEARN** pool (the only source
of few-shot examples) and a **HELD-OUT** pool (the benchmark). The agent can never regurgitate
an example it was handed — recovery is an out-of-sample generalization claim.

**2. The detector finds drift over a window, never on one query** ([`detector/detector.py`](detector/detector.py)).

A single bad query is noise; sustained degradation is drift. The detector is a state machine —
`WARMUP` (freeze a baseline) → `NORMAL` (rolling window) → `DRIFTING` (latched). It fires only
when the windowed mean stays a threshold below baseline for several consecutive records, then
emits a `DriftEvent` carrying the severity, the windowed-vs-baseline means, the dominant
**failure mode** (`valid_but_wrong` vs `invalid_sql`), and the specific `failing_run_ids` to
learn from. Transient API outages are filtered out so they can't fire false drift.

**3. Correction makes the agent learn from its failures** ([`correction/learner.py`](correction/learner.py)).

On drift, the orchestrator harvests the **full degraded window** of failures — deduplicated by
question and round-robined across database schemas so no schema is starved (capped at 24 to
bound teacher calls). For each failing case: a stronger **teacher model** generates corrected
SQL, which is then **execution-verified against the gold query** — if the teacher's SQL doesn't
produce the right result set, it's discarded and the gold SQL is used instead. A few
**anti-forgetting anchors** (easy successes) are kept so learning hard queries doesn't regress
easy ones. The result is a `CorrectionAction` whose examples are injected back into
`few_shot_examples`.

**4. A knowledge graph remembers failure patterns across runs** ([`correction/graph.py`](correction/graph.py)).

The same repaired failures are distilled into `(trap, fix)` rules attached to schema nodes —
e.g. *"trap: unnecessary JOINs to unrelated tables → fix: join only the tables the question
requires"*. Rules persist in `correction/graph_store.json` and are spliced into the agent's
prompt on future same-schema questions ("Known corrections for this schema"). Rules that recur
across ≥2 databases get promoted to global scope. Toggle injection with `AGENT_USE_RULES=0` —
useful because small students can be confused by abstract rule text (examples are the
empirically stronger channel at 1.5B scale).

---

## Instant demo (no API key, no model download)

A complete recorded run — drift, correction, recovery — ships in the repo. Replay it in the
live viewer with nothing but Python installed:

```bash
pip install -r requirements.txt
make demo
# or: VIEWER_LOG=fixtures/demo_events.jsonl uvicorn viewer.app:app --port 8011
# open http://127.0.0.1:8011
```

**Multi-drift continuous demo** (two V-shapes on one chart):

```bash
VIEWER_LOG=fixtures/demo_continuous.jsonl make demo
```

Generate the GIF locally: `make gif` (requires `pip install pillow`).

You'll see the accuracy curve crater at the change-point, the drift alarm fire, and the
recovery climb after the teacher's examples are injected — the exact telemetry from the
Qwen2.5-1.5B run in the results table above. The viewer shows **multiple drift markers**,
a **correction timeline**, and a **what-changed** panel for injected examples.

---

## Quickstart

```bash
# 1. Install (Python ≥ 3.10)
pip install -e .          # or: pip install -r requirements.txt

# 2. Generate the mock fixtures (lets any stage run standalone)
python fixtures/generate_mocks.py

# 3a. Local student (recommended — free, no key for the hot path):
ollama pull qwen2.5:1.5b-instruct
export AGENT_BASE_URL=http://localhost:11434/v1
export AGENT_MODEL=qwen2.5:1.5b-instruct

# 3b. Teacher key (only needed for the correction stage, fires on drift):
export MINIMAX_API_KEY=sk-...
```

The Spider subset and SQLite databases are already checked in under
[`fixtures/`](fixtures/). To rebuild them from a full Spider download, see
[`fixtures/prepare_spider.py`](fixtures/prepare_spider.py).

**Models** (all overridable via env): student defaults to `MiniMax-M2.7-highspeed`
(`AGENT_MODEL` + `AGENT_BASE_URL` point it at any OpenAI-compatible server, e.g. Ollama);
teacher `MiniMax-M3` (`TEACHER_MODEL`); rule distiller (`DISTILL_MODEL`). The student is
deliberately weak so it genuinely struggles on hard SQL — that's the point.

---

## Running the loop

```bash
# Full end-to-end demo: baseline → change-point → drift → correction → recovery
python orchestrator.py --full --fresh

# Spider (default) or GSM8K math adapter
python orchestrator.py --adapter spider --full --fresh
python orchestrator.py --adapter gsm8k --full --fresh

# Continuous multi-drift: repeated degrade→correct→recover; examples accumulate (cap 32)
AGENT_USE_RULES=0 python orchestrator.py --continuous --max-corrections 3 --fresh

# Multi-seed stability check (cheap probe, ~22 calls per seed)
python3 scripts/multi_seed_eval.py

# Cheap validation gates (run these before a full run — they save API calls):
python orchestrator.py --probe              # ~22 calls: do schema-relevant examples help?
python orchestrator.py --dry-run-heldout    # base accuracy on held-out (confirms headroom)
python orchestrator.py --dry-run-degraded   # confirms the detector will fire
```

Each stage also runs standalone:

```bash
python -m detector.detector --input fixtures/mock_telemetry.jsonl
python -m correction.correction --drift fixtures/mock_drift_events.jsonl
python -m harness.runner --full
uvicorn viewer.app:app --port 8011          # then open http://127.0.0.1:8011
```

The viewer reads `fixtures/mock_events.jsonl` by default; point it at a live run with
`VIEWER_LOG=events.jsonl`.

---

## How we measure honestly

The improvement claim is only as good as the measurement. Three guardrails make it defensible:

- **Stratified by difficulty.** Recovery questions are *all* hard, so the *overall* accuracy
  curve looks low by construction — the honest signal is **hard-bucket accuracy WITHOUT vs WITH
  learned examples**, comparing the *same* questions at the *same* difficulty.
- **Unique-question accuracy.** The stream samples with replacement; at `temperature=0` the
  model is deterministic, so repeated questions are de-duplicated before averaging. This removes
  windowed-mean noise that otherwise looks like phantom "learning."
- **Out-of-sample by schema.** Examples come only from the LEARN split; the benchmark is the
  disjoint HELD-OUT split, and a same-database filter drops any cross-schema example as noise.

A full run prints the bottom line — windowed drift detection plus the side-by-side comparison:

```
[detector] Drift detected! channel=execution_accuracy, severity=0.295  (window 0.48 vs baseline 0.775)
[correction] CorrectionAction: 10 examples — 3 teacher, 5 gold, 2 anchor

  Self-improvement result (hard bucket, 30 unique held-out questions):
    WITHOUT examples (base)  : 0.300
    WITH examples (recovered): 0.567
    Delta                    : +0.267
  ✓ Agent improved on hard queries after learning from its own failures.
```

---

## Project layout

```
contracts/      FROZEN shared schemas + the events.jsonl read/write helper
core/           TaskAdapter protocol
adapters/       spider_sql, gsm8k_math benchmark adapters
harness/        text-to-SQL agent, Spider execution-eval, change-point feed
detector/       windowed drift detection + failure-mode classification
correction/     teacher → verify → anchor → few-shot examples + example memory cap
viewer/         FastAPI + Chart.js live view (server-side windowing; not Streamlit)
fixtures/       Spider subset, GSM8K slice, SQLite DBs, mock generators, demo logs
docs/           design.md, demo.gif
scripts/        demo.sh, multi_seed_eval.py, generate_demo_gif.py
orchestrator.py wires the full live loop (--adapter, --continuous)
```

## Testing

```bash
pytest        # 219 tests across all stages
```

Tests are hermetic — external model calls and DB lookups are injected, so the full suite runs
offline without an API key.

## Tech stack

Python · Pydantic (typed contracts) · SQLite + Spider EX metric (execution-based eval) ·
local open-source student via Ollama (any OpenAI-compatible server) · MiniMax teacher ·
networkx knowledge graph (correction memory) · FastAPI + Chart.js (viewer).
Local-only by design, but the architecture scales horizontally without redesign: stateless 
stages behind frozen contracts, per-channel detectors, Ray fan-out for parallel training. 
The same shape generalizes from one agent to hundreds of independent telemetry channels.

## Author

Built by [Rohan Chavan](https://github.com/rohanpc0701).
</content>
