# Chiron

Research code for a simple question: can a **stronger teacher model** help a **cheaper student model** do better on hard tasks **without fine-tuning**?

We tried several answers. Some demos look good. The honest held-out story is thinner. This README says what the repo is, what we measured, and how to run it.

Public repo: [rohanpc0701/Chiron](https://github.com/rohanpc0701/Chiron)

---

## What this repo contains

A **runtime loop** (not training):

1. Run a student on a stream of problems (easy → hard).
2. Detect when accuracy drops (**drift**).
3. Call a stronger **teacher** on the failures.
4. Turn repairs into **few-shot examples** (and optional knowledge-graph rules).
5. Re-run the **same** student with that text in the prompt.

Domains plug in as adapters: **coding** (primary on `main`), Spider SQL, GSM8K. Later work (finance / PRBench / plan-transfer) lives on other branches and in `docs/` — see [Where the research went](#where-the-research-went).

---

## Results (be careful what you cite)

### Coding — same-distribution recovery (works as a demo)

Student `meta-llama/Llama-3.2-3B-Instruct` + teacher `minimax/minimax-m2.5` on [Prime](https://docs.primeintellect.ai/inference/overview). Hard-bucket unique questions after the loop:

| | Accuracy |
|---|---:|
| Without examples / rules | 0.273 |
| After correction | **0.455** |
| Δ | **+0.182** |

That run: drift fired, ~15 teacher few-shots + anchors, a few KG rules. Reproduce with `bash scripts/use_prime_student.sh full`.

### Coding — hard held-out transfer (mostly does not)

Once memory is frozen and scored on problems **not** used to build it, lift often disappears. Small students hit a **capacity floor**; sometimes examples **hurt**. Details: [`docs/FINDINGS_CODING.md`](docs/FINDINGS_CODING.md).

### Spider (historical)

| Setup | Without | With | Δ |
|---|---:|---:|---:|
| Local 1.5B + MiniMax teacher | 0.100 | 0.333 | +0.233 |
| Hackathon MiniMax student + teacher | 0.300 | 0.567 | +0.267 |

More in [`docs/design.md`](docs/design.md). Treat as early demo numbers, not the final claim.

---

## How the loop works

```
Harness ──telemetry──▶ Detector ──drift──▶ Correction (teacher)
   ▲                                              │
   └──────── few-shots (+ optional KG rules) ─────┘
                    events.jsonl ──▶ Viewer
```

| Piece | Job |
|-------|-----|
| [`harness/`](harness/) | Runs the student, scores outputs |
| [`detector/`](detector/) | Windowed accuracy; fires when it drops |
| [`correction/`](correction/) | Teacher repair → verified few-shots; optional KG |
| [`viewer/`](viewer/) | Plots the run from `events.jsonl` |

The student **weights never change**. Only the prompt grows.

**Memory lookup:** few-shots are filtered by domain (`db_id` / topic). KG rules match by **substring** (trigger phrase or table/column name in the question) — not embeddings.

---

## Setup

Python ≥ 3.10.

```bash
git clone https://github.com/rohanpc0701/Chiron.git
cd Chiron
pip install -e .
pip install -r requirements.txt   # if you want the viewer / extras
```

Hermetic tests (no API key):

```bash
python fixtures/generate_mocks.py
pytest -q
```

Put keys in `.env` (gitignored), e.g.:

```bash
PRIME_API_KEY=...          # coding on Prime
# or
OPENROUTER_API_KEY=...
```

---

## Run something real

### Coding on Prime (recommended entrypoint)

```bash
bash scripts/use_prime_student.sh list
bash scripts/use_prime_student.sh smoke    # student + teacher unit-test check
bash scripts/use_prime_student.sh probe    # cheap WITH/WITHOUT
bash scripts/use_prime_student.sh full     # full loop + recovery
```

Hard-curriculum style eval (learn on hard, score held-out hard):

```bash
bash scripts/use_prime_student.sh curriculum
```

Overrides: `PRIME_AGENT_MODEL=...`, `PRIME_TEACHER_MODEL=...`.

### Coding on OpenRouter

```bash
bash scripts/use_openrouter_student.sh smoke
bash scripts/use_openrouter_student.sh curriculum
```

### Orchestrator (any adapter)

```bash
python orchestrator.py --adapter coding --full --fresh
python orchestrator.py --adapter spider --probe
python orchestrator.py --adapter gsm8k --full --fresh
```

### Viewer (offline)

```bash
make demo
# or: VIEWER_LOG=events.jsonl uvicorn viewer.app:app --port 8011
```

---

## Layout

```
contracts/     Shared schemas + events.jsonl I/O
adapters/      coding, spider_sql, gsm8k_math, finance
harness/       Student client, feeds, eval / sandbox
detector/      Drift detection
correction/    Teacher, few-shots, KG (graph / inject)
viewer/        Live UI over the event log
fixtures/      Problem subsets + mocks
scripts/       Prime / OpenRouter entrypoints
orchestrator.py
```

---

## Where the research went

After coding, we pushed on:

- **GSM8K** — uplift-gated memory; gate worked, held-out transfer didn’t ([`docs/FINDINGS_REASONING.md`](docs/FINDINGS_REASONING.md))
- **Finance / PRBench** — rubric-graded reasoning; frozen “broadcast” memory mostly null; **per-task teacher plans** looked stronger (see feature branches `feat/finance-*`, `feat/q1-self-plan`, and sibling work)

If you’re reading this on **`main`**, you’re looking at the drift → teacher → few-shot harness. Newer claim-writing lives on other branches and in the findings docs. Prefer those over the README when the two disagree.

---

## Author

[Rohan Chavan](https://github.com/rohanpc0701)
