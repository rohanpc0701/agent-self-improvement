# Design: Diagnostics — Frozen-Memory Ablation + Injection Audit

**Date:** 2026-07-18
**Status:** Approved (pending spec review)
**Owner:** Rohan Chavan
**Context:** `docs/FABLE_HANDOFF.md` §6C reported a null result — the hard-curriculum
eval gave Δ = +0.000 self-lift on held-out hard problems (student WITHOUT 0.111 vs
WITH 0.111) and a −0.889 gap to the unaided teacher. Before building new learning
mechanisms (failure-conditioned ICL, retrieval rewrite, LoRA), this milestone
diagnoses **why** memory did not transfer.

## Goal

Answer three questions with paired, per-question measurements — not vibes:

1. **Do learned examples/rules actually reach held-out prompts?** (injection audit)
2. **Which memory channel matters — examples, KG rules, both, or neither?** (ablation)
3. **Is the 3B student's capacity the binding constraint?** (7B capacity probe)

The output is a pre-registered decision tree that picks the next build.

## Non-goals (explicitly not building yet)

- Long-horizon / multi-step agents
- LoRA / any fine-tuning
- GSM8K or other second domains
- Spider revival, viewer work
- Online multi-correct during LEARN
- Retrieval rewrite (only if the decision tree says so, as the *next* milestone)
- LiveCodeBench importer (Stage 2 — deferred until Stage 1 pool proves too small
  or too easy)

## Design

### 1. Dataset import (replaces hand-authoring)

**Stage 1 (this milestone):** `scripts/import_problems.py`

- Pull MBPP+ and HumanEval+ (EvalPlus variants; Apache-2.0 / MIT) via HF `datasets`.
- Map to the existing fixture schema:
  `{id, question, function_name, tests: [{args, expected}], topic, difficulty, gold_solution}`.
- New IDs prefixed `h2_` so the original 70 problems keep stable IDs.
- Keep dataset gold solutions. **Validate every gold against its own tests in
  `harness/sandbox.py`; drop any problem whose gold does not pass.**
- Topic tags: map from dataset metadata where available; otherwise one cheap
  teacher batch call labels each problem with one of the existing topics
  (`dp, arrays, graphs, strings, greedy, arithmetic`). Topics are required —
  they drive the same-topic retrieval filter in `adapters/coding.py`.

**Difficulty probe:** `scripts/import_problems.py --probe-difficulty`

- For each imported candidate, run the 3B student **k=2 samples at temperature 0.7**
  (bare prompt — no examples, no rules).
- Keep problems with pass-rate ≤ 0.5 as `difficulty=hard`; discard the rest
  (the fixture already has enough easy problems).
- Rationale for temp 0.7 rather than temp 0: the eval arms run at temperature 0.0
  (deterministic). Filtering on a temp-0 failure would make the WITHOUT arm's
  0.000 baseline true by construction — degenerate and unable to detect harm.
  A stochastic probe keeps the temp-0 baseline honest.
- The probe runs **before** the LEARN/held-out split and is applied uniformly,
  so it defines the hard distribution without biasing the held-out comparison.
- Target: ≥ 60 hard problems total (existing 35 + imported). If the filtered
  import yields fewer, that triggers Stage 2 (LCB) as a follow-up milestone.

**Stage 2 (deferred):** LiveCodeBench functional-subset importer with
teacher-generated, test-verified pseudo-gold. Not in this milestone.

### 2. Split

- 30 LEARN / 30 held-out from the hard pool, **topic-stratified**, seeded
  (`--seed`, default 42), implemented by extending the existing split logic in
  `harness/feed.py` (`build_hard_curriculum_stream`). Deterministic given seed.

### 3. Ablation flags

- New env var `AGENT_USE_EXAMPLES=0/1` (default 1), read in
  `adapters/coding.py:generate_code`. When 0, skip `_examples_block`. Mirrors
  the existing `AGENT_USE_RULES` / `use_rules` mechanism.
- Orchestrator flag `--ablation none|examples|rules|both|all` for the held-out
  eval. `all` runs the four arms sequentially against the **same frozen memory**
  and the **same held-out questions**.

### 4. Injection audit (contract bump — additive, announce before pushing)

- `TelemetryRecord.injection_stats: dict | None = None`. Additive with a None
  default: legacy `events.jsonl` rows load unchanged; old readers ignore it.
- Populated by `generate_code` for every run:

```json
{
  "examples_available": 19,
  "examples_injected": 2,
  "example_ids": ["..."],
  "rules_injected": 1
}
```

- Report per arm: mean examples injected per prompt, and the headline
  diagnostic — **% of held-out prompts that received 0 examples** (prime
  suspect for the Δ = 0 null, given the same-topic cap-3 filter at
  `adapters/coding.py:73`).

### 5. Arms

One learn phase on the new 30-problem LEARN pool → freeze memory
(few-shots + `correction/graph_store.json`). Then:

| Arm | Student | Examples | Rules |
|----------|--------------------------|----------|-------|
| none | Llama-3.2-3B | ✗ | ✗ |
| examples | Llama-3.2-3B | ✓ | ✗ |
| rules | Llama-3.2-3B | ✗ | ✓ |
| both | Llama-3.2-3B | ✓ | ✓ |
| cap-none | Qwen2.5-Coder-7B-Instruct| ✗ | ✗ |
| cap-both | Qwen2.5-Coder-7B-Instruct| ✓ | ✓ |

- 7B model on Prime; fallback `meta-llama/Llama-3.1-8B-Instruct` if Qwen coder
  tier unavailable. Override via `PRIME_AGENT_MODEL`.
- ~180 held-out eval calls (6 arms × 30 questions) + 1 teacher batch (episodic,
  on drift only). The difficulty probe adds ~2 cheap 3B calls per imported
  candidate (§1). All eval arms run at temperature 0.0 for paired per-question
  comparison.

### 6. Measurement

- Per-arm accuracy on the 30 held-out hard questions.
- Paired McNemar (existing `_mcnemar_report`) for:
  both vs none · examples vs none · rules vs none · cap-none vs 3B-both.
- Injection audit summary per arm (§4).
- All numbers reported as-is; nulls published (handoff doc rule: no inflation).

### 7. Pre-registered decision tree

| Observation | Next milestone |
|---|---|
| High % of prompts with 0 examples injected | Retrieval fix (algorithm/plan similarity instead of topic tags) |
| Examples reach prompts but no lift | Failure-conditioned, unit-test-gated ICL bank (handoff §8 rank 1) |
| cap-none ≫ 3B-both | Capacity floor dominant → promote 7B student, or LoRA-on-verified-repairs path |
| Lift appears | Multi-seed replication + McNemar before claiming anything |

### 8. Testing

- Hermetic pytest, no live API (matches existing ~235 tests):
  - `AGENT_USE_EXAMPLES` flag on/off changes prompt assembly as expected
  - Split determinism for fixed seed; LEARN ∩ held-out = ∅; stratification holds
  - `injection_stats` round-trips through `events.jsonl`; legacy rows without
    it still parse
  - Fixture validation: every gold passes its tests in the sandbox (fast subset
    in CI; full pass as a script)
  - Import script mapping logic (mocked dataset rows → fixture schema)

### 9. Operational notes

- Entry point: extend `scripts/use_prime_student.sh` with an `ablate` subcommand
  (curriculum learn → freeze → all arms → report).
- `--fresh` semantics unchanged; ablation arms require a prior learn phase in
  `events.jsonl` (same guard as `--compare-teacher`).
- Known pitfalls from handoff §10 still apply (`.env` Ollama override, proxy
  403s on Prime, detector's ~40 easy-warmup requirement).

## Results — 2026-07-19

**Setup:** hard pool 75 (35 original + 40 `h2_*` from EvalPlus probe). Learn:
`--hard-curriculum --fresh --n-learn 100 --n-heldout 30 --heldout-frac 0.5`.
Frozen memory: 26 examples (19 teacher / 5 gold / 2 anchor), 3 KG rules.
Student 3B = `meta-llama/Llama-3.2-3B-Instruct`. Capacity student =
`qwen/qwen3-coder` (Prime; `Qwen/Qwen2.5-Coder-7B-Instruct` 404'd).
Logs: `runs/ablate_20260719_0903.log`, `runs/ablate_capacity_coder_20260719_1019.log`.

### 3B frozen-memory ablation (n=29 unique held-out hard)

| arm | acc | n | mean_inj | zero_inj% |
|---|---:|---:|---:|---:|
| none | 0.103 | 29 | 0.00 | 100 |
| examples | 0.034 | 29 | 2.93 | 0 |
| rules | 0.103 | 29 | 0.00 | 100 |
| both | 0.034 | 29 | 2.93 | 0 |

**Injection audit:** examples reach prompts (mean_inj≈2.93, zero_inj=0% on
examples/both arms). Not a retrieval plumbing failure.

**McNemar vs none (n=29):**

| pair | Δ | discordant b+c | exact p |
|---|---:|---:|---:|
| examples vs none | −0.069 | 2 | 0.5000 |
| rules vs none | +0.000 | 0 | 1.0000 |
| both vs none | −0.069 | 2 | 0.5000 |

### Capacity probe (`qwen/qwen3-coder`, same frozen memory, n=29)

| arm | acc | n | mean_inj | zero_inj% |
|---|---:|---:|---:|---:|
| none | 0.931 | 29 | 0.00 | 100 |
| both | 0.897 | 29 | 2.93 | 0 |

both vs none: Δ = −0.034, b+c=1, p=1.0000.

### Pre-registered pair: cap-none vs 3B-both (hand-paired by question index)

| | |
|---|---|
| cap-none | 0.931 |
| 3B-both | 0.034 |
| cap>3B / 3B>cap / both✓ / both✗ | 26 / 0 / 1 / 2 |
| McNemar exact p | 0.0000 |

### Decision-tree row that fired (§7)

**Primary:** `cap-none ≫ 3B-both` → capacity floor dominant. Next milestone:
promote a mid-size student (or LoRA-on-verified-repairs), not more ICL plumbing
on 3B alone.

**Secondary:** examples reach prompts but no lift (and slight harm). On a
capacity-adequate student, failure-conditioned unit-test-gated ICL remains the
next memory-quality experiment — but only after the student can solve ~hard
problems unaided.

**Not fired:** high zero-injection % (retrieval rewrite deferred).
