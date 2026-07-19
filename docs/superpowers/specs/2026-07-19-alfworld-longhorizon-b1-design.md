# Design: Long-Horizon B1 — ALFWorld Episodes

**Date:** 2026-07-19
**Status:** Approved (build gated on coding chain results — see §10)
**Owner:** Rohan Chavan
**Context:** Coding-domain diagnostics (see `docs/FINDINGS_CODING.md`) proved the
loop runs mechanically but showed (a) student capacity is a hard precondition,
(b) teacher-verified memory is not automatically useful to the student. B1
ports the self-improvement spine to multi-step episodes with both lessons
baked in, per the long-horizon direction in `docs/FABLE_HANDOFF.md` §9.

## Goal

Demonstrate the drift → teacher → memory → recovery story on **multi-step
agent episodes** with an automatic success signal: the student agent's
windowed episode success rate drops under a task-difficulty shift, the teacher
repairs failed trajectories, and the student recovers on **held-out unseen
tasks** — measured WITHOUT vs WITH memory and vs the unaided teacher.

## Non-goals

- WebArena / SWE-bench / GAIA (milestone 2 candidates, after B1)
- RL or fine-tuning (unless the spine ladder §9 reaches rung 3)
- Multi-environment abstraction (one env first; adapter protocol already exists)
- Viewer work beyond labeling episodes

## Design

### 1. Environment

- `alfworld` (pip; TextWorld engine). No docker; hundreds of episodes are
  laptop-feasible.
- Episode = one task instance. Student runs a ReAct-style loop
  (observation → thought → action), hard cap ~30 steps.
- **Success is binary and automatic** (environment reports goal satisfaction).
- Splits: ALFWorld train tasks → LEARN pool; `valid_unseen` → held-out
  (contamination-free by construction).

### 2. Change-point feed

- Easy phase: `pick_and_place_simple` (highest base success).
- Hard shift: composite types — `pick_two_obj_and_place`,
  `pick_heat_then_place_in_recep`, `pick_cool_then_place_in_recep`,
  `pick_clean_then_place_in_recep`.
- Difficulty = task-type bucket; same stream-builder pattern as coding
  (easy warmup for detector baseline → hard LEARN → held-out hard).

### 3. Student band search FIRST (hard gate)

Lesson from coding: no experiment before the band is confirmed.

- Sweep 3–4 Prime models bare on ~30 `valid_unseen` episodes.
- Required: unaided success in **0.3–0.6**. Too weak = floor (3B repeat),
  too strong = ceiling (qwen3-coder repeat).
- No model in band → STOP; adjust task mix (more/fewer composite types) and
  re-sweep; report the sweep table either way.

### 4. Telemetry (additive contract bump)

- Reuse `TelemetryRecord`: `execution_accuracy` = episode success (0/1),
  `query_valid` = "episode produced parseable actions" (invalid-action spam
  → False), `difficulty` = task-type bucket, `domain_id` = task type.
- New optional field `episode_stats: dict | None = None`:

```json
{
  "steps": 14,
  "invalid_actions": 2,
  "repeated_states": 1,
  "terminated_by": "success | step_cap | invalid_loop",
  "trajectory_id": "..."
}
```

- Full trajectories stored outside events.jsonl (e.g. `runs/trajectories/`)
  keyed by `trajectory_id` — the event log stays light; one log format rule
  holds.
- Drift channels: windowed success rate (primary); invalid-action rate and
  loop rate as leading indicators (they degrade before success does).

### 5. Teacher trace repair (verified by replay)

- Failed episode → teacher receives goal + condensed trajectory (actions +
  key observations) → outputs a repaired action plan.
- **Verification = replay**: run the repaired plan in the same env instance;
  keep only plans that succeed. This is the unit-test equivalent — no
  unverified teacher output enters memory.

### 6. Memory: utility-gated from day 1

Two channels, both gated:

- **Trajectory few-shots**: goal → condensed successful action sequence;
  retrieval by task type (later: by goal similarity).
- **KG (trap, fix) rules**: e.g. "microwaving requires holding the object" —
  distilled from failure clusters, same graph store pattern.

**Utility gate (TraceLift lesson,
[arXiv:2605.03862](https://arxiv.org/abs/2605.03862)):** a candidate memory
item is kept only if student success on ~10 validation episodes improves
(or at minimum does not regress) vs current memory. Teacher-verified ≠
useful-to-student; the gate measures consumer gain. Gate results are logged
(kept/rejected + delta) — rejection statistics are themselves a finding.

### 7. Measurement

- Windowed success-rate curve: baseline → shift-drop → post-learning recovery.
- Held-out `valid_unseen`, n ≥ 30 episodes: WITHOUT vs WITH memory, paired
  per task → McNemar; student+memory vs unaided teacher on the same tasks.
- Step-efficiency delta (median steps-to-success) as a secondary signal —
  memory can help even when success rate moves little.
- Same honesty rules: single-seed results labeled as such; nulls published.

### 8. Injection audit carried over

`injection_stats` (already in contracts) populated per episode: trajectories
available/injected, rules injected. The zero-injection diagnostic transfers
unchanged.

### 9. Spine escalation ladder (pre-registered pivots)

The four-stage spine is **provisional, not sacred**. Each rung fires on a
measured failure, in order — no debate, no vibes:

| Rung | Trigger (measured) | Spine change |
|---|---|---|
| 0 | — (start) | Current spine + utility-gated memory (§6) |
| 1 | Drift-batch learning too slow/sparse: < 5 gated memory items after full LEARN phase | **Always-learn**: memory update after every failed episode; detector demoted to regression alarm |
| 2 | Memory grows but held-out Δ ≤ 0 (McNemar-backed) with injection confirmed | **Memory bank + retrieval** replaces append-only list: versioned, pruned by utility, CorrectionAction becomes a memory delta |
| 3 | Rung 2 null on an in-band student across 2 seeds | **Channel swap**: plan/skill library instead of Q→A exemplars; if that also nulls, LoRA-on-verified-repairs (weights > prompts for weak students) |
| 4 | Teacher-on-drift economics broken (teacher cost dominates or drift never fires cleanly) | **Selective escalation**: student self-flags low-confidence episodes → teacher handles those → memory grows from escalations |

Each rung change = one seed, held-out, McNemar, documented in
`docs/FINDINGS_*` before the next rung is considered.

### 10. Build gate

B1 implementation starts **after** the coding in-band replication chain
(mistral-nemo + Qwen3.5-4B) lands and its findings are appended to
`docs/FINDINGS_CODING.md`:

- If in-band students show real self-lift → port the working recipe to B1 as-is.
- If null again → rung 2 (utility-gated memory bank) gets built and validated
  **in the coding domain first** (cheaper iterations), then ported.

### 11. Testing

Hermetic pytest as always: feed construction (task-type buckets, split
disjointness), episode→TelemetryRecord mapping with mocked env, utility-gate
logic with stubbed validation results, trajectory condensation, replay
verification plumbing with a fake env. No live API or env in CI.
