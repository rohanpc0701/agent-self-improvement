#!/usr/bin/env python3
"""Run-to-run variance check: repeat identical arms at temperature 0.

Prime-side nondeterminism (batching / MoE routing) showed a bare-arm shift of
0.414 → 0.517 between passes of the SAME prompts. This script quantifies that
noise: k repeats of the bare arm (and optionally examples arm) on the same
held-out questions. Writes NOTHING to events.jsonl — safe to run alongside
other pipelines.

Usage:
    python3 scripts/variance_check.py --model mistralai/mistral-nemo \
        --repeats 4 --events runs/nemo_artifacts/events.jsonl
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adapters.coding import generate_code, verify_solution, _index  # noqa: E402
from contracts.eventlog import read_events  # noqa: E402
from contracts.schemas import AgentConfig  # noqa: E402


def _heldout_pool(seed: int, heldout_frac: float) -> list:
    from adapters.registry import get_adapter

    adapter = get_adapter("coding")
    items = adapter.build_hard_curriculum_feed(seed=seed, db_heldout_frac=heldout_frac)
    seen: set[str] = set()
    pool = []
    for it in items:
        if it.phase == "recovery" and it.difficulty == "hard" and it.question_id not in seen:
            pool.append(it)
            seen.add(it.question_id)
    return pool


def run_arm(model: str, pool: list, examples: list, label: str) -> dict[str, float]:
    config = AgentConfig(config_id=f"variance-{label}", model=model,
                         few_shot_examples=examples)
    per_q: dict[str, float] = {}
    for item in pool:
        problem = _index().get(item.question_id)
        if problem is None:
            continue
        text, *_ = generate_code(item.question, config, topic=item.domain_id,
                                 use_rules=False)
        acc, _, _ = verify_solution(text, problem)
        per_q[item.question_id] = acc
    return per_q


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--repeats", type=int, default=4)
    ap.add_argument("--events", default=None,
                    help="events.jsonl with a CorrectionAction (enables examples arm)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--heldout-frac", type=float, default=0.5)
    args = ap.parse_args()

    os.environ.setdefault("AGENT_BASE_URL", "https://api.pinference.ai/api/v1")
    pool = _heldout_pool(args.seed, args.heldout_frac)
    print(f"variance check: model={args.model} n={len(pool)} repeats={args.repeats}")

    examples = []
    if args.events:
        corrections = read_events(only="correction", path=args.events)
        if corrections:
            examples = corrections[-1].new_few_shot_examples
            print(f"loaded {len(examples)} frozen examples from {args.events}")

    arms = {"none": []}
    if examples:
        arms["examples"] = []

    flip_counts: dict[str, Counter] = {a: Counter() for a in arms}
    for arm in arms:
        exs = examples if arm == "examples" else []
        prev: dict[str, float] | None = None
        for r in range(args.repeats):
            per_q = run_arm(args.model, pool, exs, f"{arm}-r{r}")
            acc = mean(per_q.values())
            arms[arm].append(acc)
            flips = 0
            if prev is not None:
                flips = sum(1 for q in per_q if q in prev and per_q[q] != prev[q])
                for q in per_q:
                    if q in prev and per_q[q] != prev[q]:
                        flip_counts[arm][q] += 1
            prev = per_q
            print(f"  [{arm:<8}] repeat {r + 1}/{args.repeats}: acc={acc:.3f}"
                  + (f"  flips_vs_prev={flips}" if r else ""), flush=True)

    print(f"\n{'=' * 60}\n  VARIANCE (temp=0, identical prompts)")
    for arm, accs in arms.items():
        print(f"    {arm:<8}: mean={mean(accs):.3f}  sd={pstdev(accs):.3f}  "
              f"range=[{min(accs):.3f},{max(accs):.3f}]  runs={len(accs)}")
        unstable = [q for q, c in flip_counts[arm].items() if c]
        print(f"              unstable questions: {len(unstable)}")
    if len(arms) == 2:
        deltas = [e - n for n, e in zip(arms["none"], arms["examples"])]
        print(f"    per-repeat Δ(examples-none): "
              f"{['%+.3f' % d for d in deltas]}")
    print("=" * 60)


if __name__ == "__main__":
    main()
