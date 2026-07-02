#!/usr/bin/env python3
"""Multi-seed probe: measure hard-bucket delta stability across RNG seeds.

Cheap validation (no full 240-call loop): for each seed, runs unique held-out
questions twice (without vs with same-DB gold examples) like --probe.

Usage (from repo root, student configured):
  export AGENT_BASE_URL=http://localhost:11434/v1
  export AGENT_MODEL=qwen2.5:1.5b-instruct
  python3 scripts/multi_seed_eval.py
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from contracts.schemas import AgentConfig, FewShotExample
from harness.feed import FeedItem, build_stream
from harness.runner import run_item
from orchestrator import _BASE_MODEL, _unique_acc


def probe_seed(seed: int, full: bool) -> tuple[float, float, float]:
    from harness.spider import load_questions

    n = 80 if full else 40
    items = build_stream(
        load_questions(),
        n_baseline=n,
        n_degraded=n,
        n_recovery=n,
        seed=seed,
        same_db_split=True,
        baseline_easy_only=True,
    )
    base = AgentConfig(config_id=f"probe-{seed}", model=_BASE_MODEL, few_shot_examples=[])

    seen: set[str] = set()
    heldout: list[FeedItem] = []
    for it in items:
        if it.phase == "recovery" and it.question_id not in seen:
            heldout.append(it)
            seen.add(it.question_id)

    heldout_ids = {it.question_id for it in heldout}
    learn_by_db: dict[str, list[FeedItem]] = {}
    seen_learn: dict[str, set[str]] = {}
    for it in items:
        if it.phase == "degraded" and it.question_id not in heldout_ids:
            if it.db_id not in seen_learn:
                seen_learn[it.db_id] = set()
            if it.question_id not in seen_learn[it.db_id]:
                learn_by_db.setdefault(it.db_id, []).append(it)
                seen_learn[it.db_id].add(it.question_id)

    wo_pairs: list[tuple[str, float]] = []
    w_pairs: list[tuple[str, float]] = []
    for it in heldout:
        rec_wo = run_item(it, base, use_rules=False)
        if rec_wo is None:
            continue
        examples = [
            FewShotExample(question=l.question, correct_sql=l.gold_sql, db_id=l.db_id, source="gold")
            for l in learn_by_db.get(it.db_id, [])
        ]
        cfg_w = base.model_copy(update={"few_shot_examples": examples})
        rec_w = run_item(it, cfg_w, use_rules=False)
        if rec_w is None:
            continue
        wo_pairs.append((it.question, rec_wo.execution_accuracy))
        w_pairs.append((it.question, rec_w.execution_accuracy))

    wo_acc, _ = _unique_acc(wo_pairs)
    w_acc, _ = _unique_acc(w_pairs)
    return wo_acc, w_acc, w_acc - wo_acc


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 99])
    p.add_argument("--full", action="store_true", help="80 questions per phase (default 40)")
    args = p.parse_args()

    from harness.agent import require_api_key
    require_api_key()

    deltas = []
    print(f"Multi-seed probe ({len(args.seeds)} seeds, {'full' if args.full else 'n=40'})")
    for seed in args.seeds:
        wo, w, d = probe_seed(seed, args.full)
        deltas.append(d)
        print(f"  seed={seed}: without={wo:.3f}  with={w:.3f}  delta={d:+.3f}")

    if len(deltas) >= 2:
        mean_d = statistics.mean(deltas)
        std_d = statistics.stdev(deltas)
        print(f"\nSummary: mean delta = {mean_d:+.3f} ± {std_d:.3f} (n={len(deltas)} seeds)")
    else:
        print(f"\nSummary: delta = {deltas[0]:+.3f}")


if __name__ == "__main__":
    main()
