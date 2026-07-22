#!/usr/bin/env python3
"""Planner–executor eval on PRBench Corporate Finance.

Arms (per held-out task, k repeats each, averaged):
  A1  student alone
  A2  student + compute-matched self-critique→revise (no teacher)
  A4  student + Fable's <=300-token guidance hints
  A5  Fable alone (ceiling)

Headline: A4 > A2 (guidance beats equal compute); A4 vs A5 shows skill transferred
and lets us leak-check. Resumable: every (arm, task, rep) score cached to JSONL.
Nothing single-pass is trusted — run with --k 3 for anything shown to anyone.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adapters import prbench as pr  # noqa: E402
from contracts.schemas import AgentConfig  # noqa: E402
from harness import agent  # noqa: E402
from harness.agent import provider_fallback_count  # noqa: E402

SCORES = ROOT / "runs" / "prbench_planner_scores.jsonl"


def _cache() -> dict:
    c = {}
    if SCORES.exists():
        for line in SCORES.read_text().splitlines():
            d = json.loads(line)
            c[d["key"]] = d
    return c


def _append(rec: dict) -> None:
    SCORES.parent.mkdir(exist_ok=True)
    with open(SCORES, "a") as f:
        f.write(json.dumps(rec) + "\n")


def run_arm(arm: str, tid: str, rep: int, student_model: str, cache: dict,
            show: bool = False) -> float | None:
    key = f"{arm}|{tid}|{rep}"
    if key in cache:
        return cache[key]["norm"]
    task = pr.get_task(tid)
    cfg = AgentConfig(config_id=arm, model=student_model, few_shot_examples=[])
    agent._client = None
    meta = {}
    if arm == "A1":
        ans, meta = pr.generate_answer(task, cfg)
    elif arm == "A2":
        ans, meta = pr.answer_with_retries(task, cfg)
    elif arm == "A4":
        hints = pr.teacher_hints(task)
        meta = {"hint_chars": len(hints)}
        if show:
            print(f"\n--- Fable hint ({len(hints)} chars) ---\n{hints}\n---")
        ans, s = pr.generate_answer(task, cfg, hints=hints)
        meta.update(s)
    elif arm == "A5":
        ans = pr.answer_teacher_alone(task)
    else:
        raise ValueError(arm)
    g = pr.score_answer(tid, ans)
    rec = {"key": key, "arm": arm, "tid": tid, "rep": rep, "norm": g["normalized"],
           "ans_chars": len(ans), **meta}
    _append(rec)
    cache[key] = rec
    if show:
        print(f"[{arm}] task {tid[:8]} rep {rep}: {g['normalized']:.1f}/100  ({len(ans)} chars)")
    return g["normalized"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--student-model", default="deepseek/deepseek-v4-pro")
    ap.add_argument("--n", type=int, default=15, help="held-out tasks")
    ap.add_argument("--k", type=int, default=3, help="repeats per arm")
    ap.add_argument("--arms", default="A1,A2,A4,A5")
    ap.add_argument("--dry-run", action="store_true", help="1 task, k=1, print outputs")
    ap.add_argument("--time-budget-s", type=float, default=None)
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()

    man = pr.load_manifest()
    held = man["heldout_ids"]
    arms = args.arms.split(",")
    cache = _cache()

    if args.dry_run:
        tid = held[0]
        print(f"DRY RUN — 1 task ({tid}), k=1, arms={arms}")
        for arm in arms:
            run_arm(arm, tid, 0, args.student_model, cache, show=True)
        _summary([tid], arms, 1, cache)
        return

    tasks = held[: args.n]
    if not args.summarize:
        import time
        t0 = time.time()
        for tid in tasks:
            for arm in arms:
                for rep in range(args.k):
                    if args.time_budget_s and time.time() - t0 >= args.time_budget_s:
                        print("[time budget] pause"); _summary(tasks, arms, args.k, cache); return
                    try:
                        run_arm(arm, tid, rep, args.student_model, cache)
                    except Exception as e:
                        print(f"  {arm} {tid[:8]} r{rep} FAIL {str(e)[:70]}", flush=True)
    _summary(tasks, arms, args.k, cache)


def _summary(tasks, arms, k, cache) -> None:
    print(f"\n{'=' * 56}\n  PLANNER–EXECUTOR (per-task k-avg, then mean over tasks)")
    per_arm = {}
    for arm in arms:
        task_means = []
        for tid in tasks:
            reps = [cache[f"{arm}|{tid}|{r}"]["norm"] for r in range(k)
                    if f"{arm}|{tid}|{r}" in cache]
            if reps:
                task_means.append(mean(reps))
        per_arm[arm] = task_means
        if task_means:
            print(f"    {arm}: mean={mean(task_means):5.1f}  n_tasks={len(task_means)}")
    if "A4" in per_arm and "A2" in per_arm and per_arm["A4"] and per_arm["A2"]:
        n = min(len(per_arm["A4"]), len(per_arm["A2"]))
        gap = mean(per_arm["A4"][:n]) - mean(per_arm["A2"][:n])
        print(f"\n    HEADLINE  A4 − A2 = {gap:+.1f}  (guidance vs equal compute)")
    if all(a in per_arm and per_arm[a] for a in ("A1", "A4", "A5")):
        n = min(len(per_arm[a]) for a in ("A1", "A4", "A5"))
        a1, a4, a5 = (mean(per_arm[a][:n]) for a in ("A1", "A4", "A5"))
        if a5 > a1:
            print(f"    A1→A5 gap closed by A4: {100*(a4-a1)/(a5-a1):.0f}%  (A1={a1:.1f} A4={a4:.1f} A5={a5:.1f})")
    print(f"    provider fallbacks: {provider_fallback_count()}")
    print("=" * 56)


if __name__ == "__main__":
    main()
