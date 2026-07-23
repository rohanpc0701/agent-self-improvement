#!/usr/bin/env python3
"""EVAL frozen contrastive memory on PRBench Corp-Finance HELD-OUT tasks.

Arms (per held-out task, k reps, averaged):
  PLAIN   student alone
  REFINE  student self-critique→revise (compute-matched control)
  MEM     student + frozen memory
Delta reported vs BOTH: MEM−PLAIN (headline number) and MEM−REFINE (honest bar).
Resumable: runs/prbench_memory_scores.jsonl. Run --k 3 for anything trusted.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adapters import prbench as pr  # noqa: E402
from contracts.schemas import AgentConfig, FewShotExample  # noqa: E402
from harness import agent  # noqa: E402
from harness.agent import provider_fallback_count  # noqa: E402

SCORES = ROOT / "runs" / "prbench_memory_scores.jsonl"
MEMORY = ROOT / "runs" / "prbench_memory.json"


def _cache() -> dict:
    c = {}
    if SCORES.exists():
        for line in SCORES.read_text().splitlines():
            d = json.loads(line)
            c[d["key"]] = d
    return c


def _load_memory() -> list[FewShotExample]:
    m = json.loads(MEMORY.read_text())
    return [FewShotExample(**{k: it[k] for k in ("question", "correct_output", "domain_id", "source")})
            for it in m["items"]]


def run(arm: str, tid: str, rep: int, model: str, mem, cache: dict) -> float | None:
    key = f"{arm}|{tid}|{rep}"
    if key in cache:
        return cache[key]["norm"]
    task = pr.get_task(tid)
    cfg = AgentConfig(config_id=arm, model=model, few_shot_examples=[])
    agent._client = None
    if arm == "PLAIN":
        ans, _ = pr.generate_answer(task, cfg)
    elif arm == "REFINE":
        ans, _ = pr.answer_with_refine(task, cfg)
    elif arm == "MEM":
        ans, _ = pr.generate_answer(task, cfg, memory=mem)
    elif arm == "TEACHER":  # A5 ceiling: Fable answers the held-out task directly
        ans = pr.answer_teacher_alone(task)
    else:
        raise ValueError(arm)
    g = pr.score_answer(tid, ans)
    rec = {"key": key, "arm": arm, "tid": tid, "rep": rep, "norm": g["normalized"]}
    SCORES.parent.mkdir(exist_ok=True)
    with open(SCORES, "a") as f:
        f.write(json.dumps(rec) + "\n")
    cache[key] = rec
    return g["normalized"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--student-model", default="deepseek/deepseek-v4-pro")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--arms", default="PLAIN,REFINE,MEM")
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()

    man = pr.load_manifest()
    tasks = man["heldout_ids"][: args.n]
    arms = args.arms.split(",")
    mem = _load_memory()
    cache = _cache()
    print(f"eval: {len(tasks)} held-out, k={args.k}, memory={len(mem)} lessons")

    if not args.summarize:
        for tid in tasks:
            for arm in arms:
                for rep in range(args.k):
                    try:
                        run(arm, tid, rep, args.student_model, mem, cache)
                    except Exception as e:
                        print(f"  {arm} {tid[:8]} r{rep} FAIL {str(e)[:60]}", flush=True)
        cache = _cache()

    per = {}
    for arm in arms:
        tm = []
        for tid in tasks:
            reps = [cache[f"{arm}|{tid}|{r}"]["norm"] for r in range(args.k)
                    if f"{arm}|{tid}|{r}" in cache]
            if reps:
                tm.append(mean(reps))
        per[arm] = tm
    print(f"\n{'=' * 52}\n  CONTRASTIVE MEMORY (per-task k-avg → mean over tasks)")
    for arm in arms:
        if per[arm]:
            print(f"    {arm:7}: {mean(per[arm]):5.1f}  n={len(per[arm])}")
    if per.get("MEM") and per.get("PLAIN"):
        n = min(len(per["MEM"]), len(per["PLAIN"]))
        print(f"\n    DELTA  MEM − PLAIN  = {mean(per['MEM'][:n]) - mean(per['PLAIN'][:n]):+.1f}")
    if per.get("MEM") and per.get("REFINE"):
        n = min(len(per["MEM"]), len(per["REFINE"]))
        print(f"    DELTA  MEM − REFINE = {mean(per['MEM'][:n]) - mean(per['REFINE'][:n]):+.1f}  (honest bar)")
    print(f"    provider fallbacks: {provider_fallback_count()}")
    print("=" * 52)


if __name__ == "__main__":
    main()
