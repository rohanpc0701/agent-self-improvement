#!/usr/bin/env python3
"""Held-out TraceLift eval — A1 (student alone) vs A4 (student + frozen memory).

Also supports A5 (teacher alone). Writes incremental JSONL under runs/,
supports --resume / --max-new / --time-budget-s. Reports GAP with paired
bootstrap into stdout + docs/FINDINGS_FINANCE.md §F when complete.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adapters.finance import (  # noqa: E402
    generate_answer,
    get_problem,
    load_manifest,
)
from analysis.bootstrap import paired_bootstrap  # noqa: E402
from contracts.schemas import AgentConfig, FewShotExample  # noqa: E402
from correction.judge import JudgeParseError, grade, rubric_max_points  # noqa: E402

RUNS = ROOT / "runs"
DEFAULT_STUDENT = "qwen/qwen3.6-27b"
DEFAULT_TEACHER = "z-ai/glm-5.2"
DEFAULT_MEMORY = RUNS / "finance_tracelift_memory.json"


def _slug(model: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", model)


def _append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def load_frozen_memory(path: Path) -> list[FewShotExample]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [FewShotExample(**x) for x in data.get("items", [])]


def heldout_ids(seed: int = 42) -> list[str]:
    m = load_manifest()
    ids = list(m["heldout_ids"])
    # Stable category round-robin
    by_cat: dict[str, list[str]] = defaultdict(list)
    for qid in ids:
        by_cat[get_problem(qid)["category"]].append(qid)
    rng = random.Random(seed)
    for v in by_cat.values():
        rng.shuffle(v)
    ordered: list[str] = []
    pools = sorted(by_cat.items())
    while any(p for _, p in pools):
        for _, pool in pools:
            if pool:
                ordered.append(pool.pop())
    return ordered


def _paths(arm: str, model: str) -> tuple[Path, Path]:
    slug = _slug(model)
    return (
        RUNS / f"finance_eval_{arm}_{slug}_answers.jsonl",
        RUNS / f"finance_eval_{arm}_{slug}_grades.jsonl",
    )


def generate_arm(
    arm: str,
    model: str,
    ids: list[str],
    *,
    memory: list[FewShotExample] | None,
    max_new: int | None,
    time_budget_s: float | None,
    resume: bool,
) -> Path:
    ans_path, _ = _paths(arm, model)
    done = set()
    if resume:
        for r in _load_jsonl(ans_path):
            if r.get("answer") and not r.get("error"):
                done.add(r["id"])
    use_mem = bool(memory)
    os.environ["AGENT_USE_EXAMPLES"] = "1" if use_mem else "0"
    cfg = AgentConfig(
        config_id=f"eval-{arm}",
        model=model,
        few_shot_examples=list(memory or []),
    )
    pending = [qid for qid in ids if qid not in done]
    print(
        f"[eval/{arm}] model={model} pending={len(pending)} mem={len(memory or [])}",
        flush=True,
    )
    t0 = time.time()
    n_new = 0
    for i, qid in enumerate(pending, 1):
        if max_new is not None and n_new >= max_new:
            break
        if time_budget_s is not None and (time.time() - t0) >= time_budget_s:
            print(f"[eval/{arm}] time budget; pause", flush=True)
            break
        p = get_problem(qid)
        try:
            answer, stats = generate_answer(
                p["question"], cfg, category=p["category"], max_tokens=2048
            )
            if not answer.strip():
                raise RuntimeError("empty answer")
        except Exception as exc:
            print(f"  [ans {i}/{len(pending)}] {qid} FAILED ({exc})", flush=True)
            low = str(exc).lower()
            if any(k in low for k in ("insufficient", "401", "403")):
                raise SystemExit(f"FATAL provider error: {exc}") from exc
            continue
        _append(
            ans_path,
            {
                "id": qid,
                "arm": arm,
                "model": model,
                "category": p["category"],
                "answer": answer,
                "injection_stats": stats,
                "ts": time.time(),
            },
        )
        n_new += 1
        print(f"  [ans {i}/{len(pending)}] {qid} ok len={len(answer)}", flush=True)
    return ans_path


def grade_arm(
    arm: str,
    model: str,
    ids: list[str],
    *,
    max_new: int | None,
    time_budget_s: float | None,
    resume: bool,
) -> Path:
    ans_path, grade_path = _paths(arm, model)
    answers = {
        r["id"]: r
        for r in _load_jsonl(ans_path)
        if r.get("answer") and not r.get("error")
    }
    done = set()
    if resume:
        for r in _load_jsonl(grade_path):
            if "normalized" in r and not r.get("error"):
                done.add(r["id"])
    pending = [qid for qid in ids if qid not in done and qid in answers]
    print(f"[grade/{arm}] pending={len(pending)}", flush=True)
    t0 = time.time()
    n_new = 0
    random.Random(time.time()).shuffle(pending)
    for i, qid in enumerate(pending, 1):
        if max_new is not None and n_new >= max_new:
            break
        if time_budget_s is not None and (time.time() - t0) >= time_budget_s:
            print(f"[grade/{arm}] time budget; pause", flush=True)
            break
        p = get_problem(qid)
        try:
            rubric_max_points(p["rubric"])
        except JudgeParseError as exc:
            _append(
                grade_path,
                {
                    "id": qid,
                    "arm": arm,
                    "error": f"ungradable_rubric: {exc}",
                    "ts": time.time(),
                },
            )
            n_new += 1
            continue
        try:
            result = grade(
                question=p["question"],
                rubric=p["rubric"],
                answer=answers[qid]["answer"],
            )
        except Exception as exc:
            print(f"  [grade {i}/{len(pending)}] {qid} FAILED ({exc})", flush=True)
            continue
        _append(
            grade_path,
            {
                "id": qid,
                "arm": arm,
                "model": model,
                "category": p["category"],
                "normalized": result["normalized"],
                "traps_hit": result.get("traps_hit") or [],
                "ts": time.time(),
            },
        )
        n_new += 1
        print(
            f"  [grade {i}/{len(pending)}] {qid} norm={result['normalized']:.2f}",
            flush=True,
        )
    return grade_path


def scores_by_id(arm: str, model: str) -> dict[str, dict]:
    _, grade_path = _paths(arm, model)
    latest: dict[str, dict] = {}
    for r in _load_jsonl(grade_path):
        if "normalized" in r and not r.get("error"):
            latest[r["id"]] = r
    return latest


def summarize_a1_a4(student: str) -> dict:
    a1 = scores_by_id("a1", student)
    a4 = scores_by_id("a4", student)
    common = sorted(set(a1) & set(a4))
    if not common:
        return {"n": 0, "status": "incomplete"}
    s1 = [float(a1[i]["normalized"]) for i in common]
    s4 = [float(a4[i]["normalized"]) for i in common]
    boot = paired_bootstrap(s1, s4, n_boot=10_000, seed=42)
    # Per-category
    by_cat: dict[str, list[tuple[float, float]]] = defaultdict(list)
    traps_a1: Counter[str] = Counter()
    traps_a4: Counter[str] = Counter()
    for qid in common:
        cat = a1[qid]["category"]
        by_cat[cat].append((float(a1[qid]["normalized"]), float(a4[qid]["normalized"])))
        for t in a1[qid].get("traps_hit") or []:
            traps_a1[t] += 1
        for t in a4[qid].get("traps_hit") or []:
            traps_a4[t] += 1
    cat_gap = {
        c: sum(b - a for a, b in pairs) / len(pairs)
        for c, pairs in sorted(by_cat.items())
    }
    return {
        "n": len(common),
        "mean_a1": sum(s1) / len(s1),
        "mean_a4": sum(s4) / len(s4),
        "GAP_alone": boot["delta"],
        "ci_low": boot["ci_low"],
        "ci_high": boot["ci_high"],
        "p_value": boot["p_value"],
        "by_category_gap": cat_gap,
        "trap_hits_a1": dict(traps_a1),
        "trap_hits_a4": dict(traps_a4),
        "trap_hit_rate_a1": sum(traps_a1.values()) / len(common),
        "trap_hit_rate_a4": sum(traps_a4.values()) / len(common),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--student-model", default=os.environ.get("STUDENT_MODEL", DEFAULT_STUDENT))
    ap.add_argument("--teacher-model", default=os.environ.get("TEACHER_MODEL", DEFAULT_TEACHER))
    ap.add_argument("--memory", type=Path, default=DEFAULT_MEMORY)
    ap.add_argument(
        "--arm",
        choices=("a1", "a4", "a5", "all"),
        default="all",
    )
    ap.add_argument("--grades-only", action="store_true")
    ap.add_argument("--max-new", type=int, default=None)
    ap.add_argument("--time-budget-s", type=float, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args(argv)

    ids = heldout_ids(args.seed)
    memory = load_frozen_memory(args.memory)

    if args.summarize_only:
        summary = summarize_a1_a4(args.student_model)
        print(json.dumps(summary, indent=2))
        (RUNS / "finance_eval_a1_a4_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        return 0

    arms: list[tuple[str, str, list[FewShotExample] | None]] = []
    if args.arm in ("a1", "all"):
        arms.append(("a1", args.student_model, None))
    if args.arm in ("a4", "all"):
        if not memory:
            print(f"[warn] no frozen memory at {args.memory}; A4 will be bare", flush=True)
        arms.append(("a4", args.student_model, memory))
    if args.arm in ("a5", "all"):
        arms.append(("a5", args.teacher_model, None))

    for arm, model, mem in arms:
        if not args.grades_only:
            generate_arm(
                arm,
                model,
                ids,
                memory=mem,
                max_new=args.max_new,
                time_budget_s=args.time_budget_s,
                resume=args.resume,
            )
        grade_arm(
            arm,
            model,
            ids,
            max_new=args.max_new,
            time_budget_s=args.time_budget_s,
            resume=args.resume,
        )

    summary = summarize_a1_a4(args.student_model)
    print(json.dumps(summary, indent=2), flush=True)
    (RUNS / "finance_eval_a1_a4_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
