#!/usr/bin/env python3
"""FinancePro Phase 0 — headroom probe (G0.4) + held-out baselines (G0.3).

Modes:
  headroom      — 3 candidates × 20 stratified VALIDATION Qs (bare prompt)
  band-recheck  — re-judge chosen student's 20 answers; MAD gate on band-range
  baselines     — 120 held-out student-alone + teacher-alone

All modes write incremental JSONL under runs/ and support --resume.
Timeouts: regenerate (never silent skip); default AGENT_TIMEOUT_S=240.
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
    rubric_for,
)
from analysis.bootstrap import mean_bootstrap, paired_bootstrap  # noqa: E402
from contracts.schemas import AgentConfig  # noqa: E402
from correction import judge as judge_mod  # noqa: E402
from correction.judge import JudgeParseError, grade, rubric_max_points  # noqa: E402

RUNS = ROOT / "runs"
DEFAULT_CANDIDATES = [
    "qwen/qwen3-8b",
    "qwen/qwen3-30b-a3b-instruct-2507",
    "qwen/qwen3.6-27b",
]
FALLBACK_LARGER = "qwen/qwen3.5-35b-a3b"
# Smallest-first for student selection among in-band scorers.
SIZE_ORDER = [
    "qwen/qwen3-8b",
    "qwen/qwen3.6-27b",
    "qwen/qwen3-30b-a3b-instruct-2507",
    "qwen/qwen3.5-35b-a3b",
]
BAND_LO, BAND_HI = 15.0, 40.0
TEACHER_DEFAULT = "minimax/minimax-m3"
MAX_GEN_ATTEMPTS = 3


def _slug(model: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", model)


def stratified_ids(split: str, n: int | None, seed: int) -> list[str]:
    m = load_manifest()
    key = {"validation": "validation_ids", "heldout": "heldout_ids", "train": "train_ids"}[
        split
    ]
    ids = list(m[key])
    if n is None or n >= len(ids):
        # Full split, stable order by category round-robin for resume predictability.
        by_cat: dict[str, list[str]] = defaultdict(list)
        for qid in ids:
            by_cat[get_problem(qid)["category"]].append(qid)
        rng = random.Random(seed)
        for v in by_cat.values():
            rng.shuffle(v)
        ordered: list[str] = []
        pools = sorted(by_cat.items(), key=lambda kv: kv[0])
        while any(p for _, p in pools):
            for _, pool in pools:
                if pool:
                    ordered.append(pool.pop())
        return ordered
    by_cat: dict[str, list[str]] = defaultdict(list)
    for qid in ids:
        by_cat[get_problem(qid)["category"]].append(qid)
    rng = random.Random(seed)
    for v in by_cat.values():
        rng.shuffle(v)
    picked: list[str] = []
    pools = sorted(by_cat.items(), key=lambda kv: len(kv[1]))
    while len(picked) < n and any(p for _, p in pools):
        for _, pool in pools:
            if pool and len(picked) < n:
                picked.append(pool.pop())
    return picked


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _done_ids(path: Path, *, require_ok: bool = True) -> set[str]:
    done: set[str] = set()
    for r in _load_jsonl(path):
        if require_ok and (r.get("error") or not (r.get("answer") or "").strip()):
            continue
        done.add(r["id"])
    return done


def _append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def _answers_path(tag: str, model: str) -> Path:
    return RUNS / f"finance_{tag}_{_slug(model)}_answers.jsonl"


def _grades_path(tag: str, model: str) -> Path:
    return RUNS / f"finance_{tag}_{_slug(model)}_grades.jsonl"


def generate_for_ids(
    ids: list[str],
    model: str,
    tag: str,
    *,
    resume: bool,
    max_new: int | None = None,
    time_budget_s: float | None = None,
) -> Path:
    """Generate bare-prompt answers. Retries on failure; never silent-skip."""
    os.environ["AGENT_USE_EXAMPLES"] = "0"
    path = _answers_path(tag, model)
    done = _done_ids(path) if resume else set()
    # Drop incomplete error rows from resume consideration by regenerating them:
    # if resume and an id has only error rows, it's not in done → will retry.
    cfg = AgentConfig(config_id=f"finance-{tag}", model=model, few_shot_examples=[])
    t0 = time.time()
    n_new = 0
    pending = [qid for qid in ids if qid not in done]
    print(
        f"[gen/{tag}] model={model} pending={len(pending)}/{len(ids)} path={path.name}",
        flush=True,
    )
    for i, qid in enumerate(pending, 1):
        if max_new is not None and n_new >= max_new:
            print(f"[gen/{tag}] max-new={max_new} reached; pause", flush=True)
            break
        if time_budget_s is not None and (time.time() - t0) >= time_budget_s:
            print(f"[gen/{tag}] time budget {time_budget_s}s; pause", flush=True)
            break
        p = get_problem(qid)
        answer = ""
        err = None
        for attempt in range(1, MAX_GEN_ATTEMPTS + 1):
            try:
                answer, _ = generate_answer(
                    p["question"], cfg, p["category"], temperature=0.0, max_tokens=2048
                )
                if answer.strip():
                    err = None
                    break
                err = "empty_answer"
            except Exception as exc:
                err = f"{exc.__class__.__name__}: {exc}"
                print(
                    f"  [gen {i}/{len(pending)}] {qid} attempt {attempt}/{MAX_GEN_ATTEMPTS} "
                    f"FAILED ({err})",
                    flush=True,
                )
                # Do not burn the chunk retrying hard quota / auth failures.
                low = err.lower()
                if any(
                    k in low
                    for k in (
                        "insufficient_funds",
                        "insufficient balance",
                        "insufficient_quota",
                        "401",
                        "403",
                    )
                ):
                    raise SystemExit(
                        f"FATAL provider error — refill funds or switch endpoint: {err}"
                    ) from exc
                time.sleep(min(2 ** attempt, 15))
        if err or not answer.strip():
            # Record failure but do NOT mark done for resume (require_ok filters it).
            # Write a distinct error record for logging; next --resume will retry.
            _append(
                path,
                {
                    "id": qid,
                    "category": p["category"],
                    "model": model,
                    "answer": "",
                    "error": err or "empty_answer",
                    "ts": time.time(),
                },
            )
            print(
                f"  [gen {i}/{len(pending)}] {qid} STILL FAILED after "
                f"{MAX_GEN_ATTEMPTS} attempts — will regenerate on --resume",
                flush=True,
            )
            n_new += 1
            continue
        _append(
            path,
            {
                "id": qid,
                "category": p["category"],
                "model": model,
                "answer": answer,
                "ts": time.time(),
            },
        )
        n_new += 1
        print(
            f"  [gen {i}/{len(pending)}] {qid} chars={len(answer)}",
            flush=True,
        )
    return path


def _latest_ok_answers(path: Path) -> dict[str, dict]:
    """Last non-error answer per id."""
    out: dict[str, dict] = {}
    for r in _load_jsonl(path):
        if r.get("error") or not (r.get("answer") or "").strip():
            continue
        out[r["id"]] = r
    return out


def grade_for_ids(
    ids: list[str],
    model: str,
    tag: str,
    *,
    resume: bool,
    judge_passes: int,
    pass_label: int = 1,
    max_new: int | None = None,
    time_budget_s: float | None = None,
) -> Path:
    path = _grades_path(tag, model)
    done: set[tuple[str, int]] = set()
    if resume:
        for r in _load_jsonl(path):
            done.add((r["id"], int(r.get("pass_i", 1))))
    answers = _latest_ok_answers(_answers_path(tag, model))
    t0 = time.time()
    n_new = 0
    # Only grade ids that already have answers — unanswered wait for a later resume.
    pending = [
        qid
        for qid in ids
        if (qid, pass_label) not in done and qid in answers
    ]
    skipped_no_ans = sum(
        1 for qid in ids if (qid, pass_label) not in done and qid not in answers
    )
    print(
        f"[grade/{tag}] model={model} pass={pass_label} pending={len(pending)} "
        f"(no-answer-yet={skipped_no_ans}) judge_passes={judge_passes}",
        flush=True,
    )
    # Shuffle so persistent empty-judge failures don't monopolize every chunk.
    random.Random(time.time()).shuffle(pending)
    failed: list[str] = []
    queue = list(pending)

    def _grade_one(qid: str, i: int, total: int) -> bool:
        """Return True if a row was written (success or permanent skip)."""
        nonlocal n_new
        row_a = answers[qid]
        p = get_problem(qid)
        try:
            rubric_max_points(p["rubric"])
        except JudgeParseError as exc:
            _append(
                path,
                {
                    "id": qid,
                    "category": p["category"],
                    "model": model,
                    "pass_i": pass_label,
                    "error": f"ungradable_rubric: {exc}",
                    "ts": time.time(),
                },
            )
            n_new += 1
            print(f"  [grade {i}/{total}] {qid} SKIP ungradable rubric", flush=True)
            return True
        try:
            result = grade(
                question=p["question"],
                rubric=rubric_for(qid, role="judge"),
                answer=row_a["answer"],
                passes=judge_passes,
            )
        except Exception as exc:
            print(
                f"  [grade {i}/{total}] {qid} FAILED ({exc}) — defer",
                flush=True,
            )
            return False
        _append(
            path,
            {
                "id": qid,
                "category": p["category"],
                "model": model,
                "pass_i": pass_label,
                "normalized": result["normalized"],
                "total": result["total"],
                "max": result["max"],
                "traps_hit": result.get("traps_hit") or [],
                "bonuses": result.get("bonuses") or [],
                "ts": time.time(),
            },
        )
        n_new += 1
        print(
            f"  [grade {i}/{total}] {qid} norm={result['normalized']:.2f} "
            f"traps={result.get('traps_hit')}",
            flush=True,
        )
        return True

    for i, qid in enumerate(queue, 1):
        if max_new is not None and n_new >= max_new:
            print(f"[grade/{tag}] max-new={max_new} reached; pause", flush=True)
            break
        if time_budget_s is not None and (time.time() - t0) >= time_budget_s:
            print(f"[grade/{tag}] time budget {time_budget_s}s; pause", flush=True)
            break
        if not _grade_one(qid, i, len(queue)):
            failed.append(qid)

    # One deferred pass on failures if budget remains.
    if failed and (time_budget_s is None or (time.time() - t0) < time_budget_s):
        print(f"[grade/{tag}] retrying {len(failed)} deferred failures", flush=True)
        for i, qid in enumerate(failed, 1):
            if max_new is not None and n_new >= max_new:
                break
            if time_budget_s is not None and (time.time() - t0) >= time_budget_s:
                print(f"[grade/{tag}] time budget {time_budget_s}s; pause", flush=True)
                break
            _grade_one(qid, i, len(failed))
    return path


def mean_for_model(tag: str, model: str, pass_i: int = 1) -> dict:
    # Keep latest OK grade per id (concurrent resumes may append duplicates).
    latest: dict[str, dict] = {}
    n_skip = 0
    for r in _load_jsonl(_grades_path(tag, model)):
        if int(r.get("pass_i", 1)) != pass_i:
            continue
        if r.get("error") or "normalized" not in r:
            n_skip += 1
            continue
        latest[r["id"]] = r
    scores = []
    traps: Counter[str] = Counter()
    by_cat: dict[str, list[float]] = defaultdict(list)
    ids_scores: dict[str, float] = {}
    for qid, r in latest.items():
        scores.append(float(r["normalized"]))
        by_cat[r["category"]].append(float(r["normalized"]))
        ids_scores[qid] = float(r["normalized"])
        for t in r.get("traps_hit") or []:
            traps[t] += 1
    if not scores:
        return {
            "n": 0,
            "n_skip": n_skip,
            "mean": float("nan"),
            "by_category": {},
            "traps": {},
        }
    return {
        "n": len(scores),
        "n_skip": n_skip,
        "mean": sum(scores) / len(scores),
        "scores": scores,
        "by_category": {c: sum(v) / len(v) for c, v in sorted(by_cat.items())},
        "by_category_n": {c: len(v) for c, v in sorted(by_cat.items())},
        "traps": dict(traps),
        "ids_scores": ids_scores,
    }


def _gradable_ids(ids: list[str]) -> list[str]:
    out = []
    for qid in ids:
        try:
            rubric_max_points(get_problem(qid)["rubric"])
            out.append(qid)
        except JudgeParseError:
            continue
    return out


def baselines_complete(ids: list[str], student: str, teacher: str) -> bool:
    """True when every gradable id has an OK answer + grade for both arms."""
    gradable = set(_gradable_ids(ids))
    for model in (student, teacher):
        ans = set(_latest_ok_answers(_answers_path("heldout", model)))
        graded = {
            r["id"]
            for r in _load_jsonl(_grades_path("heldout", model))
            if int(r.get("pass_i", 1)) == 1
            and "normalized" in r
            and not r.get("error")
        }
        if not gradable <= ans:
            return False
        if not gradable <= graded:
            return False
    return True


def pick_student(means: dict[str, float]) -> dict:
    """Smallest model with mean in [15, 40]."""
    in_band = [
        m
        for m in SIZE_ORDER
        if m in means and BAND_LO <= means[m] <= BAND_HI
    ]
    if in_band:
        chosen = in_band[0]
        return {
            "chosen": chosen,
            "reason": f"smallest in-band [{BAND_LO}-{BAND_HI}]",
            "in_band": in_band,
            "means": means,
        }
    if means and all(v < BAND_LO for v in means.values()):
        return {
            "chosen": None,
            "reason": f"all < {BAND_LO}; probe {FALLBACK_LARGER}",
            "fallback": FALLBACK_LARGER,
            "means": means,
        }
    if means and all(v > BAND_HI for v in means.values()):
        return {
            "chosen": None,
            "reason": f"all > {BAND_HI}; add smaller qwen",
            "means": means,
        }
    # Mixed but none in band — pick closest below band, else closest above.
    ranked = sorted(means.items(), key=lambda kv: (abs(kv[1] - 27.5), SIZE_ORDER.index(kv[0]) if kv[0] in SIZE_ORDER else 99))
    return {
        "chosen": None,
        "reason": "no model in band; see means",
        "nearest": ranked[0] if ranked else None,
        "means": means,
    }


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denx = sum((x - mx) ** 2 for x in xs) ** 0.5
    deny = sum((y - my) ** 2 for y in ys) ** 0.5
    if denx == 0 or deny == 0:
        return float("nan")
    return num / (denx * deny)


def summarize_headroom(models: list[str], tag: str = "headroom") -> dict:
    means = {}
    detail = {}
    for m in models:
        info = mean_for_model(tag, m, pass_i=1)
        if info["n"]:
            means[m] = info["mean"]
            detail[m] = {k: info[k] for k in ("n", "mean", "by_category", "traps")}
    decision = pick_student(means)
    summary = {
        "tag": tag,
        "means": means,
        "detail": detail,
        "decision": decision,
        "band": [BAND_LO, BAND_HI],
        "JUDGE_PASSES": judge_mod.JUDGE_PASSES,
    }
    out = RUNS / "finance_headroom_summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def summarize_band_recheck(model: str, tag: str = "headroom") -> dict:
    by_id: dict[str, dict[int, float]] = defaultdict(dict)
    for r in _load_jsonl(_grades_path(tag, model)):
        by_id[r["id"]][int(r["pass_i"])] = float(r["normalized"])
    pairs = [(v[1], v[2]) for v in by_id.values() if 1 in v and 2 in v]
    if not pairs:
        raise SystemExit("band-recheck: no complete grade pairs (need pass_i 1 and 2)")
    a = [p[0] for p in pairs]
    b = [p[1] for p in pairs]
    mad = sum(abs(x - y) for x, y in pairs) / len(pairs)
    r = pearson(a, b)
    if mad <= 5:
        gate, passes = "PASS_SINGLE", 1
    elif mad <= 8:
        gate, passes = "DOUBLE_PASS", 2
    else:
        gate, passes = "STOP_K1", None
    summary = {
        "label": "reliability (band-range)",
        "student_model": model,
        "n": len(pairs),
        "pearson_r": r,
        "mad": mad,
        "gate": gate,
        "JUDGE_PASSES": passes,
        "mean_pass1": sum(a) / len(a),
        "mean_pass2": sum(b) / len(b),
    }
    out = RUNS / "finance_band_recheck_summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def summarize_baselines(student: str, teacher: str) -> dict:
    s = mean_for_model("heldout", student, pass_i=1)
    t = mean_for_model("heldout", teacher, pass_i=1)
    if not s["n"] or not t["n"]:
        raise SystemExit(f"incomplete grades: student n={s['n']} teacher n={t['n']}")
    # Align on common ids for paired stats
    common = sorted(set(s["ids_scores"]) & set(t["ids_scores"]))
    sa = [s["ids_scores"][i] for i in common]
    ta = [t["ids_scores"][i] for i in common]
    s_ci = mean_bootstrap(sa, n_boot=10_000, seed=0)
    t_ci = mean_bootstrap(ta, n_boot=10_000, seed=1)
    delta = paired_bootstrap(sa, ta, n_boot=10_000, seed=2)

    # Per-category trap hits (student)
    traps_by_cat: dict[str, Counter] = defaultdict(Counter)
    for r in _load_jsonl(_grades_path("heldout", student)):
        if int(r.get("pass_i", 1)) != 1 or r.get("error") or "normalized" not in r:
            continue
        for trap in r.get("traps_hit") or []:
            traps_by_cat[r["category"]][trap] += 1

    # Per-category means both arms
    cats = sorted(set(s["by_category"]) | set(t["by_category"]))
    per_cat = []
    for c in cats:
        per_cat.append(
            {
                "category": c,
                "student_mean": s["by_category"].get(c),
                "student_n": s["by_category_n"].get(c, 0),
                "teacher_mean": t["by_category"].get(c),
                "teacher_n": t["by_category_n"].get(c, 0),
                "traps_hit_student": dict(traps_by_cat.get(c, {})),
            }
        )

    summary = {
        "n_common": len(common),
        "student_model": student,
        "teacher_model": teacher,
        "JUDGE_PASSES": judge_mod.JUDGE_PASSES,
        "student": s_ci,
        "teacher": t_ci,
        "delta_teacher_minus_student": delta,
        "per_category": per_cat,
        "traps_student_total": s["traps"],
        "traps_teacher_total": t["traps"],
    }
    out = RUNS / "finance_baselines_summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def _load_env() -> None:
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _require_keys() -> None:
    if not (
        os.environ.get("PRIME_API_KEY")
        or os.environ.get("OPENROUTER_API_KEY")
        or os.environ.get("JUDGE_API_KEY")
    ):
        raise SystemExit("Need PRIME_API_KEY or OPENROUTER_API_KEY in env/.env")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--mode",
        choices=("headroom", "band-recheck", "baselines", "summarize-headroom", "summarize-band", "summarize-baselines"),
        required=True,
    )
    ap.add_argument("--models", nargs="+", default=None, help="Override candidate models")
    ap.add_argument("--student-model", default=None)
    ap.add_argument("--teacher-model", default=TEACHER_DEFAULT)
    ap.add_argument(
        "--only-arm",
        choices=("student", "teacher"),
        default=None,
        help="Baselines: run only one arm this chunk",
    )
    ap.add_argument("--n", type=int, default=20, help="Headroom sample size")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-new", type=int, default=None, help="Max new gen/grade ops this run")
    ap.add_argument("--time-budget-s", type=float, default=None, help="Wall-clock pause budget")
    ap.add_argument("--judge-passes", type=int, default=None)
    ap.add_argument("--answers-only", action="store_true")
    ap.add_argument("--grades-only", action="store_true")
    ap.add_argument("--pass-label", type=int, default=1, help="Grade pass index (2 for recheck)")
    args = ap.parse_args()

    _load_env()
    os.environ.setdefault("AGENT_BASE_URL", "https://api.pinference.ai/api/v1")
    os.environ.setdefault("AGENT_TIMEOUT_S", "240")
    os.environ["AGENT_USE_EXAMPLES"] = "0"
    if args.judge_passes is not None:
        os.environ["JUDGE_PASSES"] = str(args.judge_passes)
        judge_mod.JUDGE_PASSES = args.judge_passes

    judge_passes = (
        args.judge_passes
        if args.judge_passes is not None
        else judge_mod.JUDGE_PASSES
    )

    if args.mode.startswith("summarize"):
        pass
    else:
        _require_keys()

    if args.mode == "headroom":
        models = args.models or list(DEFAULT_CANDIDATES)
        ids = stratified_ids("validation", args.n, args.seed)
        print(f"[headroom] n={len(ids)} models={models} timeout={os.environ['AGENT_TIMEOUT_S']}s", flush=True)
        for m in models:
            if not args.grades_only:
                generate_for_ids(
                    ids,
                    m,
                    "headroom",
                    resume=args.resume,
                    max_new=args.max_new,
                    time_budget_s=args.time_budget_s,
                )
            if args.answers_only:
                continue
            grade_for_ids(
                ids,
                m,
                "headroom",
                resume=args.resume,
                judge_passes=judge_passes,
                pass_label=1,
                max_new=args.max_new,
                time_budget_s=args.time_budget_s,
            )
        s = summarize_headroom(models)
        print(json.dumps(s, indent=2))
        # Auto-extend if all below band
        dec = s["decision"]
        if dec.get("fallback") and (args.models is None):
            fb = dec["fallback"]
            print(f"[headroom] all <{BAND_LO}; also probing {fb}", flush=True)
            if not args.grades_only:
                generate_for_ids(
                    ids, fb, "headroom", resume=args.resume, max_new=args.max_new, time_budget_s=args.time_budget_s
                )
            if not args.answers_only:
                grade_for_ids(
                    ids,
                    fb,
                    "headroom",
                    resume=args.resume,
                    judge_passes=judge_passes,
                    pass_label=1,
                    max_new=args.max_new,
                    time_budget_s=args.time_budget_s,
                )
            s = summarize_headroom(models + [fb])
            print(json.dumps(s, indent=2))

    elif args.mode == "summarize-headroom":
        models = args.models or list(DEFAULT_CANDIDATES)
        fb_path = _grades_path("headroom", FALLBACK_LARGER)
        if fb_path.exists() and FALLBACK_LARGER not in models:
            models = models + [FALLBACK_LARGER]
        print(json.dumps(summarize_headroom(models), indent=2))

    elif args.mode == "band-recheck":
        model = args.student_model
        if not model:
            hs = json.loads((RUNS / "finance_headroom_summary.json").read_text())
            model = hs["decision"].get("chosen")
            if not model:
                raise SystemExit("no chosen student in headroom summary; pass --student-model")
        ids = stratified_ids("validation", args.n, args.seed)
        # Ensure pass-1 grades exist; then grade pass-2
        if not args.grades_only:
            # answers should already exist from headroom
            ans = _answers_path("headroom", model)
            if not ans.exists():
                generate_for_ids(ids, model, "headroom", resume=True)
        # Fill any missing pass-1
        grade_for_ids(
            ids,
            model,
            "headroom",
            resume=True,
            judge_passes=1,  # each fresh context is one call
            pass_label=1,
            max_new=args.max_new,
            time_budget_s=args.time_budget_s,
        )
        grade_for_ids(
            ids,
            model,
            "headroom",
            resume=args.resume,
            judge_passes=1,
            pass_label=2,
            max_new=args.max_new,
            time_budget_s=args.time_budget_s,
        )
        s = summarize_band_recheck(model)
        print(json.dumps(s, indent=2))
        if s["gate"] == "STOP_K1":
            raise SystemExit(2)

    elif args.mode == "summarize-band":
        model = args.student_model
        if not model:
            hs = json.loads((RUNS / "finance_headroom_summary.json").read_text())
            model = hs["decision"]["chosen"]
        print(json.dumps(summarize_band_recheck(model), indent=2))

    elif args.mode == "baselines":
        student = args.student_model
        if not student:
            hs = json.loads((RUNS / "finance_headroom_summary.json").read_text())
            student = hs["decision"].get("chosen")
            if not student:
                raise SystemExit("need --student-model (no headroom choice)")
        teacher = args.teacher_model
        ids = stratified_ids("heldout", None, args.seed)
        print(
            f"[baselines] n={len(ids)} student={student} teacher={teacher} "
            f"JUDGE_PASSES={judge_passes}",
            flush=True,
        )
        arms = [(student, "student"), (teacher, "teacher")]
        if args.only_arm == "student":
            arms = [(student, "student")]
        elif args.only_arm == "teacher":
            arms = [(teacher, "teacher")]
        n_gradable = len(_gradable_ids(ids))
        for arm_model, label in arms:
            print(f"=== arm {label}: {arm_model} ===", flush=True)
            if not args.grades_only:
                generate_for_ids(
                    ids,
                    arm_model,
                    "heldout",
                    resume=args.resume,
                    max_new=args.max_new,
                    time_budget_s=args.time_budget_s,
                )
            if args.answers_only:
                continue
            grade_for_ids(
                ids,
                arm_model,
                "heldout",
                resume=args.resume,
                judge_passes=judge_passes,
                pass_label=1,
                max_new=args.max_new,
                time_budget_s=args.time_budget_s,
            )
        s_n = mean_for_model("heldout", student)["n"]
        t_n = mean_for_model("heldout", teacher)["n"]
        print(
            f"[baselines] graded student={s_n}/{n_gradable} teacher={t_n}/{n_gradable} "
            f"(heldout={len(ids)}, ungradable_rubrics={len(ids) - n_gradable})",
            flush=True,
        )
        if baselines_complete(ids, student, teacher):
            print(json.dumps(summarize_baselines(student, teacher), indent=2))
        else:
            print("[baselines] incomplete — re-run with --resume", flush=True)

    elif args.mode == "summarize-baselines":
        student = args.student_model
        teacher = args.teacher_model
        if not student:
            hs = json.loads((RUNS / "finance_headroom_summary.json").read_text())
            student = hs["decision"]["chosen"]
        print(json.dumps(summarize_baselines(student, teacher), indent=2))


if __name__ == "__main__":
    main()
