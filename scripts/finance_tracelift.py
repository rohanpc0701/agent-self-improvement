#!/usr/bin/env python3
"""Finance TraceLift build loop — train failures → repair → distill → uplift gate.

Pipeline (docs/superpowers/plans/2026-07-20-finance-tracelift.md Task C):
  1. Student on TRAIN-STREAM → collect low-score failures
  2. teacher_repair → distill_memory_item (playbook/trap/skeleton)
  3. Uplift-gate each candidate on VALIDATION (u in normalized pts; keep u > +1, K=2)
  4. Stopping rule: window(15) mean uplift < +0.5 OR admission rate < 20%
  5. Freeze admitted memory to runs/finance_tracelift_memory.json

Chunked + --resume. Hermetic tests mock the adapter / teacher.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adapters.finance import (  # noqa: E402
    FinanceAdapter,
    distill_memory_item,
    generate_answer,
    get_problem,
    load_manifest,
    teacher_repair,
)
from contracts.schemas import AgentConfig, FewShotExample  # noqa: E402
from correction.judge import JudgeParseError, grade, rubric_max_points  # noqa: E402
from harness.feed import FeedItem  # noqa: E402

RUNS = ROOT / "runs"
DEFAULT_STUDENT = "qwen/qwen3.6-27b"
DEFAULT_MEMORY = RUNS / "finance_tracelift_memory.json"
DEFAULT_STATE = RUNS / "finance_tracelift_state.jsonl"
FAIL_THRESHOLD = 40.0  # normalized pts — below this counts as a failure
MIN_U_NORM = 1.0  # keep candidates with u > +1 normalized pt
STOP_WINDOW = 15
STOP_MEAN_U = 0.5
STOP_ADMIT_RATE = 0.20
UPLIFT_K = 2


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def train_ids(seed: int = 42) -> list[str]:
    m = load_manifest()
    ids = list(m["train_ids"])
    rng = random.Random(seed)
    rng.shuffle(ids)
    return ids


def validation_items(n: int | None = None, seed: int = 42) -> list[FeedItem]:
    m = load_manifest()
    ids = list(m["validation_ids"])
    rng = random.Random(seed)
    rng.shuffle(ids)
    if n is not None:
        ids = ids[:n]
    items = []
    for qid in ids:
        p = get_problem(qid)
        items.append(
            FeedItem(
                question_id=qid,
                question=p["question"],
                gold_output="",
                domain_id=p["category"],
                difficulty="hard",
                phase="degraded",
            )
        )
    return items


def is_failure(normalized: float, threshold: float = FAIL_THRESHOLD) -> bool:
    return normalized < threshold


def u_normalized(u_accuracy: float) -> float:
    """Convert execution_accuracy delta (0–1) → normalized score points (0–100)."""
    return float(u_accuracy) * 100.0


def should_stop(
    recent_u: list[float],
    n_considered: int,
    n_admitted: int,
    *,
    window: int = STOP_WINDOW,
    mean_u_floor: float = STOP_MEAN_U,
    admit_rate_floor: float = STOP_ADMIT_RATE,
) -> tuple[bool, str]:
    """Stopping rule (plan §3): window mean uplift or admission rate."""
    if n_considered >= window:
        w = recent_u[-window:]
        mean_u = sum(w) / len(w)
        if mean_u < mean_u_floor:
            return True, f"window({window}) mean uplift {mean_u:.3f} < {mean_u_floor}"
    if n_considered >= window:
        rate = n_admitted / max(1, n_considered)
        if rate < admit_rate_floor:
            return True, f"admission rate {rate:.2%} < {admit_rate_floor:.0%}"
    return False, ""


def compact_memory(
    items: list[FewShotExample],
    *,
    max_per_category: int = 6,
) -> list[FewShotExample]:
    """Hard-cap memory store size per category (merge-over-append)."""
    by_cat: dict[str, list[FewShotExample]] = {}
    for ex in items:
        by_cat.setdefault(ex.domain_id or "", []).append(ex)
    out: list[FewShotExample] = []
    for cat, group in sorted(by_cat.items()):
        out.extend(group[:max_per_category])
    return out


def freeze_memory(
    items: list[FewShotExample],
    path: Path,
    *,
    meta: dict | None = None,
) -> dict:
    payload = {
        "version": 1,
        "n": len(items),
        "items": [ex.model_dump() for ex in items],
        "meta": meta or {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return payload


def load_memory(path: Path) -> list[FewShotExample]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [FewShotExample(**x) for x in data.get("items", [])]


def done_keys(state_path: Path, kind: str) -> set[str]:
    keys: set[str] = set()
    for row in _load_jsonl(state_path):
        if row.get("kind") == kind and row.get("ok"):
            keys.add(str(row.get("key")))
    return keys


def collect_train_failures(
    student_model: str,
    ids: list[str],
    *,
    state_path: Path,
    threshold: float = FAIL_THRESHOLD,
    max_new: int | None = None,
    time_budget_s: float | None = None,
    resume: bool = True,
    target_failures: int | None = None,
) -> list[dict]:
    """Run/grade student on train ids; return failure rows (and log state)."""
    done = done_keys(state_path, "train_grade") if resume else set()
    failures: list[dict] = []
    # Recover prior failures from state.
    for row in _load_jsonl(state_path):
        if row.get("kind") == "train_grade" and row.get("ok") and row.get("failure"):
            failures.append(row)
    # Dedup recovered failures early so target_failures sees current count.
    latest_f: dict[str, dict] = {r["key"]: r for r in failures}
    failures = list(latest_f.values())

    cfg = AgentConfig(config_id="tracelift-student", model=student_model, few_shot_examples=[])
    os.environ.setdefault("AGENT_USE_EXAMPLES", "0")
    t0 = time.time()
    n_new = 0
    pending = [qid for qid in ids if qid not in done]
    print(
        f"[tracelift/train] pending={len(pending)} done={len(done)} "
        f"failures_so_far={len(failures)}"
        + (f" target={target_failures}" if target_failures else ""),
        flush=True,
    )
    if target_failures is not None and len(failures) >= target_failures:
        print(
            f"[tracelift/train] already have {len(failures)}≥{target_failures} failures; skip",
            flush=True,
        )
        return failures

    for i, qid in enumerate(pending, 1):
        if max_new is not None and n_new >= max_new:
            break
        if time_budget_s is not None and (time.time() - t0) >= time_budget_s:
            print(f"[tracelift/train] time budget {time_budget_s}s; pause", flush=True)
            break
        if target_failures is not None and len(failures) >= target_failures:
            print(
                f"[tracelift/train] hit target_failures={target_failures}; pause",
                flush=True,
            )
            break
        p = get_problem(qid)
        try:
            rubric_max_points(p["rubric"])
        except JudgeParseError as exc:
            _append_jsonl(
                state_path,
                {
                    "kind": "train_grade",
                    "key": qid,
                    "ok": True,
                    "failure": False,
                    "error": f"ungradable_rubric: {exc}",
                    "ts": time.time(),
                },
            )
            n_new += 1
            continue
        try:
            # STUDENT_MAX_TOKENS (default 8192) via generate_answer — avoid
            # empty-content retry loops that blow the LIVE chunk budget.
            answer, _stats = generate_answer(
                p["question"], cfg, category=p["category"]
            )
            result = grade(question=p["question"], rubric=p["rubric"], answer=answer)
            norm = float(result["normalized"])
        except Exception as exc:
            print(f"  [train {i}/{len(pending)}] {qid} FAIL gen/grade ({exc})", flush=True)
            continue
        failed = is_failure(norm, threshold)
        row = {
            "kind": "train_grade",
            "key": qid,
            "ok": True,
            "failure": failed,
            "normalized": norm,
            "answer": answer,
            "category": p["category"],
            "traps_hit": result.get("traps_hit") or [],
            "ts": time.time(),
        }
        _append_jsonl(state_path, row)
        if failed:
            failures.append(row)
            # keep unique by key
            failures = list({r["key"]: r for r in failures}.values())
        n_new += 1
        print(
            f"  [train {i}/{len(pending)}] {qid} norm={norm:.1f} "
            f"{'FAIL' if failed else 'ok'} (fails={len(failures)})",
            flush=True,
        )
    # Dedup failures by qid (latest wins).
    latest: dict[str, dict] = {}
    for row in failures:
        latest[row["key"]] = row
    return [r for r in latest.values() if r.get("failure")]


def build_candidates_from_failures(
    failures: list[dict],
    *,
    state_path: Path,
    kinds: tuple[str, ...] = ("playbook", "trap", "skeleton"),
    max_new: int | None = None,
    time_budget_s: float | None = None,
    resume: bool = True,
) -> list[FewShotExample]:
    """teacher_repair + distill for each failure × kinds."""
    done = done_keys(state_path, "candidate") if resume else set()
    # Reload previously admitted candidates from state.
    candidates: list[FewShotExample] = []
    for row in _load_jsonl(state_path):
        if row.get("kind") == "candidate" and row.get("ok") and row.get("example"):
            candidates.append(FewShotExample(**row["example"]))

    t0 = time.time()
    n_new = 0
    for fail in failures:
        qid = fail["key"]
        student_answer = fail.get("answer") or ""
        for kind in kinds:
            key = f"{qid}:{kind}"
            if key in done:
                continue
            if max_new is not None and n_new >= max_new:
                return candidates
            if time_budget_s is not None and (time.time() - t0) >= time_budget_s:
                print(f"[tracelift/cand] time budget; pause", flush=True)
                return candidates
            try:
                repaired = teacher_repair(qid, student_answer)
                ex = distill_memory_item(qid, repaired, kind=kind)
            except Exception as exc:
                print(f"  [cand] {key} FAILED ({exc})", flush=True)
                continue
            _append_jsonl(
                state_path,
                {
                    "kind": "candidate",
                    "key": key,
                    "ok": True,
                    "qid": qid,
                    "mem_kind": kind,
                    "example": ex.model_dump(),
                    "ts": time.time(),
                },
            )
            candidates.append(ex)
            n_new += 1
            print(f"  [cand] {key} distilled source={ex.source}", flush=True)
    # Dedup by (domain_id, question, correct_output)
    seen: set[tuple] = set()
    uniq: list[FewShotExample] = []
    for ex in candidates:
        sig = (ex.domain_id, ex.question, ex.correct_output)
        if sig in seen:
            continue
        seen.add(sig)
        uniq.append(ex)
    return uniq


def _candidate_key(cand: FewShotExample) -> str:
    return f"{cand.domain_id}|{cand.question}|{(cand.correct_output or '')[:40]}"


def _is_usable_candidate(cand: FewShotExample) -> bool:
    """Skip empty / near-empty distillations (teacher scrub wipeouts)."""
    body = (cand.correct_output or "").strip()
    meaningful = re.sub(r"^Category playbook \([^)]+\):\s*", "", body, flags=re.I)
    meaningful = re.sub(r"^[-•*\s]+", "", meaningful).strip()
    return len(meaningful) >= 24


def _mean_scores(
    adapter: FinanceAdapter,
    cfg: AgentConfig,
    items: list[FeedItem],
    k: int,
) -> list[float]:
    scores: list[float] = []
    for item in items:
        for _ in range(k):
            rec = adapter.run_item(item, cfg, use_rules=False)
            if rec is not None:
                scores.append(float(rec.execution_accuracy))
    return scores


def _val_slice_for_candidate(
    cand: FewShotExample, val_items: list[FeedItem], *, min_n: int = 3
) -> list[FeedItem]:
    """Prefer same-category validation (memory is category-keyed); else full slice."""
    same = [it for it in val_items if (it.domain_id or "") == (cand.domain_id or "")]
    if len(same) >= min_n:
        return same
    return list(val_items)


def gate_candidates(
    candidates: list[FewShotExample],
    student_model: str,
    val_items: list[FeedItem],
    *,
    state_path: Path,
    k: int = UPLIFT_K,
    min_u_norm: float = MIN_U_NORM,
    max_new: int | None = None,
    time_budget_s: float | None = None,
    resume: bool = True,
    adapter: FinanceAdapter | None = None,
) -> tuple[list[FewShotExample], list[dict]]:
    """Uplift-gate candidates; return (admitted, scored rows with u_norm).

    Bare (no-memory) student scores are cached per validation-id set so we do
    not re-run the empty-prompt arm once per candidate.
    """
    ad = adapter or FinanceAdapter()
    cfg = AgentConfig(config_id="tracelift-uplift", model=student_model, few_shot_examples=[])
    done = done_keys(state_path, "uplift") if resume else set()
    admitted: list[FewShotExample] = []
    scored: list[dict] = []

    # Restore prior uplift decisions.
    for row in _load_jsonl(state_path):
        if row.get("kind") == "uplift" and row.get("ok"):
            scored.append(row)
            if row.get("admitted") and row.get("example"):
                admitted.append(FewShotExample(**row["example"]))

    t0 = time.time()
    n_new = 0
    recent_u: deque[float] = deque(maxlen=STOP_WINDOW * 2)
    for row in scored:
        recent_u.append(float(row.get("u_norm", 0.0)))

    bare_cache: dict[tuple[str, ...], list[float]] = {}
    usable = [c for c in candidates if _is_usable_candidate(c)]
    skipped = len(candidates) - len(usable)
    if skipped:
        print(f"[tracelift/uplift] skipping {skipped} empty/weak candidates", flush=True)

    for i, cand in enumerate(usable):
        key = _candidate_key(cand)
        if key in done:
            continue
        if max_new is not None and n_new >= max_new:
            break
        if time_budget_s is not None and (time.time() - t0) >= time_budget_s:
            print("[tracelift/uplift] time budget; pause", flush=True)
            break

        # Stopping rule checked before expensive uplift.
        stop, reason = should_stop(
            list(recent_u),
            n_considered=len(scored),
            n_admitted=len(admitted),
        )
        if stop:
            print(f"[tracelift/uplift] STOP: {reason}", flush=True)
            _append_jsonl(
                state_path,
                {"kind": "stop", "ok": True, "reason": reason, "ts": time.time()},
            )
            break

        slice_items = _val_slice_for_candidate(cand, val_items)
        cache_key = tuple(sorted(it.question_id for it in slice_items))
        try:
            if cache_key not in bare_cache:
                print(
                    f"  [uplift] computing bare baseline on n={len(slice_items)} k={k} …",
                    flush=True,
                )
                bare_cache[cache_key] = _mean_scores(ad, cfg, slice_items, k)
            bare_scores = bare_cache[cache_key]
            with_cfg = cfg.model_copy(
                update={"few_shot_examples": [cand], "config_id": "uplift-with"}
            )
            with_scores = _mean_scores(ad, with_cfg, slice_items, k)
            if not bare_scores or not with_scores:
                u_acc = 0.0
            else:
                u_acc = (sum(with_scores) / len(with_scores)) - (
                    sum(bare_scores) / len(bare_scores)
                )
            u_norm = u_normalized(u_acc)
        except Exception as exc:
            print(f"  [uplift] {key[:60]} FAILED ({exc})", flush=True)
            _append_jsonl(
                state_path,
                {
                    "kind": "uplift",
                    "key": key,
                    "ok": False,
                    "error": str(exc),
                    "ts": time.time(),
                },
            )
            n_new += 1
            continue
        keep = u_norm > min_u_norm
        row = {
            "kind": "uplift",
            "key": key,
            "ok": True,
            "u_acc": u_acc,
            "u_norm": u_norm,
            "admitted": keep,
            "example": cand.model_dump() if keep else None,
            "mem_kind": cand.question.split()[0] if cand.question else "",
            "domain_id": cand.domain_id,
            "val_n": len(slice_items),
            "k": k,
            "ts": time.time(),
        }
        _append_jsonl(state_path, row)
        scored.append(row)
        recent_u.append(u_norm)
        if keep:
            admitted.append(cand)
        n_new += 1
        print(
            f"  [uplift {i+1}/{len(usable)}] u_norm={u_norm:+.2f} "
            f"{'ADMIT' if keep else 'reject'} {cand.domain_id} "
            f"(val_n={len(slice_items)})",
            flush=True,
        )
    return admitted, scored


def summarize_gate(scored: list[dict], stop_reason: str = "") -> dict:
    us = [float(r["u_norm"]) for r in scored if "u_norm" in r]
    admitted = [r for r in scored if r.get("admitted")]
    us_sorted = sorted(us)
    def _pct(p: float) -> float | None:
        if not us_sorted:
            return None
        idx = min(len(us_sorted) - 1, int(p * (len(us_sorted) - 1)))
        return us_sorted[idx]
    return {
        "n_candidates": len(scored),
        "n_admitted": len(admitted),
        "admission_rate": (len(admitted) / len(scored)) if scored else 0.0,
        "u_norm_mean": (sum(us) / len(us)) if us else None,
        "u_norm_p50": _pct(0.5),
        "u_norm_p10": _pct(0.1),
        "u_norm_p90": _pct(0.9),
        "stop_reason": stop_reason,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--student-model", default=os.environ.get("STUDENT_MODEL", DEFAULT_STUDENT))
    ap.add_argument("--state", type=Path, default=DEFAULT_STATE)
    ap.add_argument("--memory-out", type=Path, default=DEFAULT_MEMORY)
    ap.add_argument(
        "--val-n",
        type=int,
        default=int(os.environ.get("FINANCE_UPLIFT_VAL_N", "12")),
        help="Validation slice size (default 12; plan text says 80 — cost tradeoff)",
    )
    ap.add_argument("--k", type=int, default=UPLIFT_K)
    ap.add_argument("--fail-threshold", type=float, default=FAIL_THRESHOLD)
    ap.add_argument("--min-u", type=float, default=MIN_U_NORM, help="Min u in normalized pts")
    ap.add_argument("--max-new", type=int, default=None)
    ap.add_argument("--time-budget-s", type=float, default=None)
    ap.add_argument(
        "--target-failures",
        type=int,
        default=None,
        help="Stop train phase once this many unique failures are collected",
    )
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--phase",
        choices=("train", "candidates", "gate", "all", "freeze"),
        default="all",
    )
    ap.add_argument("--dry-run", action="store_true", help="Skip LLM; synthetic path for wiring")
    args = ap.parse_args(argv)

    RUNS.mkdir(parents=True, exist_ok=True)
    summary: dict = {"student": args.student_model}

    if args.phase in ("train", "all") and not args.dry_run:
        fails = collect_train_failures(
            args.student_model,
            train_ids(args.seed),
            state_path=args.state,
            threshold=args.fail_threshold,
            max_new=args.max_new,
            time_budget_s=args.time_budget_s,
            resume=args.resume,
            target_failures=args.target_failures,
        )
        summary["n_failures"] = len(fails)
    else:
        fails = [
            r
            for r in _load_jsonl(args.state)
            if r.get("kind") == "train_grade" and r.get("failure") and r.get("ok")
        ]
        # latest per key
        latest = {r["key"]: r for r in fails}
        fails = list(latest.values())
        summary["n_failures"] = len(fails)

    if args.phase in ("candidates", "all") and not args.dry_run:
        cands = build_candidates_from_failures(
            fails,
            state_path=args.state,
            max_new=args.max_new,
            time_budget_s=args.time_budget_s,
            resume=args.resume,
        )
        summary["n_candidates_built"] = len(cands)
    else:
        cands = []
        for row in _load_jsonl(args.state):
            if row.get("kind") == "candidate" and row.get("ok") and row.get("example"):
                cands.append(FewShotExample(**row["example"]))
        # dedup
        seen = set()
        uniq = []
        for ex in cands:
            sig = (ex.domain_id, ex.question, ex.correct_output)
            if sig not in seen:
                seen.add(sig)
                uniq.append(ex)
        cands = uniq
        summary["n_candidates_built"] = len(cands)

    stop_reason = ""
    if args.phase in ("gate", "all") and not args.dry_run:
        val = validation_items(n=args.val_n, seed=args.seed)
        admitted, scored = gate_candidates(
            cands,
            args.student_model,
            val,
            state_path=args.state,
            k=args.k,
            min_u_norm=args.min_u,
            max_new=args.max_new,
            time_budget_s=args.time_budget_s,
            resume=args.resume,
        )
        for row in _load_jsonl(args.state):
            if row.get("kind") == "stop":
                stop_reason = row.get("reason") or stop_reason
        gate_sum = summarize_gate(scored, stop_reason)
        summary["gate"] = gate_sum
        admitted = compact_memory(admitted)
        freeze_memory(
            admitted,
            args.memory_out,
            meta={"summary": summary, "stop_reason": stop_reason},
        )
        summary["n_frozen"] = len(admitted)
        summary["memory_path"] = str(args.memory_out)
    elif args.phase == "freeze":
        admitted = []
        for row in _load_jsonl(args.state):
            if row.get("kind") == "uplift" and row.get("admitted") and row.get("example"):
                admitted.append(FewShotExample(**row["example"]))
        admitted = compact_memory(admitted)
        freeze_memory(admitted, args.memory_out, meta={"summary": summary})
        summary["n_frozen"] = len(admitted)

    if args.dry_run:
        # Synthetic admitted item for wiring tests.
        ex = FewShotExample(
            question="[FINANCE_PLAYBOOK] Accounting",
            correct_output="Always gate equity sufficiency before voting model.",
            domain_id="Accounting",
            source="tracelift",
        )
        freeze_memory([ex], args.memory_out, meta={"dry_run": True})
        summary["dry_run"] = True
        summary["n_frozen"] = 1

    out_path = RUNS / "finance_tracelift_summary.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
