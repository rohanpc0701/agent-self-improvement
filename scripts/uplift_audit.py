#!/usr/bin/env python3
"""Per-item uplift audit (RSI-Mem G1.1).

For each memory item i in a frozen bundle:
  u_i = mean_pass(validation, memory={i}) − mean_pass(validation, memory=∅)

Also measures u_full (whole bundle) and u_loo for the most negative items.
Validation ids come from the held-out manifest — NEVER the held-out set.

Live:
  python3 scripts/uplift_audit.py --model Qwen/Qwen3.5-4B \\
    --events runs/qwen35-4b_artifacts/events.jsonl \\
    --manifest fixtures/heldout_manifest.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from contracts.schemas import FewShotExample  # noqa: E402


@dataclass(frozen=True)
class RunSpec:
    """One (arm, question) cell to evaluate."""

    arm_key: str  # "empty" | "item_{i}" | "full" | "loo_{i}"
    question_id: str
    item_index: int | None  # None for empty/full; index for item_/loo_
    examples: tuple[FewShotExample, ...]


def audit_plan(
    bundle: Sequence[FewShotExample],
    validation_ids: Sequence[str],
    *,
    include_full: bool = False,
    loo_indices: Sequence[int] | None = None,
) -> list[RunSpec]:
    """Build cells: empty + each singleton (± full / LOO)."""
    specs: list[RunSpec] = []
    empty: tuple[FewShotExample, ...] = ()
    arms: list[tuple[str, int | None, tuple[FewShotExample, ...]]] = [
        ("empty", None, empty),
    ]
    for i, ex in enumerate(bundle):
        arms.append((f"item_{i}", i, (ex,)))
    if include_full:
        arms.append(("full", None, tuple(bundle)))
    for i in loo_indices or ():
        loo = tuple(ex for j, ex in enumerate(bundle) if j != i)
        arms.append((f"loo_{i}", i, loo))
    for arm_key, idx, examples in arms:
        for qid in validation_ids:
            specs.append(
                RunSpec(
                    arm_key=arm_key,
                    question_id=qid,
                    item_index=idx,
                    examples=examples,
                )
            )
    return specs


def _arm_means(results: list[dict[str, Any]]) -> dict[str, float]:
    by_arm: dict[str, list[float]] = {}
    for r in results:
        by_arm.setdefault(r["arm_key"], []).append(float(r["pass"]))
    return {k: mean(v) for k, v in by_arm.items() if v}


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute per-item u, u_full, counts, P1 verdict from cell results."""
    means = _arm_means(results)
    if "empty" not in means:
        raise ValueError("summarize requires an 'empty' arm")
    base = means["empty"]
    per_item: dict[str, float] = {}
    for key, m in means.items():
        if key.startswith("item_"):
            per_item[key] = m - base
    n_pos = sum(1 for u in per_item.values() if u > 0)
    n_neg = sum(1 for u in per_item.values() if u <= 0)
    frac_nonpos = (n_neg / len(per_item)) if per_item else 0.0
    verdict = "P1 CONFIRMED" if frac_nonpos >= 0.30 else "P1 NOT CONFIRMED"
    out: dict[str, Any] = {
        "per_item": dict(sorted(per_item.items(), key=lambda kv: kv[1])),
        "u_full": (means["full"] - base) if "full" in means else None,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "frac_u_le_0": frac_nonpos,
        "p1_verdict": verdict,
        "empty_mean": base,
    }
    loo = {k: means[k] - base for k in means if k.startswith("loo_")}
    if loo:
        out["u_loo"] = dict(sorted(loo.items(), key=lambda kv: kv[1]))
    return out


def parse_items_range(spec: str | None, n_bundle: int) -> range | None:
    """Parse ``--items 0-5`` into a half-open range over bundle indices, or None=all."""
    if spec is None:
        return None
    m = re.fullmatch(r"(\d+)-(\d+)", spec.strip())
    if not m:
        raise ValueError(f"bad --items {spec!r}; expected START-END")
    start, end = int(m.group(1)), int(m.group(2))
    return range(start, min(end + 1, n_bundle))


def filter_plan_for_items(
    plan: list[RunSpec], item_range: range | None
) -> list[RunSpec]:
    if item_range is None:
        return plan
    keep = {"empty"} | {f"item_{i}" for i in item_range}
    return [s for s in plan if s.arm_key in keep]


def load_done_cells(path: Path) -> set[tuple[str, str]]:
    done: set[tuple[str, str]] = set()
    if not path.exists():
        return done
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        done.add((row["arm_key"], row["question_id"]))
    return done


def pending_specs(
    plan: list[RunSpec], done: set[tuple[str, str]]
) -> list[RunSpec]:
    return [s for s in plan if (s.arm_key, s.question_id) not in done]


def _load_bundle(events_path: Path) -> list[FewShotExample]:
    from contracts.eventlog import read_events

    corrections = read_events(only="correction", path=str(events_path))
    if not corrections:
        raise SystemExit(f"no CorrectionAction in {events_path}")
    return list(corrections[-1].new_few_shot_examples)


def _validation_problems(manifest_path: Path) -> list[Any]:
    """Build pool-like objects with question_id / question / domain_id for run_arm."""
    from adapters.coding import _index

    m = json.loads(manifest_path.read_text())
    val_ids = m["validation_ids"]
    idx = _index()
    pool = []
    missing = []
    for qid in val_ids:
        p = idx.get(qid)
        if p is None:
            missing.append(qid)
            continue

        class _Item:
            pass

        it = _Item()
        it.question_id = qid
        it.question = p["question"]
        it.domain_id = p.get("topic") or p.get("db_id") or "arrays"
        pool.append(it)
    if missing:
        raise SystemExit(f"validation ids missing from coding index: {missing[:5]}")
    return pool, val_ids


def run_live(args: argparse.Namespace) -> None:
    from scripts.variance_check import run_arm

    bundle = _load_bundle(Path(args.events))
    pool, val_ids = _validation_problems(Path(args.manifest))
    item_range = parse_items_range(args.items, len(bundle))

    plan = audit_plan(bundle, val_ids, include_full=args.full)
    plan = filter_plan_for_items(plan, item_range)

    safe_model = args.model.replace("/", "_")
    out_path = Path(args.out or f"runs/uplift_audit_{safe_model}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done_cells(out_path) if args.resume else set()
    todo = pending_specs(plan, done)
    print(
        f"[uplift] bundle={len(bundle)} validation={len(val_ids)} "
        f"plan={len(plan)} done={len(done)} todo={len(todo)} → {out_path}",
        flush=True,
    )

    # Group by arm so we call run_arm once per arm (reuses variance_check).
    by_arm: dict[str, list[RunSpec]] = {}
    for s in todo:
        by_arm.setdefault(s.arm_key, []).append(s)

    with out_path.open("a", encoding="utf-8") as out:
        for arm_key, specs in by_arm.items():
            examples = list(specs[0].examples)
            arm_pool = [it for it in pool if any(s.question_id == it.question_id for s in specs)]
            # Preserve validation order for missing-only subset
            wanted = {s.question_id for s in specs}
            arm_pool = [it for it in pool if it.question_id in wanted]
            per_q = run_arm(args.model, arm_pool, examples, arm_key)
            for s in specs:
                acc = float(per_q.get(s.question_id, 0.0))
                row = {
                    "arm_key": s.arm_key,
                    "question_id": s.question_id,
                    "item_index": s.item_index,
                    "pass": acc,
                }
                out.write(json.dumps(row) + "\n")
                out.flush()
            print(f"  [{arm_key}] n={len(specs)} mean={mean(per_q.values()):.3f}", flush=True)

    # Re-read all results for summarize (may be partial if chunked).
    all_rows = [json.loads(l) for l in out_path.read_text().splitlines() if l.strip()]
    # Only summarize if empty + all item_* arms for this chunk are present
    try:
        summary = summarize(all_rows)
    except ValueError as e:
        print(f"[uplift] partial run — {e}")
        return
    print("\n=== UPLIFT SUMMARY ===")
    for k, u in summary["per_item"].items():
        print(f"  {k}: u={u:+.3f}")
    if summary.get("u_full") is not None:
        print(f"  u_full={summary['u_full']:+.3f}")
    print(
        f"  n_pos={summary['n_pos']} n_neg={summary['n_neg']} "
        f"frac_u<=0={summary['frac_u_le_0']:.2f} → {summary['p1_verdict']}"
    )
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {summary_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--events", required=True, help="events.jsonl with CorrectionAction")
    ap.add_argument("--manifest", default="fixtures/heldout_manifest.json")
    ap.add_argument("--items", default=None, help="bundle index range START-END")
    ap.add_argument("--full", action="store_true", help="also eval full bundle arm")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="print plan sizes only (no API)",
    )
    args = ap.parse_args()
    if args.dry_run:
        bundle = _load_bundle(Path(args.events))
        m = json.loads(Path(args.manifest).read_text())
        val_ids = m["validation_ids"]
        plan = audit_plan(bundle, val_ids)
        item_range = parse_items_range(args.items, len(bundle))
        plan = filter_plan_for_items(plan, item_range)
        print(
            f"dry-run: bundle={len(bundle)} validation={len(val_ids)} "
            f"cells={len(plan)} (= ({len(bundle)}+1)*{len(val_ids)} if full)"
        )
        return
    run_live(args)


if __name__ == "__main__":
    main()
