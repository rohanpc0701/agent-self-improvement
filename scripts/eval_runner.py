#!/usr/bin/env python3
"""Three-arm multi-seed eval runner (RSI-Mem G0.2).

Arms
----
- student-alone: student, empty memory (run once; reused across seeds)
- student+memory: student with seed-specific memory bundle
- teacher-alone: teacher model, empty memory (run once; reused across seeds)

Seed semantics
--------------
A seed controls LEARN-stream sampling + teacher generation that BUILDS the
memory bundle (independent memory builds per seed). The held-out problem set
and decoding temperature (0) stay fixed across seeds.

Dry-run prints the run matrix without API calls:
  python3 scripts/eval_runner.py --dry-run --seeds 3
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.bootstrap import paired_bootstrap  # noqa: E402
from contracts.schemas import FewShotExample  # noqa: E402


@dataclass(frozen=True)
class EvalCell:
    arm: str  # student_alone | student_memory | teacher_alone
    seed: int | None  # None for baseline arms shared across seeds
    question_id: str


def load_heldout_ids(manifest_path: Path) -> list[str]:
    m = json.loads(manifest_path.read_text())
    return list(m["heldout_ids"])


def build_run_matrix(
    heldout_ids: Sequence[str],
    seeds: Sequence[int],
) -> list[EvalCell]:
    """student/teacher alone once; student+memory once per seed."""
    cells: list[EvalCell] = []
    for qid in heldout_ids:
        cells.append(EvalCell("student_alone", None, qid))
        cells.append(EvalCell("teacher_alone", None, qid))
        for seed in seeds:
            cells.append(EvalCell("student_memory", seed, qid))
    return cells


def memory_bundle_for_seed(
    seed: int,
    build_fn: Callable[[int], list[FewShotExample]],
) -> list[FewShotExample]:
    """Build (or mock) a memory bundle for this seed."""
    return build_fn(seed)


def summarize_eval(
    per_arm_scores: dict[str, list[float]],
    *,
    n_boot: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    """per_arm_scores keys: student_alone, teacher_alone, student_memory_{seed}.

    GAP(seed) = mean(student_memory_seed) - mean(student_alone).
    teacher_gap_closed = GAP / (teacher_alone - student_alone) when denom > 0.
    """
    alone = per_arm_scores["student_alone"]
    teacher = per_arm_scores["teacher_alone"]
    mem_keys = sorted(k for k in per_arm_scores if k.startswith("student_memory_"))
    per_seed: list[dict[str, Any]] = []
    gaps: list[float] = []
    for mk in mem_keys:
        mem = per_arm_scores[mk]
        boot = paired_bootstrap(alone, mem, n_boot=n_boot, seed=seed)
        gap = boot["delta"]
        gaps.append(gap)
        denom = mean(teacher) - mean(alone)
        closed = (gap / denom) if denom > 1e-9 else None
        per_seed.append(
            {
                "arm": mk,
                "gap": gap,
                "ci_low": boot["ci_low"],
                "ci_high": boot["ci_high"],
                "p_value": boot["p_value"],
                "teacher_gap_closed": closed,
                "student_alone_mean": mean(alone),
                "student_memory_mean": mean(mem),
                "teacher_alone_mean": mean(teacher),
            }
        )
    # Pool gaps across seeds (mean of per-seed gaps; bootstrap on concatenated
    # is left to caller — here we report mean GAP + per-seed detail).
    return {
        "per_seed": per_seed,
        "mean_gap": mean(gaps) if gaps else 0.0,
        "n_seeds": len(gaps),
    }


def load_done_cells(path: Path) -> set[tuple[str, int | None, str]]:
    done: set[tuple[str, int | None, str]] = set()
    if not path.exists():
        return done
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        done.add((row["arm"], row.get("seed"), row["question_id"]))
    return done


def pending_cells(
    matrix: list[EvalCell], done: set[tuple[str, int | None, str]]
) -> list[EvalCell]:
    return [c for c in matrix if (c.arm, c.seed, c.question_id) not in done]


def default_mock_bundle(seed: int) -> list[FewShotExample]:
    """Deterministic fake bundle for dry-run / tests (no API)."""
    return [
        FewShotExample(
            question=f"seed-{seed}-q",
            correct_output=f"def f_{seed}(): pass",
            domain_id="arrays",
            source="teacher",
        )
    ]


def dry_run(args: argparse.Namespace) -> None:
    heldout = (
        load_heldout_ids(Path(args.manifest))
        if Path(args.manifest).exists()
        else [f"h_{i}" for i in range(args.fake_heldout)]
    )
    seeds = parse_seeds(args.seeds)
    matrix = build_run_matrix(heldout, seeds)
    print("=== eval_runner dry-run ===")
    print(f"heldout={len(heldout)} seeds={seeds}")
    print(
        f"cells={len(matrix)} "
        f"(student_alone={len(heldout)} + teacher_alone={len(heldout)} "
        f"+ student_memory={len(heldout)*len(seeds)})"
    )
    for seed in seeds:
        bundle = memory_bundle_for_seed(seed, default_mock_bundle)
        print(f"  seed={seed} memory_bundle_size={len(bundle)} id_hint={bundle[0].question}")
    out = Path(args.out) if args.out else Path("runs") / f"eval_dryrun_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    print(f"would write under {out}/")


def parse_seeds(n: int) -> list[int]:
    return list(range(n))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--manifest", default="fixtures/heldout_manifest.json")
    ap.add_argument("--fake-heldout", type=int, default=5, help="dry-run without manifest")
    ap.add_argument("--out", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--student-model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--teacher-model", default="minimax/minimax-m2.5")
    args = ap.parse_args()
    if args.dry_run:
        dry_run(args)
        return
    raise SystemExit(
        "Live eval_runner not enabled in this work order — use --dry-run. "
        "Full [LIVE] Phase-2 eval waits for the gated pipeline."
    )


if __name__ == "__main__":
    main()
