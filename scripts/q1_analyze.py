#!/usr/bin/env python3
"""Q1 analysis with the D3 corrections applied. Read-only, no API calls.

Kept separate from scripts/q1_self_plan.py so the harness is never edited while a run
is in flight (mixed instrumentation is the confound D1 exists to prevent).

The four D3 corrections (docs/prereg/PREREG_Q1_SELF_PLAN.md, committed blind to results):

1. One-sided gate with a magnitude floor on the recovery denominator. The shipped gate
   `not (ci_low <= 0 <= ci_high)` admits a teacher effect whose CI lies entirely BELOW
   zero, then divides by a negative denominator and prints a confident branch.
2. A paired CI on the recovery ratio, formed INSIDE one resampling loop. Zipping two
   separate paired_bootstrap calls pairs order statistics (their `boots` are sorted in
   place), which measured ~2.4x too narrow.
3. Per-unit normalization basis re-derived from each unit's persisted judge output,
   because the judge's basis is unstable within a question. Reported both ways.
4. The artifact says OpenRouter, provider-pinned — the stack D1 actually used.
"""
from __future__ import annotations

import glob
import json
import random
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from correction.judge import (  # noqa: E402
    _BONUS_LINE,
    _R_LINE,
    _TOTAL_LINE,
    _TRAP_LINE,
    rubric_declared_max,
    rubric_max_points,
)

ARMS = ("A", "B", "C")
K_REPS = 3
MIN_DENOM = 1.0  # normalized points; below this the ratio is not interpretable
N_BOOT = 10_000
SEED = 42
LABEL = "OpenRouter (student provider-pinned), matched-baseline, preregistered"


def load_units() -> dict:
    units = {}
    for f in sorted(glob.glob(str(ROOT / "runs" / "q1_state_shard*.json"))):
        units.update(json.load(open(f))["units"])
    legacy = ROOT / "runs" / "q1_state.json"
    if legacy.exists():
        for k, v in json.load(open(legacy))["units"].items():
            units.setdefault(k, v)
    return units


def item_arithmetic(raw: str) -> float | None:
    """ΣR − ΣtrapPenalty + ΣBonus from the judge's own lines, or None if no items."""
    text = (raw or "").strip()
    items = [float(m.group(2)) for m in _R_LINE.finditer(text)]
    if not items:
        return None
    traps = sum(float(m.group(2)) for m in _TRAP_LINE.finditer(text))
    bonuses = sum(float(m.group(2)) for m in _BONUS_LINE.finditer(text))
    return sum(items) - traps + bonuses


def unit_basis(raw: str, ladder: float, declared: float | None) -> tuple[float, str]:
    """Which basis did the judge actually report TOTAL on, for THIS unit?

    If TOTAL reconciles with the judge's own item arithmetic, it is on the item ladder.
    Otherwise the judge rescaled, so the rubric's declared target is the basis.
    """
    text = (raw or "").strip()
    # Mirror the judge's anchoring: LAST TOTAL, not first (see correction/judge.py).
    ms = list(_TOTAL_LINE.finditer(text))
    if not ms:
        return (declared or ladder), "no_total_in_raw"
    total = float(ms[-1].group(1))
    arith = item_arithmetic(text)
    if arith is not None and abs(total - arith) <= 0.51:
        return ladder, "ladder_reconciled"
    if declared:
        return declared, "rescaled_to_declared"
    return ladder, "ladder_fallback"


def per_question_means(units: dict, basis_mode: str) -> tuple[dict, dict]:
    """basis_mode: 'as_stored' | 'per_unit'. Returns (means_by_qid, diagnostics)."""
    from adapters.finance import get_problem

    qids = sorted({k.split("|")[0] for k in units})
    means, diag = {}, {"basis_reasons": {}, "renormalized": 0, "skipped_incomplete": []}
    for qid in qids:
        p = get_problem(qid)
        ladder = rubric_max_points(p["rubric"])
        declared = rubric_declared_max(p["rubric"])
        arm_means = {}
        ok = True
        for arm in ARMS:
            vals = []
            for rep in range(K_REPS):
                u = units.get(f"{qid}|{arm}|{rep}", {})
                g = u.get("grade")
                if not g:
                    ok = False
                    break
                if basis_mode == "as_stored":
                    vals.append(float(g["normalized"]))
                else:
                    basis, reason = unit_basis(g.get("raw", ""), ladder, declared)
                    diag["basis_reasons"][reason] = diag["basis_reasons"].get(reason, 0) + 1
                    total = float(g["total"])
                    renorm = max(0.0, min(100.0, total / basis * 100.0))
                    if abs(renorm - float(g["normalized"])) > 0.01:
                        diag["renormalized"] += 1
                    vals.append(renorm)
            if not ok:
                break
            arm_means[arm] = sum(vals) / len(vals)
        if ok:
            means[qid] = arm_means
        else:
            diag["skipped_incomplete"].append(qid)
    return means, diag


def paired_with_ratio(a: list[float], b: list[float], c: list[float]) -> dict:
    """One resampling loop -> co-indexed deltas AND the ratio's own CI (D3 #2)."""
    n = len(a)
    d_ba = sum(b[i] - a[i] for i in range(n)) / n
    d_ca = sum(c[i] - a[i] for i in range(n)) / n
    rng = random.Random(SEED)
    ba_boots, ca_boots, ratios = [], [], []
    n_bad_denom = 0
    for _ in range(N_BOOT):
        idx = [rng.randrange(n) for _ in range(n)]
        x = sum(b[i] - a[i] for i in idx) / n
        y = sum(c[i] - a[i] for i in idx) / n
        ba_boots.append(x)
        ca_boots.append(y)
        if y > MIN_DENOM:
            ratios.append(x / y)
        else:
            n_bad_denom += 1

    def ci(v: list[float]) -> tuple[float, float]:
        s = sorted(v)
        return s[int(0.025 * len(s))], s[min(len(s) - 1, int(0.975 * len(s)))]

    def pval(boots: list[float], delta: float) -> float:
        if delta >= 0:
            return min(1.0, 2.0 * sum(1 for d in boots if d <= 0) / len(boots))
        return min(1.0, 2.0 * sum(1 for d in boots if d >= 0) / len(boots))

    ba_lo, ba_hi = ci(ba_boots)
    ca_lo, ca_hi = ci(ca_boots)
    out = {
        "B_minus_A": {"delta": d_ba, "ci_low": ba_lo, "ci_high": ba_hi,
                      "p_value": pval(ba_boots, d_ba)},
        "C_minus_A": {"delta": d_ca, "ci_low": ca_lo, "ci_high": ca_hi,
                      "p_value": pval(ca_boots, d_ca)},
        "frac_resamples_denominator_unusable": n_bad_denom / N_BOOT,
    }
    if ratios:
        r_lo, r_hi = ci(ratios)
        out["recovery_ci"] = [r_lo, r_hi]
    return out


def branch_for(r: float) -> str:
    if r >= 2 / 3:
        return "STRUCTURE -> Q2 platform check"
    if r <= 1 / 3:
        return "KNOWLEDGE -> plan-caching probe"
    return "HYBRID -> plan-caching + self-plan fallback layer"


def analyze(units: dict, basis_mode: str) -> dict:
    means, diag = per_question_means(units, basis_mode)
    qids = sorted(means)
    a = [means[q]["A"] for q in qids]
    b = [means[q]["B"] for q in qids]
    c = [means[q]["C"] for q in qids]
    res = paired_with_ratio(a, b, c)
    ca = res["C_minus_A"]
    out = {
        "label": f"{LABEL}, n={len(qids)}, basis={basis_mode}",
        "n_questions": len(qids),
        "mean_A_noplan": statistics.mean(a),
        "mean_B_selfplan": statistics.mean(b),
        "mean_C_teacherplan": statistics.mean(c),
        **res,
        "diagnostics": diag,
    }
    # Transparency: what the ORIGINAL registered rule (CI merely excludes 0) would have
    # concluded, reported alongside the D3-tightened rule so the tightening's effect on the
    # verdict is visible rather than buried.
    if ca["ci_low"] > 0.0:
        r0 = res["B_minus_A"]["delta"] / ca["delta"]
        out["registered_rule_unfloored"] = {
            "gate_passes": True,
            "recovery": r0,
            "branch": branch_for(r0),
        }
    else:
        out["registered_rule_unfloored"] = {"gate_passes": False, "recovery": None}

    # D3 #1: one-sided gate with a magnitude floor.
    if ca["ci_low"] > MIN_DENOM:
        r = res["B_minus_A"]["delta"] / ca["delta"]
        out["recovery"] = r
        out["branch"] = branch_for(r)
        out["branch_key"] = branch_for(r).split()[0]
        if "recovery_ci" in res:
            lo, hi = res["recovery_ci"]
            spanned = sorted({branch_for(lo), branch_for(r), branch_for(hi)})
            out["branches_spanned_by_ratio_ci"] = spanned
            if len(spanned) > 1:
                out["branch_identifiable"] = False
    else:
        out["recovery"] = None
        out["branch_key"] = "STOP"
        out["branch"] = (
            "STOP / re-scope: the teacher-plan effect is not established on this stack "
            f"(C-A CI lower bound {ca['ci_low']:.2f} does not clear the +{MIN_DENOM} "
            "magnitude floor). recovery is undefined and must not be quoted."
        )
    return out


def main() -> int:
    units = load_units()
    graded = sum(1 for u in units.values() if u.get("grade"))
    expected = len({k.split("|")[0] for k in units}) * len(ARMS) * K_REPS
    print(f"[analyze] {graded} graded units of {expected} expected", flush=True)
    if graded < expected:
        raise SystemExit(
            f"FATAL: run incomplete ({graded}/{expected}) — analysis only after all cells land"
        )
    results = {}
    for mode in ("per_unit", "as_stored"):
        results[mode] = analyze(units, mode)
    primary, sens = results["per_unit"], results["as_stored"]
    results["sensitivity"] = {
        "primary_basis": "per_unit",
        "branch_agrees_across_bases": primary.get("branch_key") == sens.get("branch_key"),
        "recovery_per_unit": primary.get("recovery"),
        "recovery_as_stored": sens.get("recovery"),
    }
    print(json.dumps(results, indent=2))
    (ROOT / "runs" / "q1_summary.json").write_text(json.dumps(results, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
