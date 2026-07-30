"""Regressions for the D3 analysis corrections (Q1).

The shipped gate in scripts/q1_self_plan.py:439 was two-sided, so a teacher effect whose
CI lay entirely BELOW zero passed it, divided by a negative denominator, and printed
"KNOWLEDGE load-bearing" from data showing teacher plans hurt.
"""
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from q1_analyze import MIN_DENOM, branch_for, paired_with_ratio  # noqa: E402


def _base(n: int = 40) -> list[float]:
    rng = random.Random(0)
    return [rng.gauss(70, 10) for _ in range(n)]


def test_negative_teacher_effect_is_rejected_not_branched():
    base = _base()
    r = paired_with_ratio(base, [x + 2 for x in base], [x - 6 for x in base])
    ca = r["C_minus_A"]
    assert ca["delta"] < 0
    assert ca["ci_high"] < 0, "CI lies entirely below zero"
    assert not (ca["ci_low"] > MIN_DENOM), "one-sided gate must reject a harmful teacher effect"


def test_tiny_positive_denominator_is_rejected():
    base = _base()
    r = paired_with_ratio(base, [x + 2 for x in base], [x + 0.5 for x in base])
    ca = r["C_minus_A"]
    assert ca["delta"] > 0
    assert not (ca["ci_low"] > MIN_DENOM), "magnitude floor must reject a 0.5-point denominator"


def test_real_effect_passes_and_ratio_ci_is_reported():
    base = _base()
    r = paired_with_ratio(base, [x + 2 for x in base], [x + 5 for x in base])
    ca = r["C_minus_A"]
    assert ca["ci_low"] > MIN_DENOM
    assert "recovery_ci" in r, "the ratio must ship with its own CI"
    assert branch_for(r["B_minus_A"]["delta"] / ca["delta"]).startswith("HYBRID")


def test_branch_thresholds_match_prereg():
    assert branch_for(0.70).startswith("STRUCTURE")
    assert branch_for(0.20).startswith("KNOWLEDGE")
    assert branch_for(0.50).startswith("HYBRID")


if __name__ == "__main__":
    test_negative_teacher_effect_is_rejected_not_branched()
    test_tiny_positive_denominator_is_rejected()
    test_real_effect_passes_and_ratio_ci_is_reported()
    test_branch_thresholds_match_prereg()
    print("D3 analysis corrections: OK")
