"""Regressions for two measurement bugs found 2026-07-30 (Q1 run).

1. Truncated judge output (cut off before TOTAL) was silently scored by summing the
   partial item list -> deflated scores, correlated with answer length and therefore
   with arm. Must raise instead.
2. Rubrics mostly declare `MAX: 100` while the item ladder sums to 70-92; normalizing
   TOTAL by the ladder sum over-scaled 30 of 40 Q1 questions by 1.06-1.35x.
"""
import pytest

from correction.judge import JudgeParseError, parse_judge_output

TRUNCATED = """R1: 8 / 10 — solid coverage
R2: 4 / 10 — partial
trap T5: -6 — treats the roll-up as neutral and not as captur"""

COMPLETE = """R1: 8 / 10 — solid coverage
R2: 4 / 10 — partial
trap T1: -6 — missed the covenant
SCALED BASE: 33 / 94
TRAPS: -6
TOTAL: 18
MAX: 100"""


def test_truncated_output_raises_instead_of_deflating():
    with pytest.raises(JudgeParseError, match="missing TOTAL"):
        parse_judge_output(TRUNCATED, max_points=74)


def test_fraction_style_items_parse():
    out = parse_judge_output(COMPLETE, max_points=74)
    assert out["items"] == {"R1": 8.0, "R2": 4.0}, "the `/ <max>` form must parse"
    assert out["traps_hit"] == ["T1"]


def test_declared_max_read_from_rubric_not_judge_output():
    """The denominator must come from the rubric (deterministic per question), not
    from the judge's output — an output-derived basis varies with completeness."""
    from correction.judge import rubric_declared_max

    assert rubric_declared_max("## Output Format\n- followed by a line exactly: `MAX: 100`") == 100.0
    assert rubric_declared_max("- Max (before bonuses): 80\n- MAX: 100") == 100.0
    assert rubric_declared_max("no basis declared here") is None
    # A judge-emitted MAX line must NOT move the denominator.
    out = parse_judge_output(COMPLETE, max_points=74)
    assert out["max"] == 74.0, "parse must use the caller's basis, not the output's"


if __name__ == "__main__":
    test_truncated_output_raises_instead_of_deflating()
    test_fraction_style_items_parse()
    test_declared_max_read_from_rubric_not_judge_output()
    print("judge parse fixes: OK")
