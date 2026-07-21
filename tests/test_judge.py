"""Hermetic tests for rubric judge parser + model firewall."""
from __future__ import annotations

import pytest

from correction.judge import (
    JudgeParseError,
    parse_judge_output,
    rubric_max_points,
)


SAMPLE_RUBRIC = """
## Item R1 — Foo (max 10)
- **0 pts** ...
- **10 pts** ...
## Item R2 — Bar (max 8)
- **0 pts** ...
- **8 pts** ...
## Trap T1 — Bad pattern (−3)
## Insight Bonus B1 (+2)
"""


CLEAN = """
R1: 10 — cites ASC 810
R2: 4 — partial kick-out analysis
trap T1: −3
bonus B1: +2
TOTAL: 13
"""


class TestRubricMax:
    def test_sum_item_caps(self):
        assert rubric_max_points(SAMPLE_RUBRIC) == 18.0


class TestParse:
    def test_clean(self):
        out = parse_judge_output(CLEAN, max_points=18.0)
        assert out["total"] == 13.0
        assert out["items"]["R1"] == 10.0
        assert out["items"]["R2"] == 4.0
        assert out["traps_hit"] == ["T1"]
        assert "B1" in out["bonuses"]
        assert abs(out["normalized"] - (13 / 18) * 100) < 1e-6

    def test_malformed_then_repaired(self):
        bad = "R1: 5 — something\n(no total)"
        with pytest.raises(JudgeParseError):
            parse_judge_output(bad, max_points=18.0)
        # After repair, clean parse works
        out = parse_judge_output(CLEAN, max_points=18.0)
        assert out["total"] == 13.0

    def test_traps_and_bonuses(self):
        raw = "R1: 0 — none\ntrap T2: −5\nB3: +1\nTOTAL: 0\n"
        out = parse_judge_output(raw, max_points=10.0)
        assert out["traps_hit"] == ["T2"]
        assert out["bonuses"] == ["B3"]

    def test_missing_total_errors(self):
        with pytest.raises(JudgeParseError, match="TOTAL"):
            parse_judge_output("R1: 1 — x\n", max_points=10.0)

    def test_empty_errors(self):
        with pytest.raises(JudgeParseError):
            parse_judge_output("   ", max_points=10.0)


class TestJudgeNeTeacher:
    def test_import_time_models_differ(self):
        import correction.judge as j

        assert j.JUDGE_MODEL != j.TEACHER_MODEL
