"""Hermetic tests for rubric judge parser + model firewall."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from correction.judge import (
    JudgeParseError,
    _one_pass,
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

    def test_bold_total_accepted(self):
        raw = "R1: 5 — x\n**TOTAL: 5**\n"
        out = parse_judge_output(raw, max_points=10.0)
        assert out["total"] == 5.0

    def test_empty_errors(self):
        with pytest.raises(JudgeParseError):
            parse_judge_output("   ", max_points=10.0)


def _fake_chat_response(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
    )


class TestRepairRetry:
    """Missing TOTAL must trigger one repair-retry before hard failure."""

    def test_one_pass_retries_then_succeeds(self):
        missing = "R1: 10 — cites ASC 810\nR2: 4 — partial\n"  # no TOTAL
        calls: list[str] = []

        def fake_chat(client, *, model, messages, temperature, max_tokens):
            # Second call is the repair pass.
            text = CLEAN if len(calls) else missing
            calls.append(messages[-1]["content"] if messages else "")
            return _fake_chat_response(text)

        with (
            patch("correction.judge._judge_client", return_value=MagicMock()),
            patch("correction.judge._chat_with_retry", side_effect=fake_chat),
        ):
            out = _one_pass(
                "q?",
                SAMPLE_RUBRIC,
                "student answer",
                model="openai/gpt-5.2",
                max_points=18.0,
            )
        assert len(calls) == 2
        assert "TOTAL" in calls[1]  # repair prompt emphasizes TOTAL
        assert out["total"] == 13.0

    def test_one_pass_retries_then_still_raises(self):
        missing = "R1: 1 — x\n"

        def fake_chat(client, *, model, messages, temperature, max_tokens):
            return _fake_chat_response(missing)

        with (
            patch("correction.judge._judge_client", return_value=MagicMock()),
            patch("correction.judge._chat_with_retry", side_effect=fake_chat),
            pytest.raises(JudgeParseError, match="TOTAL"),
        ):
            _one_pass(
                "q?",
                SAMPLE_RUBRIC,
                "student answer",
                model="openai/gpt-5.2",
                max_points=18.0,
            )

    def test_grade_prompt_requires_total_line(self):
        from correction.judge import _build_judge_messages

        msgs = _build_judge_messages("q", SAMPLE_RUBRIC, "a")
        blob = msgs[0]["content"] + "\n" + msgs[1]["content"]
        assert "TOTAL: <number>" in blob
        assert "MUST be exactly" in blob or "last non-empty line MUST" in blob


class TestJudgeNeTeacher:
    def test_import_time_models_differ(self):
        import correction.judge as j

        assert j.JUDGE_MODEL != j.TEACHER_MODEL
