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

    def test_missing_total_with_items_uses_fallback(self):
        # Missing TOTAL but scorable R items → reconstruct total, don't raise.
        bad = "R1: 5 — something\n(no total)"
        out = parse_judge_output(bad, max_points=18.0)
        assert out["total"] == 5.0
        # Clean parse still works and prefers the explicit TOTAL line.
        out2 = parse_judge_output(CLEAN, max_points=18.0)
        assert out2["total"] == 13.0

    def test_traps_and_bonuses(self):
        raw = "R1: 0 — none\ntrap T2: −5\nB3: +1\nTOTAL: 0\n"
        out = parse_judge_output(raw, max_points=10.0)
        assert out["traps_hit"] == ["T2"]
        assert out["bonuses"] == ["B3"]

    def test_missing_total_no_items_errors(self):
        # No TOTAL and no scorable items → unrecoverable, must raise.
        with pytest.raises(JudgeParseError, match="TOTAL"):
            parse_judge_output("just prose, no scores\n", max_points=10.0)

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
    """Missing TOTAL / empty output must trigger one repair-retry before hard failure."""

    def test_one_pass_retries_then_succeeds(self):
        # Genuinely unparseable (no TOTAL, no R items) → triggers repair-retry.
        # (missing-TOTAL-WITH-items now succeeds via deterministic fallback.)
        missing = "This response has no structured scores at all.\n"
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

    def test_one_pass_retries_empty_then_succeeds(self):
        """Empty first completion must repair without an empty assistant turn."""
        calls: list[list] = []

        def fake_chat(client, *, model, messages, temperature, max_tokens):
            calls.append(messages)
            if len(calls) == 1:
                return _fake_chat_response("")  # empty judge output
            return _fake_chat_response(CLEAN)

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
        # Repair is a fresh user nudge — no blank assistant turn.
        roles = [m["role"] for m in calls[1]]
        assert roles[-1] == "user"
        assert "EMPTY" in calls[1][-1]["content"]
        assert not any(
            m.get("role") == "assistant" and not (m.get("content") or "").strip()
            for m in calls[1]
        )
        assert out["total"] == 13.0

    def test_one_pass_retries_then_still_raises(self):
        # Unparseable on both passes (no items, no TOTAL) → still raises.
        missing = "no scores here either\n"

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

    def test_grade_retries_empty_via_shared_path(self):
        """grade() → _one_pass so uplift gate (FinanceAdapter.run_item) is covered."""
        from correction.judge import grade

        calls: list[dict] = []

        def fake_chat(client, *, model, messages, temperature, max_tokens):
            calls.append({"max_tokens": max_tokens})
            text = "" if len(calls) == 1 else CLEAN
            return _fake_chat_response(text)

        with (
            patch("correction.judge._judge_client", return_value=MagicMock()),
            patch("correction.judge._chat_with_retry", side_effect=fake_chat),
            patch("correction.judge._assert_judge_ne_teacher"),
        ):
            out = grade("q?", SAMPLE_RUBRIC, "student answer", model="openai/gpt-5.2")
        assert len(calls) == 2
        assert calls[1]["max_tokens"] == 3072  # empty → bumped budget
        assert out["total"] == 13.0

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


def test_total_fallback_from_items_when_total_missing():
    from correction.judge import parse_judge_output
    # No TOTAL line — must reconstruct from R items minus traps plus bonuses.
    raw = (
        "R1: 8 — solid framework\n"
        "R2: 5 — partial calc\n"
        "trap T3: -4 aggregation error\n"
        "bonus B1: +2 nice synthesis\n"
    )
    out = parse_judge_output(raw, max_points=20.0)
    assert out["total"] == 8 + 5 - 4 + 2  # 11
    assert out["traps_hit"] == ["T3"]
    assert out["items"] == {"R1": 8.0, "R2": 5.0}


def test_still_errors_when_no_total_and_no_items():
    from correction.judge import parse_judge_output, JudgeParseError
    import pytest
    with pytest.raises(JudgeParseError):
        parse_judge_output("some prose with no scores at all", max_points=20.0)
