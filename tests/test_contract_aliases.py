"""Tests for domain-agnostic contract names + legacy SQL aliases."""
from __future__ import annotations

from contracts.schemas import Difficulty, FailureMode, FewShotExample, TelemetryRecord


def test_telemetry_accepts_legacy_sql_keys():
    rec = TelemetryRecord.model_validate(
        {
            "run_id": "r1",
            "timestamp": 1.0,
            "difficulty": "hard",
            "execution_accuracy": 0.0,
            "query_valid": False,
            "generated_sql": "SELECT 1",
            "db_id": "concert_singer",
        }
    )
    assert rec.generated_output == "SELECT 1"
    assert rec.domain_id == "concert_singer"


def test_few_shot_accepts_legacy_keys():
    ex = FewShotExample.model_validate(
        {"question": "q", "correct_sql": "SELECT 1", "db_id": "db"}
    )
    assert ex.correct_output == "SELECT 1"
    assert ex.domain_id == "db"


def test_failure_mode_maps_invalid_sql():
    assert FailureMode("invalid_sql") is FailureMode.INVALID_OUTPUT
    assert FailureMode.INVALID_OUTPUT.value == "invalid_output"


def test_dump_uses_canonical_names():
    rec = TelemetryRecord(
        run_id="r1",
        timestamp=1.0,
        difficulty=Difficulty.HARD,
        execution_accuracy=1.0,
        query_valid=True,
        generated_output="def f():\n    return 1",
        domain_id="dp",
    )
    data = rec.model_dump()
    assert "generated_output" in data
    assert "domain_id" in data
    assert "generated_sql" not in data
