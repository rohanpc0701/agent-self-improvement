"""Tests for correction.memory and harness continuous feed."""
from __future__ import annotations

import pytest

from contracts.schemas import FewShotExample
from correction.memory import merge_examples
from harness.feed import build_continuous_stream


def _ex(q: str, db: str = "db1") -> FewShotExample:
    return FewShotExample(question=q, correct_sql="SELECT 1", db_id=db, source="gold")


class TestMergeExamples:
    def test_dedupes_by_question(self):
        a = [_ex("q1"), _ex("q2")]
        b = [_ex("q1", "db1"), _ex("q3")]
        merged = merge_examples(a, b)
        assert len(merged) == 3
        assert {e.question for e in merged} == {"q1", "q2", "q3"}

    def test_per_db_cap_fifo(self):
        existing = [_ex(f"q{i}", "dbA") for i in range(10)]
        merged = merge_examples(existing, [], max_per_db=3, max_total=32)
        assert len(merged) == 3
        assert merged[0].question == "q7"

    def test_total_cap(self):
        items = [_ex(f"q{i}", f"db{i}") for i in range(40)]
        merged = merge_examples([], items, max_total=10)
        assert len(merged) == 10


class TestDetectorResume:
    def test_resume_allows_second_drift(self):
        from detector.config import DetectorConfig
        from detector.detector import Detector
        from detector.tests.test_detector import _make_rec

        cfg = DetectorConfig(baseline_len=5, window=3, min_sustained=2, drop_threshold=0.2)
        det = Detector(cfg)
        for i in range(5):
            det.update(_make_rec(i, acc=1.0))
        ev1 = None
        for i in range(5, 15):
            ev1 = det.update(_make_rec(i, acc=0.0))
            if ev1 is not None:
                break
        assert ev1 is not None
        det.resume_after_correction(cooldown=0)
        ev2 = None
        for i in range(15, 25):
            ev2 = det.update(_make_rec(i, acc=0.0))
            if ev2 is not None:
                break
        assert ev2 is not None


class TestContinuousFeed:
    def test_multiple_cycles(self):
        qs = (
            [{"id": f"e{i}", "question": "q", "expected_sql": "SELECT 1", "db_id": "d", "difficulty": "easy"} for i in range(5)]
            + [{"id": f"h{i}", "question": "q", "expected_sql": "SELECT 1", "db_id": "d2", "difficulty": "hard"} for i in range(10)]
            + [{"id": f"x{i}", "question": "q", "expected_sql": "SELECT 1", "db_id": "d3", "difficulty": "extra"} for i in range(5)]
        )
        items = build_continuous_stream(qs, n_baseline=4, n_degraded=3, n_recovery=3, n_cycles=2, seed=1)
        phases = [i.phase for i in items]
        assert phases[:4] == ["baseline"] * 4
        assert phases.count("degraded") == 6
        assert phases.count("recovery") == 6


class TestGSM8KAdapter:
    def test_verify_answer(self):
        from adapters.gsm8k_math import verify_answer, extract_answer

        assert verify_answer("Step by step...\n#### 42", "42") == 1.0
        assert verify_answer("The answer is 41", "42") == 0.0
        assert extract_answer("foo\n#### 1,234") == "1234"

    def test_load_fixture(self):
        from adapters.gsm8k_math import load_gsm8k_questions

        qs = load_gsm8k_questions()
        assert len(qs) >= 40
        assert all("expected_sql" in q for q in qs)

