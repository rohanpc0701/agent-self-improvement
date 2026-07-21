"""Hermetic tests for the PRBench weighted-criteria judge + adapter firewall."""
from __future__ import annotations

import json

import pytest

from correction.prbench_judge import (
    PRBenchJudgeError,
    parse_decisions,
    rubric_max_points,
    score_from_decisions,
)

RUBRIC = [
    {"description": "sizes the shock", "weight": 10.0, "weight_class": "critically important"},
    {"description": "builds interest cost", "weight": 6.0, "weight_class": "important"},
    {"description": "recommends dividend cut w/o analysis", "weight": -5.0, "weight_class": "detrimental"},
]


class TestScoring:
    def test_max_points_is_sum_of_positive(self):
        assert rubric_max_points(RUBRIC) == 16.0

    def test_all_positive_satisfied_full_score(self):
        d = {1: True, 2: True, 3: False}
        out = score_from_decisions(RUBRIC, d)
        assert out["raw"] == 16.0
        assert out["normalized"] == 100.0

    def test_detrimental_committed_penalized(self):
        d = {1: True, 2: True, 3: True}  # committed the bad thing
        out = score_from_decisions(RUBRIC, d)
        assert out["raw"] == 11.0  # 16 - 5
        assert abs(out["normalized"] - (11 / 16 * 100)) < 1e-6

    def test_negative_raw_clamped_to_zero(self):
        bad = [{"description": "x", "weight": 2.0, "weight_class": "important"},
               {"description": "y", "weight": -9.0, "weight_class": "critically detrimental"}]
        out = score_from_decisions(bad, {1: False, 2: True})
        assert out["raw"] == -9.0
        assert out["normalized"] == 0.0

    def test_no_positive_weights_errors(self):
        with pytest.raises(PRBenchJudgeError):
            score_from_decisions([{"description": "x", "weight": -3.0, "weight_class": "detrimental"}], {1: True})


class TestParse:
    def test_parses_yes_no_lines(self):
        raw = "C1: yes\nC2: no\nC3: Yes"
        assert parse_decisions(raw, 3) == {1: True, 2: False, 3: True}

    def test_ignores_out_of_range(self):
        assert parse_decisions("C9: yes", 3) == {}

    def test_empty_when_unparseable(self):
        assert parse_decisions("some prose", 3) == {}


class TestAdapterFirewall:
    def test_student_never_sees_rubric(self, monkeypatch):
        ds = {"items": [{"id": "t1", "topic": "Corporate Finance", "question": "Q",
                         "rubric": [{"description": "d", "weight": 5.0, "weight_class": "important"}]}]}
        man = {"train_ids": ["t1"], "validation_ids": [], "heldout_ids": []}
        import adapters.prbench as pr
        monkeypatch.setattr(pr, "_TASKS", {t["id"]: t for t in ds["items"]})
        monkeypatch.setattr(pr, "load_manifest", lambda path=None: man)
        with pytest.raises(PermissionError):
            pr.rubric_for("t1", role="student")
        # teacher CAN see a train rubric
        assert pr.rubric_for("t1", role="teacher") == ds["items"][0]["rubric"]

    def test_teacher_blocked_on_heldout_rubric(self, monkeypatch):
        ds = {"items": [{"id": "h1", "topic": "Corporate Finance", "question": "Q",
                         "rubric": [{"description": "d", "weight": 5.0, "weight_class": "important"}]}]}
        man = {"train_ids": [], "validation_ids": [], "heldout_ids": ["h1"]}
        import adapters.prbench as pr
        monkeypatch.setattr(pr, "_TASKS", {t["id"]: t for t in ds["items"]})
        monkeypatch.setattr(pr, "load_manifest", lambda path=None: man)
        with pytest.raises(PermissionError):
            pr.rubric_for("h1", role="teacher")
