"""Tests for orchestrator.py: _build_feed phase structure and dry_run_heldout aggregation.

All tests are hermetic — no API keys, no SQLite, no events.jsonl writes.
run_item and load_questions are patched at the orchestrator module level.
"""
from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from contracts.schemas import (
    AgentConfig,
    CorrectionAction,
    Difficulty,
    FewShotExample,
    TelemetryRecord,
)
from harness.feed import FeedItem
from orchestrator import _build_feed, compare_teacher_run, dry_run_heldout, run_ablation_eval


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_questions(n_easy: int = 20, n_hard: int = 20) -> list[dict]:
    easy = [
        {"id": f"e{i}", "question": "q", "expected_sql": "SELECT 1", "db_id": "db", "difficulty": "easy"}
        for i in range(n_easy)
    ]
    hard = [
        {"id": f"h{i}", "question": "q", "expected_sql": "SELECT 1", "db_id": "db", "difficulty": "hard"}
        for i in range(n_hard)
    ]
    return easy + hard


def _make_item(phase: str, difficulty: str = "hard", idx: int = 0) -> FeedItem:
    return FeedItem(
        question_id=f"q_{phase}_{idx}",
        question=f"question {idx}",
        gold_output="SELECT COUNT(*) FROM t",
        domain_id="db",
        difficulty=difficulty,
        phase=phase,
    )


def _make_record(accuracy: float, difficulty: str = "hard") -> TelemetryRecord:
    return TelemetryRecord(
        run_id=f"run_{int(accuracy * 100)}",
        timestamp=time.time(),
        difficulty=Difficulty(difficulty),
        execution_accuracy=accuracy,
        query_valid=accuracy > 0.0,
    )


def _make_mixed_items(n_baseline: int = 2, n_recovery: int = 4) -> list[FeedItem]:
    items = [_make_item("baseline", "easy", i) for i in range(n_baseline)]
    for i in range(n_recovery):
        diff = "hard" if i % 2 == 0 else "extra"
        items.append(_make_item("recovery", diff, 100 + i))
    return items


# ---------------------------------------------------------------------------
# _build_feed
# ---------------------------------------------------------------------------

class TestBuildFeed:
    def test_phases_present(self):
        with patch("harness.spider.load_questions", return_value=_make_questions()):
            items = _build_feed(n=5, full=False)
        assert {i.phase for i in items} == {"baseline", "degraded", "recovery"}

    def test_n_per_phase(self):
        with patch("harness.spider.load_questions", return_value=_make_questions()):
            items = _build_feed(n=7, full=False)
        for phase in ("baseline", "degraded", "recovery"):
            assert len([i for i in items if i.phase == phase]) == 7

    def test_full_uses_80(self):
        with patch("harness.spider.load_questions", return_value=_make_questions()):
            items = _build_feed(n=5, full=True)
        assert len([i for i in items if i.phase == "baseline"]) == 80

    def test_full_overrides_n(self):
        with patch("harness.spider.load_questions", return_value=_make_questions()):
            items = _build_feed(n=10, full=True)
        # full=True always wins regardless of --n
        assert len([i for i in items if i.phase == "recovery"]) == 80

    def test_degraded_and_recovery_disjoint(self):
        """Core benchmark-credibility invariant: no leakage from learn into held-out."""
        with patch("harness.spider.load_questions", return_value=_make_questions()):
            items = _build_feed(n=10, full=False)
        deg_ids = {i.question_id for i in items if i.phase == "degraded"}
        rec_ids = {i.question_id for i in items if i.phase == "recovery"}
        assert deg_ids.isdisjoint(rec_ids), (
            "LEAKAGE: recovery pool overlaps learn pool — recovery accuracy "
            "would be inflated by memorisation, not learning."
        )

    def test_deterministic_across_calls(self):
        qs = _make_questions()
        with patch("harness.spider.load_questions", return_value=qs):
            items1 = _build_feed(n=5, full=False)
        with patch("harness.spider.load_questions", return_value=qs):
            items2 = _build_feed(n=5, full=False)
        assert [(i.question_id, i.phase) for i in items1] == [(i.question_id, i.phase) for i in items2]

    def test_baseline_difficulty(self):
        with patch("harness.spider.load_questions", return_value=_make_questions()):
            items = _build_feed(n=10, full=False)
        baseline = [i for i in items if i.phase == "baseline"]
        assert all(i.difficulty in ("easy", "medium") for i in baseline)

    def test_recovery_difficulty(self):
        with patch("harness.spider.load_questions", return_value=_make_questions()):
            items = _build_feed(n=10, full=False)
        recovery = [i for i in items if i.phase == "recovery"]
        assert all(i.difficulty in ("hard", "extra") for i in recovery)


# ---------------------------------------------------------------------------
# dry_run_heldout
# ---------------------------------------------------------------------------

class TestDryRunHeldout:
    def test_only_recovery_items_run(self):
        items = _make_mixed_items(n_baseline=3, n_recovery=4)
        called_phases = []

        def fake_run(item, config, adapter_name="spider", use_rules=True):
            called_phases.append(item.phase)
            return _make_record(0.5)

        with patch("orchestrator._run_item", side_effect=fake_run):
            dry_run_heldout(items)

        assert all(p == "recovery" for p in called_phases)
        assert len(called_phases) == 4

    def test_uses_empty_few_shot_examples(self):
        items = _make_mixed_items(n_recovery=2)
        seen_configs: list[AgentConfig] = []

        def fake_run(item, config, adapter_name="spider", use_rules=True):
            seen_configs.append(config)
            return _make_record(1.0)

        with patch("orchestrator._run_item", side_effect=fake_run):
            dry_run_heldout(items)

        assert all(len(c.few_shot_examples) == 0 for c in seen_configs), (
            "dry_run_heldout must not inject any few-shot examples — "
            "it measures base-config performance only."
        )

    def test_overall_accuracy_is_mean(self):
        items = _make_mixed_items(n_recovery=4)
        records = [_make_record(1.0), _make_record(0.0), _make_record(1.0), _make_record(0.0)]

        with patch("orchestrator._run_item", side_effect=records):
            result = dry_run_heldout(items)

        assert result["overall"] == pytest.approx(0.5)

    def test_none_records_excluded_from_accuracy(self):
        items = _make_mixed_items(n_recovery=3)
        # gold-SQL failure (None), then one correct, one wrong
        records = [None, _make_record(1.0), _make_record(0.0)]

        with patch("orchestrator._run_item", side_effect=records):
            result = dry_run_heldout(items)

        # only 2 scored: (1.0 + 0.0) / 2 = 0.5
        assert result["overall"] == pytest.approx(0.5)

    def test_all_skipped_returns_zero(self):
        items = _make_mixed_items(n_recovery=3)
        with patch("orchestrator._run_item", return_value=None):
            result = dry_run_heldout(items)
        assert result["overall"] == 0.0

    def test_explicit_config_forwarded(self):
        items = _make_mixed_items(n_recovery=1)
        custom = AgentConfig(config_id="custom-test", model="test-model", few_shot_examples=[])
        seen: list[AgentConfig] = []

        def fake_run(item, config, adapter_name="spider", use_rules=True):
            seen.append(config)
            return _make_record(1.0)

        with patch("orchestrator._run_item", side_effect=fake_run):
            dry_run_heldout(items, config=custom)

        assert len(seen) == 1
        assert seen[0].config_id == "custom-test"

    def test_default_config_uses_base_model(self):
        items = _make_mixed_items(n_recovery=1)
        seen: list[AgentConfig] = []

        def fake_run(item, config, adapter_name="spider", use_rules=True):
            seen.append(config)
            return _make_record(0.0)

        with patch("orchestrator._run_item", side_effect=fake_run):
            dry_run_heldout(items)

        # _BASE_MODEL honors the AGENT_MODEL env override, so assert against it
        # rather than a hardcoded name.
        from orchestrator import _BASE_MODEL
        assert seen[0].model == _BASE_MODEL

    def test_returns_dict_with_overall_key(self):
        items = _make_mixed_items(n_recovery=2)
        records = [_make_record(0.8), _make_record(0.4)]

        with patch("orchestrator._run_item", side_effect=records):
            result = dry_run_heldout(items)

        assert isinstance(result, dict)
        assert result["overall"] == pytest.approx(0.6)

    def test_returns_per_difficulty_keys(self):
        # Items: 2 hard, 2 extra recovery items
        items = _make_mixed_items(n_recovery=4)
        records = [
            _make_record(1.0, "hard"),
            _make_record(0.0, "extra"),
            _make_record(1.0, "hard"),
            _make_record(0.0, "extra"),
        ]
        with patch("orchestrator._run_item", side_effect=records):
            result = dry_run_heldout(items)
        # hard bucket: (1.0 + 1.0)/2 = 1.0; extra bucket: 0.0
        assert "hard" in result or "extra" in result  # at least one difficulty key

    def test_no_recovery_items_returns_zero_overall(self):
        items = [_make_item("baseline", "easy", i) for i in range(3)]
        with patch("orchestrator._run_item") as mock_run:
            result = dry_run_heldout(items)
        mock_run.assert_not_called()
        assert result["overall"] == 0.0


class TestCompareTeacher:
    def test_student_memory_vs_teacher_table(self, capsys):
        items = [
            _make_item("recovery", "hard", 0),
            _make_item("recovery", "hard", 1),
            _make_item("baseline", "easy", 2),
        ]
        action = CorrectionAction(
            triggered_by="execution_accuracy",
            new_few_shot_examples=[
                FewShotExample(question="q", correct_output="SELECT 1", domain_id="db")
            ],
        )

        def fake_run(item, config, adapter_name="coding", use_rules=True):
            # student gets Q0 right, Q1 wrong
            acc = 1.0 if item.question_id.endswith("_0") else 0.0
            return _make_record(acc, item.difficulty)

        def fake_teacher(item, adapter_name):
            # teacher gets both right
            return 1.0

        with (
            patch("contracts.eventlog.read_events", return_value=[action]),
            patch("orchestrator._run_item", side_effect=fake_run),
            patch("orchestrator._teacher_score_item", side_effect=fake_teacher),
        ):
            compare_teacher_run(items, adapter_name="coding")

        out = capsys.readouterr().out
        assert "Student + few-shots/KG" in out
        assert "Teacher (unaided)" in out
        assert "0.500" in out  # student mean
        assert "1.000" in out  # teacher mean


class TestAblationArms:
    def test_arm_env_matrix(self, monkeypatch):
        import orchestrator as orch

        calls = []

        def fake_run_item(item, config, adapter_name="coding", use_rules=True):
            import os

            calls.append((os.environ.get("AGENT_USE_EXAMPLES"), use_rules))
            return TelemetryRecord(
                run_id=f"{item.question_id}_x",
                timestamp=0.0,
                difficulty=Difficulty(item.difficulty),
                execution_accuracy=1.0,
                query_valid=True,
                injection_stats={
                    "examples_available": 2,
                    "examples_injected": 0,
                    "example_ids": [],
                    "rules_injected": 0,
                },
            )

        monkeypatch.setattr(orch, "_run_item", fake_run_item)
        monkeypatch.setattr(orch, "_load_latest_examples", lambda: [])
        monkeypatch.setattr(orch, "append_event", lambda *_a, **_k: None)

        items = [
            FeedItem(
                question_id="q1",
                question="?",
                gold_output="def f(): pass",
                domain_id="dp",
                difficulty="hard",
                phase="recovery",
            )
        ]

        result = orch.run_ablation_eval(items, "coding", ["none", "both"])

        assert ("0", False) in calls  # none arm
        assert ("1", True) in calls  # both arm
        assert result["none"]["n"] == 1
        assert result["none"]["zero_injection_pct"] == 100.0
