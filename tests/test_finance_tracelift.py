"""Hermetic tests for finance TraceLift build loop helpers."""
from __future__ import annotations

from pathlib import Path

from contracts.schemas import Difficulty, FewShotExample, TelemetryRecord
from harness.feed import FeedItem
from scripts import finance_tracelift as tl


class TestStoppingRule:
    def test_window_mean_triggers(self):
        recent = [0.1] * 15
        stop, reason = tl.should_stop(recent, n_considered=15, n_admitted=10)
        assert stop
        assert "mean uplift" in reason

    def test_low_admit_rate_triggers(self):
        recent = [2.0] * 15  # high uplift but few admits
        stop, reason = tl.should_stop(recent, n_considered=15, n_admitted=2)
        assert stop
        assert "admission rate" in reason

    def test_no_stop_early(self):
        stop, _ = tl.should_stop([0.0] * 5, n_considered=5, n_admitted=0)
        assert not stop


class TestCompactAndFreeze:
    def test_compact_caps_per_category(self, tmp_path: Path):
        items = [
            FewShotExample(
                question=f"[FINANCE_TRAP] Accounting {i}",
                correct_output=f"trap-{i}",
                domain_id="Accounting",
                source="tracelift",
            )
            for i in range(10)
        ]
        compact = tl.compact_memory(items, max_per_category=3)
        assert len(compact) == 3
        out = tmp_path / "mem.json"
        payload = tl.freeze_memory(compact, out, meta={"t": 1})
        assert out.exists()
        assert payload["n"] == 3
        loaded = tl.load_memory(out)
        assert len(loaded) == 3
        assert loaded[0].source == "tracelift"


class TestUNorm:
    def test_accuracy_to_normalized(self):
        assert abs(tl.u_normalized(0.01) - 1.0) < 1e-9


class TestGateWithMockAdapter:
    def test_admits_positive_uplift(self, tmp_path: Path):
        state = tmp_path / "state.jsonl"
        cand = FewShotExample(
            question="[FINANCE_PLAYBOOK] Accounting",
            correct_output="gate equity first",
            domain_id="Accounting",
            source="tracelift",
        )
        val = [
            FeedItem(
                question_id="v1",
                question="q",
                gold_output="",
                domain_id="Accounting",
                difficulty="hard",
                phase="degraded",
            )
        ]

        class FakeAdapter:
            def run_item(self, item, config, use_rules=True):
                # With memory → higher score
                score = 0.40 if config.few_shot_examples else 0.20
                return TelemetryRecord(
                    run_id="r",
                    timestamp=0.0,
                    difficulty=Difficulty.HARD,
                    execution_accuracy=score,
                    query_valid=True,
                    generated_complexity=0,
                    required_complexity=0,
                    generated_output="a",
                    gold_output="",
                    domain_id=item.domain_id,
                )

        admitted, scored = tl.gate_candidates(
            [cand],
            "fake-model",
            val,
            state_path=state,
            k=1,
            min_u_norm=1.0,  # 0.20 accuracy delta = 20 pts
            resume=False,
            adapter=FakeAdapter(),
        )
        assert len(admitted) == 1
        assert scored[0]["admitted"] is True
        assert scored[0]["u_norm"] == 20.0
