"""Hermetic tests for uplift-gated memory selection."""
from __future__ import annotations

from contracts.schemas import AgentConfig, Difficulty, FewShotExample, TelemetryRecord
from correction.tracelift import build_val_slice, select_uplift_memory
from harness.feed import FeedItem


def _item(qid: str, q: str, phase: str = "degraded") -> FeedItem:
    return FeedItem(
        question_id=qid,
        question=q,
        gold_output="1",
        domain_id="algebra",
        difficulty="hard",
        phase=phase,
    )


class _FakeAdapter:
    """Deterministic pass rates: with helper example → always 1; bare → 0 on val."""

    def __init__(self, helpful_questions: set[str]):
        self.helpful_questions = helpful_questions

    def run_item(self, item, config, use_rules=True):
        del use_rules
        has_ex = bool(config.few_shot_examples)
        if has_ex:
            q = config.few_shot_examples[0].question
            acc = 1.0 if q in self.helpful_questions else 0.0
        else:
            acc = 0.0
        return TelemetryRecord(
            run_id=f"{item.question_id}_x",
            timestamp=0.0,
            difficulty=Difficulty.HARD,
            execution_accuracy=acc,
            query_valid=True,
        )


class TestSelectUplift:
    def test_keeps_positive_u_and_caps(self):
        helpful = FewShotExample(
            question="help me", correct_output="2", domain_id="algebra", source="gold"
        )
        useless = FewShotExample(
            question="no help", correct_output="3", domain_id="algebra", source="gold"
        )
        adapter = _FakeAdapter({"help me"})
        val = [_item("v1", "val problem one"), _item("v2", "val problem two")]
        cfg = AgentConfig(config_id="t", model="m", few_shot_examples=[])
        kept, scored = select_uplift_memory(
            adapter, cfg, [useless, helpful], val, max_keep=1, k=1, min_u=0.0
        )
        assert len(kept) == 1
        assert kept[0].question == "help me"
        assert kept[0].source == "uplift"
        by_q = {e.question: u for e, u in scored}
        assert by_q["help me"] > 0
        assert by_q["no help"] <= 0

    def test_drops_all_nonpositive(self):
        bad = FewShotExample(
            question="no help", correct_output="3", domain_id="algebra", source="gold"
        )
        adapter = _FakeAdapter(set())
        val = [_item("v1", "val problem")]
        cfg = AgentConfig(config_id="t", model="m")
        kept, _ = select_uplift_memory(
            adapter, cfg, [bad], val, max_keep=5, k=1, min_u=0.0
        )
        assert kept == []


class TestValSlice:
    def test_excludes_candidate_questions(self):
        items = [
            _item("a", "cand q"),
            _item("b", "other q"),
            _item("c", "other q 2"),
        ]
        out = build_val_slice(items, {"cand q"}, n=10)
        assert [i.question for i in out] == ["other q", "other q 2"]
