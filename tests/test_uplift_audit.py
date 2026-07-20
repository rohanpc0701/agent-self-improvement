"""Hermetic tests for uplift_audit core."""
from __future__ import annotations

import json

from contracts.schemas import FewShotExample
from scripts.uplift_audit import (
    audit_plan,
    filter_plan_for_items,
    load_done_cells,
    parse_items_range,
    pending_specs,
    summarize,
)


def _bundle(n: int) -> list[FewShotExample]:
    return [
        FewShotExample(
            question=f"q{i}",
            correct_output=f"def f{i}(): pass",
            domain_id="dp",
            source="teacher",
        )
        for i in range(n)
    ]


class TestAuditPlan:
    def test_cell_count(self):
        bundle = _bundle(11)
        val = [f"v{i}" for i in range(30)]
        plan = audit_plan(bundle, val)
        assert len(plan) == (11 + 1) * 30

    def test_arms(self):
        plan = audit_plan(_bundle(2), ["a", "b"])
        keys = {s.arm_key for s in plan}
        assert keys == {"empty", "item_0", "item_1"}


class TestSummarize:
    def test_math_and_p1(self):
        # empty mean 0.5; item_0 → 0.7 (u=+0.2); item_1 → 0.4 (u=-0.1); item_2 → 0.5 (u=0)
        results = []
        for q in ("q1", "q2"):
            results.append({"arm_key": "empty", "question_id": q, "pass": 0.5})
            results.append({"arm_key": "item_0", "question_id": q, "pass": 0.7})
            results.append({"arm_key": "item_1", "question_id": q, "pass": 0.4})
            results.append({"arm_key": "item_2", "question_id": q, "pass": 0.5})
        s = summarize(results)
        assert abs(s["per_item"]["item_0"] - 0.2) < 1e-9
        assert abs(s["per_item"]["item_1"] - (-0.1)) < 1e-9
        assert abs(s["per_item"]["item_2"]) < 1e-9
        assert s["n_pos"] == 1
        assert s["n_neg"] == 2  # u<=0
        assert s["p1_verdict"] == "P1 CONFIRMED"  # 2/3 >= 0.30

    def test_p1_not_confirmed(self):
        results = []
        for q in ("q1",):
            results.append({"arm_key": "empty", "question_id": q, "pass": 0.0})
            results.append({"arm_key": "item_0", "question_id": q, "pass": 1.0})
            results.append({"arm_key": "item_1", "question_id": q, "pass": 1.0})
            results.append({"arm_key": "item_2", "question_id": q, "pass": 1.0})
            results.append({"arm_key": "item_3", "question_id": q, "pass": 0.0})  # u=0
        s = summarize(results)
        # 1 of 4 with u<=0 → 25% < 30%
        assert s["frac_u_le_0"] == 0.25
        assert s["p1_verdict"] == "P1 NOT CONFIRMED"


class TestResume:
    def test_pending_skips_done(self, tmp_path):
        plan = audit_plan(_bundle(2), ["a", "b", "c"])
        path = tmp_path / "out.jsonl"
        # Complete empty/a and item_0/b
        path.write_text(
            json.dumps({"arm_key": "empty", "question_id": "a", "pass": 1.0})
            + "\n"
            + json.dumps({"arm_key": "item_0", "question_id": "b", "pass": 0.0})
            + "\n"
        )
        done = load_done_cells(path)
        todo = pending_specs(plan, done)
        assert ("empty", "a") not in {(s.arm_key, s.question_id) for s in todo}
        assert ("item_0", "b") not in {(s.arm_key, s.question_id) for s in todo}
        assert len(todo) == len(plan) - 2

    def test_items_filter(self):
        plan = audit_plan(_bundle(5), ["a"])
        r = parse_items_range("1-2", 5)
        filtered = filter_plan_for_items(plan, r)
        keys = {s.arm_key for s in filtered}
        assert keys == {"empty", "item_1", "item_2"}
