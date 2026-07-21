"""Tests for finance adapter + rubric firewall."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from adapters.finance import (
    FinanceAdapter,
    assert_rubric_allowed_for_teacher,
    build_student_prompt,
    build_teacher_prompt,
    load_finance_questions,
    load_manifest,
)
from contracts.schemas import AgentConfig, FewShotExample


@pytest.fixture(scope="module")
def manifest():
    path = Path("fixtures/finance_manifest.json")
    if not path.exists():
        pytest.skip("finance_manifest.json not frozen yet")
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def dataset():
    path = Path("fixtures/finance_pro_bench.json")
    if not path.exists():
        pytest.skip("finance_pro_bench.json missing")
    raw = json.loads(path.read_text())
    return {x["id"]: x for x in raw["items"]}


class TestLoad:
    def test_adapter_registered(self):
        from adapters import get_adapter

        assert get_adapter("finance").name == "finance"

    def test_load_splits(self, manifest):
        train = load_finance_questions("train")
        val = load_finance_questions("validation")
        held = load_finance_questions("heldout")
        assert len(train) == 200
        assert len(val) == 80
        assert len(held) == 120


class TestRubricFirewall:
    def test_student_prompt_never_contains_rubric(self, manifest, dataset):
        cfg = AgentConfig(config_id="t", model="m", few_shot_examples=[])
        for qid in manifest["heldout_ids"][:5] + manifest["validation_ids"][:5]:
            p = dataset[qid]
            prompt, _ = build_student_prompt(p["question"], cfg, p["category"])
            # Full rubric must not appear; also reject a long unique rubric stem.
            stem = p["rubric"].strip().split("\n", 1)[0][:120]
            assert p["rubric"] not in prompt
            if len(stem) > 40:
                assert stem not in prompt

    def test_teacher_blocked_on_heldout_and_validation(self, manifest, dataset):
        for qid in manifest["heldout_ids"][:3] + manifest["validation_ids"][:3]:
            p = dataset[qid]
            with pytest.raises(PermissionError):
                build_teacher_prompt(
                    p["question"], qid=qid, rubric=p["rubric"], manifest=manifest
                )
            with pytest.raises(PermissionError):
                assert_rubric_allowed_for_teacher(qid, manifest)

    def test_teacher_allowed_on_train(self, manifest, dataset):
        qid = manifest["train_ids"][0]
        p = dataset[qid]
        prompt = build_teacher_prompt(
            p["question"], qid=qid, rubric=p["rubric"], manifest=manifest
        )
        assert p["rubric"] in prompt
        assert p["question"] in prompt

    def test_teacher_rejects_mismatched_rubric_text(self, manifest, dataset):
        train_id = manifest["train_ids"][0]
        held_id = manifest["heldout_ids"][0]
        with pytest.raises(PermissionError, match="does not match"):
            build_teacher_prompt(
                dataset[train_id]["question"],
                qid=train_id,
                rubric=dataset[held_id]["rubric"],
                manifest=manifest,
            )

    def test_examples_do_not_smuggle_rubric(self, dataset, manifest):
        from adapters.finance import all_rubric_stems

        qid = manifest["heldout_ids"][0]
        p = dataset[qid]
        stems = all_rubric_stems()
        cfg = AgentConfig(
            config_id="t",
            model="m",
            few_shot_examples=[
                FewShotExample(
                    question="other q",
                    correct_output=p["rubric"][:200],
                    domain_id=p["category"],
                    source="teacher",
                )
            ],
        )
        with pytest.raises(PermissionError, match="rubric firewall"):
            build_student_prompt(
                p["question"], cfg, p["category"], forbidden_rubric_stems=stems
            )

    def test_load_questions_defaults_to_train(self, manifest):
        ad = FinanceAdapter()
        qs = ad.load_questions()
        assert len(qs) == 200
        assert {q["id"] for q in qs} == set(manifest["train_ids"])

    def test_rubric_for_acl(self, manifest):
        from adapters.finance import rubric_for

        held = manifest["heldout_ids"][0]
        with pytest.raises(PermissionError):
            rubric_for(held, role="student")
        with pytest.raises(PermissionError):
            rubric_for(held, role="teacher", manifest=manifest)
        assert rubric_for(held, role="judge")


class TestFeed:
    def test_build_feed_train_only(self, manifest):
        ad = FinanceAdapter()
        items = ad.build_feed(n=10, full=False, seed=42)
        assert len(items) == 10
        train = set(manifest["train_ids"])
        assert all(it.question_id in train for it in items)
