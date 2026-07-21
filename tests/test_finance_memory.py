"""Hermetic tests for finance teacher-repair + memory distillation (TraceLift Task A)."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from adapters.finance import (
    distill_memory_item,
    extract_named_entities,
    strip_named_entities,
    teacher_repair,
)
from contracts.schemas import FewShotExample


@pytest.fixture(scope="module")
def manifest():
    path = Path("fixtures/finance_manifest.json")
    if not path.exists():
        pytest.skip("finance_manifest.json missing")
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def dataset():
    path = Path("fixtures/finance_pro_bench.json")
    if not path.exists():
        pytest.skip("finance_pro_bench.json missing")
    raw = json.loads(path.read_text())
    return {x["id"]: x for x in raw["items"]}


class TestEntityAudit:
    def test_extract_and_strip_company_names(self):
        text = (
            "AeroDyne Components is preparing statements. "
            "The Precision Castings division was split."
        )
        ents = extract_named_entities(text)
        assert any("AeroDyne" in e for e in ents)
        cleaned = strip_named_entities(text, ents)
        assert "AeroDyne" not in cleaned
        assert "Precision Castings" not in cleaned
        assert "[ENTITY]" in cleaned

    def test_strip_is_idempotent_enough(self):
        text = "Acme Holdings LLC reported GAAP EPS."
        once = strip_named_entities(text)
        twice = strip_named_entities(once)
        assert "Acme Holdings" not in once
        assert twice == once or "[ENTITY]" in twice


class TestTeacherRepair:
    def test_blocked_on_heldout(self, manifest):
        held = manifest["heldout_ids"][0]
        with pytest.raises(PermissionError):
            teacher_repair(held, "student answer")

    def test_blocked_on_validation(self, manifest):
        val = manifest["validation_ids"][0]
        with pytest.raises(PermissionError):
            teacher_repair(val, "student answer")

    def test_calls_teacher_with_train_rubric_and_max_tokens(
        self, manifest, dataset, monkeypatch
    ):
        qid = manifest["train_ids"][0]
        p = dataset[qid]
        monkeypatch.setenv("TEACHER_MAX_TOKENS", "4000")

        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=MagicMock(content="REPAIRED ANSWER"))]
        mock_client.chat.completions.create.return_value = mock_resp

        with patch(
            "adapters.finance.teacher_client_and_model",
            return_value=(mock_client, "z-ai/glm-5.2"),
        ):
            out = teacher_repair(qid, "broken student text", client=None)

        assert out == "REPAIRED ANSWER"
        kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert kwargs["model"] == "z-ai/glm-5.2"
        assert kwargs["max_tokens"] == 4000
        user = kwargs["messages"][1]["content"]
        assert p["question"] in user
        assert p["rubric"] in user
        assert "broken student text" in user
        assert "OFFICIAL RUBRIC" in user


class TestDistillMemoryItem:
    def test_skeleton_source_and_domain(self, manifest, dataset):
        qid = manifest["train_ids"][0]
        p = dataset[qid]
        repaired = (
            "Issue: impairment trigger.\n"
            "Framework: ASC 360 recoverability.\n"
            "Steps: (1) identify asset group (2) undiscounted CF test "
            "(3) fair value measure (4) conclude write-down.\n"
            "Conclusion: recognize impairment of X.\n"
            + ("pad " * 200)
        )
        ex = distill_memory_item(qid, repaired, kind="skeleton")
        assert isinstance(ex, FewShotExample)
        assert ex.source == "tracelift"
        assert ex.domain_id == p["category"]
        assert ex.question.startswith("[FINANCE_SKELETON]")
        # ≤300 tokens rough: whitespace-split tokens
        assert len(ex.correct_output.split()) <= 300

    def test_playbook_and_trap_kinds(self, manifest):
        qid = manifest["train_ids"][1]
        pb = distill_memory_item(qid, "checklist: gate equity first", kind="playbook")
        tr = distill_memory_item(qid, "avoid fee safe-harbor without aggregation", kind="trap")
        assert pb.question.startswith("[FINANCE_PLAYBOOK]")
        assert tr.question.startswith("[FINANCE_TRAP]")
        assert pb.source == tr.source == "tracelift"

    def test_no_heldout_or_validation_question_leakage(self, manifest, dataset):
        qid = manifest["train_ids"][0]
        # Deliberately try to smuggle held-out question text into the repair.
        held_q = dataset[manifest["heldout_ids"][0]]["question"]
        val_q = dataset[manifest["validation_ids"][0]]["question"]
        smuggled = (
            "Generic framework checklist.\n"
            + held_q[:180]
            + "\n"
            + val_q[:180]
        )
        ex = distill_memory_item(qid, smuggled, kind="skeleton")
        blob = f"{ex.question}\n{ex.correct_output}"
        # Long unique stems from forbidden splits must not survive.
        assert held_q[:80] not in blob
        assert val_q[:80] not in blob

    def test_strips_entities_from_source_question(self, manifest, dataset):
        qid = manifest["train_ids"][0]
        p = dataset[qid]
        ents = extract_named_entities(p["question"])
        assert ents, "expected entities in finance questions"
        ex = distill_memory_item(qid, "Framework: apply recoverability test first.", kind="skeleton")
        blob = f"{ex.question}\n{ex.correct_output}"
        for e in ents[:5]:
            if len(e) >= 8:
                assert e not in blob, f"entity leaked: {e!r}"
