"""Hermetic tests for finance_eval A1/A4 summary helpers."""
from __future__ import annotations

import json
from pathlib import Path

import scripts.finance_eval as fe
from contracts.schemas import FewShotExample


def test_summarize_a1_a4(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(fe, "RUNS", tmp_path)
    student = "qwen/qwen3.6-27b"
    # Write paired grades
    for arm, scores in (
        ("a1", [10.0, 20.0, 30.0, 40.0]),
        ("a4", [14.0, 24.0, 34.0, 44.0]),
    ):
        path = tmp_path / f"finance_eval_{arm}_{fe._slug(student)}_grades.jsonl"
        with path.open("w") as f:
            for i, s in enumerate(scores):
                f.write(
                    json.dumps(
                        {
                            "id": f"q{i}",
                            "arm": arm,
                            "category": "Accounting",
                            "normalized": s,
                            "traps_hit": ["T1"] if arm == "a1" and i == 0 else [],
                        }
                    )
                    + "\n"
                )
    out = fe.summarize_a1_a4(student)
    assert out["n"] == 4
    assert abs(out["GAP_alone"] - 4.0) < 1e-9
    assert out["p_value"] < 0.05
    assert out["trap_hit_rate_a1"] == 0.25
    assert out["trap_hit_rate_a4"] == 0.0


def test_load_frozen_memory(tmp_path: Path):
    path = tmp_path / "mem.json"
    path.write_text(
        json.dumps(
            {
                "items": [
                    {
                        "question": "[FINANCE_PLAYBOOK] Accounting",
                        "correct_output": "gate first",
                        "domain_id": "Accounting",
                        "source": "tracelift",
                    }
                ]
            }
        )
    )
    items = fe.load_frozen_memory(path)
    assert len(items) == 1
    assert isinstance(items[0], FewShotExample)
    assert items[0].source == "tracelift"
