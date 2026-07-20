"""Hermetic tests for eval_runner seed plumbing + summary."""
from __future__ import annotations

import json

from scripts.eval_runner import (
    build_run_matrix,
    default_mock_bundle,
    load_done_cells,
    memory_bundle_for_seed,
    pending_cells,
    summarize_eval,
)


class TestMatrix:
    def test_shared_baselines_distinct_memory(self):
        held = ["h1", "h2"]
        seeds = [0, 1, 2]
        matrix = build_run_matrix(held, seeds)
        # 2 alone + 2 teacher + 2*3 memory = 2+2+6 = 10
        assert len(matrix) == 10
        alone = [c for c in matrix if c.arm == "student_alone"]
        assert len(alone) == 2 and all(c.seed is None for c in alone)
        mem = [c for c in matrix if c.arm == "student_memory"]
        assert {c.seed for c in mem} == {0, 1, 2}


class TestBundles:
    def test_distinct_per_seed(self):
        b0 = memory_bundle_for_seed(0, default_mock_bundle)
        b1 = memory_bundle_for_seed(1, default_mock_bundle)
        assert b0[0].question != b1[0].question


class TestSummary:
    def test_gap_math(self):
        alone = [0.0, 0.0, 1.0, 0.0]
        teacher = [1.0, 1.0, 1.0, 1.0]
        mem0 = [1.0, 0.0, 1.0, 0.0]  # mean 0.5 vs alone 0.25 → gap 0.25
        out = summarize_eval(
            {
                "student_alone": alone,
                "teacher_alone": teacher,
                "student_memory_0": mem0,
            },
            n_boot=500,
            seed=0,
        )
        assert abs(out["mean_gap"] - 0.25) < 1e-9
        assert out["n_seeds"] == 1
        assert out["per_seed"][0]["teacher_gap_closed"] is not None


class TestResume:
    def test_resume(self, tmp_path):
        matrix = build_run_matrix(["a", "b"], [0])
        path = tmp_path / "cells.jsonl"
        path.write_text(
            json.dumps(
                {"arm": "student_alone", "seed": None, "question_id": "a", "pass": 1.0}
            )
            + "\n"
        )
        done = load_done_cells(path)
        todo = pending_cells(matrix, done)
        assert len(todo) == len(matrix) - 1
