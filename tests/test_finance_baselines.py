"""Hermetic tests for finance headroom / baseline plumbing."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from analysis.bootstrap import mean_bootstrap

_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "finance_baselines", _ROOT / "scripts" / "finance_baselines.py"
)
assert _SPEC and _SPEC.loader
fb = importlib.util.module_from_spec(_SPEC)
sys.modules["finance_baselines"] = fb
_SPEC.loader.exec_module(fb)


class TestMeanBootstrap:
    def test_known_mean(self):
        xs = [10.0, 20.0, 30.0]
        out = mean_bootstrap(xs, n_boot=2000, seed=0)
        assert out["mean"] == 20.0
        assert out["ci_low"] <= 20.0 <= out["ci_high"]

    def test_deterministic(self):
        xs = [1.0, 2.0, 3.0, 4.0]
        assert mean_bootstrap(xs, n_boot=500, seed=3) == mean_bootstrap(
            xs, n_boot=500, seed=3
        )


class TestPickStudent:
    def test_smallest_in_band(self):
        means = {
            "qwen/qwen3-8b": 8.0,
            "qwen/qwen3.6-27b": 22.0,
            "qwen/qwen3-30b-a3b-instruct-2507": 35.0,
        }
        d = fb.pick_student(means)
        assert d["chosen"] == "qwen/qwen3.6-27b"

    def test_all_below_triggers_fallback(self):
        means = {m: 5.0 for m in fb.DEFAULT_CANDIDATES}
        d = fb.pick_student(means)
        assert d["chosen"] is None
        assert d["fallback"] == fb.FALLBACK_LARGER

    def test_8b_in_band_wins(self):
        means = {
            "qwen/qwen3-8b": 18.0,
            "qwen/qwen3.6-27b": 30.0,
        }
        assert fb.pick_student(means)["chosen"] == "qwen/qwen3-8b"


class TestResumePlumbing:
    def test_done_ids_skips_errors(self, tmp_path: Path):
        p = tmp_path / "a.jsonl"
        p.write_text(
            json.dumps({"id": "a", "answer": "", "error": "Timeout"})
            + "\n"
            + json.dumps({"id": "b", "answer": "ok"})
            + "\n"
        )
        assert fb._done_ids(p) == {"b"}

    def test_latest_ok_answers(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        p = tmp_path / "a.jsonl"
        p.write_text(
            json.dumps({"id": "x", "answer": "", "error": "e"})
            + "\n"
            + json.dumps({"id": "x", "answer": "final"})
            + "\n"
        )
        monkeypatch.setattr(fb, "_answers_path", lambda tag, model: p)
        # call through _latest_ok_answers directly
        got = fb._latest_ok_answers(p)
        assert got["x"]["answer"] == "final"


class TestGenerateRetry:
    def test_regenerates_on_failure(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(fb, "RUNS", tmp_path)
        monkeypatch.setattr(fb, "MAX_GEN_ATTEMPTS", 2)

        calls = {"n": 0}

        def fake_gen(question, cfg, category, **kwargs):
            calls["n"] += 1
            if calls["n"] < 2:
                raise TimeoutError("slow")
            return ("good answer", {})

        monkeypatch.setattr(fb, "generate_answer", fake_gen)
        monkeypatch.setattr(
            fb,
            "get_problem",
            lambda qid: {"question": "Q?", "category": "cat", "rubric": "R"},
        )
        path = fb.generate_for_ids(["q1"], "m", "headroom", resume=False)
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        assert any(r.get("answer") == "good answer" for r in rows)
        assert calls["n"] == 2


class TestGradeMocked:
    def test_grade_writes_traps(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(fb, "RUNS", tmp_path)
        ans = tmp_path / "finance_headroom_m_answers.jsonl"
        ans.write_text(
            json.dumps({"id": "q1", "category": "cat", "answer": "A", "model": "m"})
            + "\n"
        )

        def fake_grade(**kwargs):
            return {
                "normalized": 25.0,
                "total": 5.0,
                "max": 20.0,
                "traps_hit": ["T1"],
                "bonuses": [],
            }

        monkeypatch.setattr(fb, "grade", fake_grade)
        monkeypatch.setattr(
            fb,
            "get_problem",
            lambda qid: {"question": "Q?", "category": "cat", "rubric": "Item R1 (max 10)"},
        )
        monkeypatch.setattr(fb, "rubric_for", lambda qid, role="judge": "Item R1 (max 10)")
        path = fb.grade_for_ids(["q1"], "m", "headroom", resume=False, judge_passes=1)
        row = json.loads(path.read_text().splitlines()[0])
        assert row["normalized"] == 25.0
        assert row["traps_hit"] == ["T1"]
