"""Tests for freeze_heldout + HELDOUT_MANIFEST feed wiring."""
from __future__ import annotations

import json

from harness.feed import build_stream
from scripts.freeze_heldout import (
    build_manifest,
    drop_near_duplicates,
    jaccard,
    topic_stratified_split,
)


def _prob(i: int, topic: str, q: str | None = None) -> dict:
    return {
        "id": f"h_{i}",
        "question": q or f"unique question number {i} about {topic}",
        "function_name": f"f{i}",
        "tests": [],
        "topic": topic,
        "difficulty": "hard",
        "gold_solution": "def f(): pass",
        "expected_sql": "def f(): pass",
        "db_id": topic,
    }


class TestJaccard:
    def test_identical_high(self):
        assert jaccard("sort a list of numbers", "sort a list of numbers") == 1.0

    def test_disjoint_low(self):
        assert jaccard("aaaa bbbb", "cccc dddd") == 0.0


class TestDropNearDupes:
    def test_drops_learn_copy(self):
        held = [_prob(1, "dp", "find the longest common subsequence of two strings")]
        learn = [
            _prob(2, "dp", "find the longest common subsequence of two strings"),
            _prob(3, "dp", "completely different knapsack capacity problem"),
        ]
        kept, dropped = drop_near_duplicates(learn, held, threshold=0.8)
        assert dropped == ["h_2"]
        assert [p["id"] for p in kept] == ["h_3"]


class TestSplit:
    def test_disjoint_and_sizes(self):
        hard = [_prob(i, "dp" if i < 100 else "arrays") for i in range(250)]
        h, l, v = topic_stratified_split(hard, n_heldout=200, n_validation=30, seed=42)
        assert len(h) == 200
        assert len(v) == 30
        assert len(l) == 20
        assert not (set(h) & set(l) | set(h) & set(v) | set(l) & set(v))

    def test_deterministic(self):
        hard = [_prob(i, "dp") for i in range(250)]
        a = topic_stratified_split(hard, 200, 30, seed=42)
        b = topic_stratified_split(hard, 200, 30, seed=42)
        assert a == b


class TestManifestFeed:
    def test_feed_respects_manifest(self, tmp_path, monkeypatch):
        hard = [_prob(i, "dp" if i % 2 == 0 else "arrays") for i in range(250)]
        easy = [
            {
                "id": f"e_{i}",
                "question": f"easy {i}",
                "expected_sql": "x",
                "db_id": "dp",
                "difficulty": "easy",
            }
            for i in range(40)
        ]
        fixture = easy + hard
        fix_path = tmp_path / "fix.json"
        fix_path.write_text(json.dumps(fixture))
        m = build_manifest(fix_path, n_heldout=200, n_validation=30, seed=42)
        man_path = tmp_path / "manifest.json"
        man_path.write_text(json.dumps(m))
        monkeypatch.setenv("HELDOUT_MANIFEST", str(man_path))

        qs = []
        for p in fixture:
            qs.append(
                {
                    "id": p["id"],
                    "question": p["question"],
                    "expected_sql": p.get("expected_sql", "x"),
                    "db_id": p.get("topic") or p.get("db_id", "dp"),
                    "difficulty": p["difficulty"],
                }
            )
        items = build_stream(
            qs, n_baseline=10, n_degraded=20, n_recovery=50, seed=1, same_db_split=True
        )
        rec_ids = {it.question_id for it in items if it.phase == "recovery"}
        assert rec_ids.issubset(set(m["heldout_ids"]))
        deg_ids = {it.question_id for it in items if it.phase == "degraded"}
        assert deg_ids.isdisjoint(set(m["heldout_ids"]))
