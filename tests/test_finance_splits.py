"""Tests for finance_freeze_splits."""
from __future__ import annotations

import json

from scripts.finance_freeze_splits import (
    build_manifest,
    check_manifest,
    stratified_split,
)


def _items(n_cats: int = 5, per: int = 10) -> list[dict]:
    items = []
    i = 0
    for c in range(n_cats):
        for _ in range(per):
            items.append(
                {
                    "id": f"q{i}",
                    "category": f"cat{c}",
                    "question": f"question {i}",
                    "rubric": f"rubric {i}",
                }
            )
            i += 1
    return items


class TestStratifiedSplit:
    def test_sizes_and_disjoint(self):
        items = _items(5, 8)  # 40
        # scale: use 20/8/12
        t, v, h, tiny = stratified_split(
            items, n_train=20, n_validation=8, n_heldout=12, seed=42
        )
        assert len(t) == 20 and len(v) == 8 and len(h) == 12
        assert not (set(t) & set(v) | set(t) & set(h) | set(v) & set(h))
        assert tiny == []

    def test_tiny_categories_train_only(self):
        items = _items(3, 10)  # 30
        items += [
            {"id": "tiny0", "category": "rare", "question": "a", "rubric": "r"},
            {"id": "tiny1", "category": "rare", "question": "b", "rubric": "r"},
        ]
        t, v, h, tiny = stratified_split(
            items, n_train=18, n_validation=6, n_heldout=8, seed=0
        )
        assert "rare" in tiny
        assert "tiny0" in t and "tiny1" in t
        assert "tiny0" not in h and "tiny1" not in v

    def test_deterministic(self):
        items = _items(4, 10)
        a = stratified_split(items, 20, 8, 12, seed=42)
        b = stratified_split(items, 20, 8, 12, seed=42)
        assert a == b


class TestManifestFile:
    def test_real_fixture_roundtrip(self, tmp_path):
        # Use committed fixture if present
        from pathlib import Path

        ds = Path("fixtures/finance_pro_bench.json")
        if not ds.exists():
            return
        m = build_manifest(ds, seed=42)
        out = tmp_path / "manifest.json"
        out.write_text(json.dumps(m))
        check_manifest(out, ds)
        assert len(m["train_ids"]) == 200
        assert len(m["validation_ids"]) == 80
        assert len(m["heldout_ids"]) == 120
