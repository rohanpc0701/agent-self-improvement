#!/usr/bin/env python3
"""Freeze category-stratified FinancePro-Bench splits (RSI-Mem v2 G0.1).

200 train-stream / 80 validation / 120 held-out, seed 42.
Categories with <3 questions go to train only (logged).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATASET_PATH = ROOT / "fixtures" / "finance_pro_bench.json"
MANIFEST_PATH = ROOT / "fixtures" / "finance_manifest.json"


def load_items(path: Path = DATASET_PATH) -> list[dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "items" in raw:
        return list(raw["items"])
    if isinstance(raw, list):
        return raw
    raise ValueError(f"unexpected dataset shape in {path}")


def dataset_sha256(path: Path = DATASET_PATH) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stratified_split(
    items: list[dict],
    n_train: int = 200,
    n_validation: int = 80,
    n_heldout: int = 120,
    seed: int = 42,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Return (train_ids, validation_ids, heldout_ids, tiny_category_names)."""
    if n_train + n_validation + n_heldout != len(items):
        raise ValueError(
            f"split sizes {n_train}+{n_validation}+{n_heldout} != n_items {len(items)}"
        )
    rng = random.Random(seed)
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for it in items:
        by_cat[it["category"]].append(it)
    for qs in by_cat.values():
        rng.shuffle(qs)

    train: list[dict] = []
    validation: list[dict] = []
    heldout: list[dict] = []
    tiny: list[str] = []
    remain_pools: list[list[dict]] = []

    for cat, pool in sorted(by_cat.items(), key=lambda kv: kv[0]):
        if len(pool) < 3:
            tiny.append(cat)
            train.extend(pool)
            continue
        # Guarantee train + held-out presence when size permits.
        train.append(pool.pop())
        heldout.append(pool.pop())
        if pool:
            remain_pools.append(pool)

    # Round-robin fill held-out → validation → train from remainders.
    def _rr_take(target: list[dict], need: int) -> None:
        while len(target) < need and any(remain_pools):
            for pool in list(remain_pools):
                if len(target) >= need:
                    break
                if pool:
                    target.append(pool.pop(0))
            remain_pools[:] = [p for p in remain_pools if p]

    _rr_take(heldout, n_heldout)
    _rr_take(validation, n_validation)
    for pool in remain_pools:
        train.extend(pool)
    remain_pools.clear()

    # If quotas overshot (tiny cats), rebalance by moving from train into
    # underfilled pools without emptying a category's train presence.
    def _ids(xs: list[dict]) -> set[str]:
        return {x["id"] for x in xs}

    # Trim heldout/validation if somehow over (shouldn't happen often).
    while len(heldout) > n_heldout:
        train.append(heldout.pop())
    while len(validation) > n_validation:
        train.append(validation.pop())

    # Top up heldout/validation from train (prefer cats that still keep ≥1 train).
    train_by_cat: dict[str, list[dict]] = defaultdict(list)
    for it in train:
        train_by_cat[it["category"]].append(it)

    def _donate(to: list[dict], need: int) -> None:
        cats = sorted(train_by_cat.keys())
        while len(to) < need:
            progressed = False
            for cat in cats:
                if len(to) >= need:
                    break
                bucket = train_by_cat[cat]
                if len(bucket) <= 1:
                    continue
                item = bucket.pop()
                train.remove(item)
                to.append(item)
                progressed = True
            if not progressed:
                raise ValueError(
                    f"cannot reach quota need={need} have={len(to)}; "
                    f"train={len(train)} tiny={tiny}"
                )

    _donate(heldout, n_heldout)
    _donate(validation, n_validation)

    if len(train) != n_train or len(validation) != n_validation or len(heldout) != n_heldout:
        raise ValueError(
            f"final sizes train={len(train)} val={len(validation)} "
            f"heldout={len(heldout)} (want {n_train}/{n_validation}/{n_heldout})"
        )

    t_ids = [x["id"] for x in train]
    v_ids = [x["id"] for x in validation]
    h_ids = [x["id"] for x in heldout]
    assert not (set(t_ids) & set(v_ids) | set(t_ids) & set(h_ids) | set(v_ids) & set(h_ids))
    return t_ids, v_ids, h_ids, tiny


def build_manifest(
    dataset_path: Path = DATASET_PATH,
    seed: int = 42,
) -> dict:
    items = load_items(dataset_path)
    train_ids, val_ids, held_ids, tiny = stratified_split(items, seed=seed)
    if tiny:
        print(f"[freeze] tiny categories (<3) → train only: {tiny}")
    return {
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "dataset_sha256": dataset_sha256(dataset_path),
        "n_total": len(items),
        "train_ids": train_ids,
        "validation_ids": val_ids,
        "heldout_ids": held_ids,
        "tiny_categories": tiny,
    }


def check_manifest(
    manifest_path: Path = MANIFEST_PATH, dataset_path: Path = DATASET_PATH
) -> None:
    m = json.loads(manifest_path.read_text())
    sha = dataset_sha256(dataset_path)
    if m["dataset_sha256"] != sha:
        raise SystemExit(
            f"dataset sha mismatch: manifest={m['dataset_sha256'][:12]}… "
            f"file={sha[:12]}…"
        )
    t, v, h = set(m["train_ids"]), set(m["validation_ids"]), set(m["heldout_ids"])
    if t & v or t & h or v & h:
        raise SystemExit("disjointness violated")
    if len(t) != 200 or len(v) != 80 or len(h) != 120:
        raise SystemExit(f"size mismatch train={len(t)} val={len(v)} heldout={len(h)}")
    items = {x["id"] for x in load_items(dataset_path)}
    missing = (t | v | h) - items
    if missing:
        raise SystemExit(f"manifest ids missing from dataset: {list(missing)[:5]}")
    print(
        f"[check] ok n={m['n_total']} train={len(t)} validation={len(v)} "
        f"heldout={len(h)} seed={m['seed']}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--out", type=Path, default=MANIFEST_PATH)
    ap.add_argument("--dataset", type=Path, default=DATASET_PATH)
    args = ap.parse_args()
    if args.check:
        check_manifest(args.out, args.dataset)
        return
    m = build_manifest(args.dataset, seed=args.seed)
    args.out.write_text(json.dumps(m, indent=2) + "\n")
    print(
        f"[freeze] wrote {args.out} train={len(m['train_ids'])} "
        f"validation={len(m['validation_ids'])} heldout={len(m['heldout_ids'])}"
    )


if __name__ == "__main__":
    main()
