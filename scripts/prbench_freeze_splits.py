#!/usr/bin/env python3
"""Freeze a seeded train/validation/held-out split for PRBench Corporate Finance.

~59 single-turn tasks → default 30 train / 10 validation / rest held-out.
Held-out is touched once per evaluated arm; validation is for uplift gating only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "fixtures" / "prbench_corpfin.json"
MANIFEST = ROOT / "fixtures" / "prbench_corpfin_manifest.json"


def build(n_train: int, n_val: int, seed: int) -> dict:
    items = json.loads(DATASET.read_text())["items"]
    ids = [t["id"] for t in items]
    rng = random.Random(seed)
    rng.shuffle(ids)
    train = ids[:n_train]
    val = ids[n_train:n_train + n_val]
    held = ids[n_train + n_val:]
    sha = hashlib.sha256(DATASET.read_bytes()).hexdigest()
    return {
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "dataset_sha256": sha,
        "n_total": len(ids),
        "train_ids": train,
        "validation_ids": val,
        "heldout_ids": held,
    }


def check() -> None:
    m = json.loads(MANIFEST.read_text())
    t, v, h = set(m["train_ids"]), set(m["validation_ids"]), set(m["heldout_ids"])
    assert not (t & v or t & h or v & h), "splits overlap"
    sha = hashlib.sha256(DATASET.read_bytes()).hexdigest()
    assert m["dataset_sha256"] == sha, "dataset changed since freeze"
    print(f"[check] ok train={len(t)} validation={len(v)} heldout={len(h)} seed={m['seed']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=30)
    ap.add_argument("--n-val", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.check:
        check()
        return
    m = build(args.n_train, args.n_val, args.seed)
    MANIFEST.write_text(json.dumps(m, indent=1))
    print(f"froze {m['n_total']} tasks: train={len(m['train_ids'])} "
          f"validation={len(m['validation_ids'])} heldout={len(m['heldout_ids'])} "
          f"seed={m['seed']} -> {MANIFEST}")


if __name__ == "__main__":
    main()
