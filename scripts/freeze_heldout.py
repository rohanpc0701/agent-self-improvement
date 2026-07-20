#!/usr/bin/env python3
"""Freeze a topic-stratified LEARN / held-out / validation split for RSI-Mem.

Writes fixtures/heldout_manifest.json. Honesty rule: held-out is never used for
memory selection, uplift gating, or iteration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FIXTURE_PATH = ROOT / "fixtures" / "coding_subset.json"
MANIFEST_PATH = ROOT / "fixtures" / "heldout_manifest.json"


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def drop_near_duplicates(
    learn: list[dict], heldout: list[dict], threshold: float = 0.8
) -> tuple[list[dict], list[str]]:
    """Drop LEARN items that are near-duplicates of any held-out question."""
    kept: list[dict] = []
    dropped: list[str] = []
    for p in learn:
        if any(jaccard(p["question"], h["question"]) > threshold for h in heldout):
            dropped.append(p["id"])
            continue
        kept.append(p)
    return kept, dropped


def topic_stratified_split(
    hard: list[dict],
    n_heldout: int,
    n_validation: int,
    seed: int,
) -> tuple[list[str], list[str], list[str]]:
    """Return (heldout_ids, learn_ids, validation_ids), all disjoint."""
    rng = random.Random(seed)
    by_topic: dict[str, list[dict]] = defaultdict(list)
    for p in hard:
        by_topic[p.get("topic") or "arrays"].append(p)
    for qs in by_topic.values():
        rng.shuffle(qs)

    # Round-robin fill held-out so scarce topics survive.
    pools = sorted(by_topic.items(), key=lambda kv: len(kv[1]))
    heldout: list[dict] = []
    while len(heldout) < n_heldout and any(pool for _, pool in pools):
        for _, pool in pools:
            if pool and len(heldout) < n_heldout:
                heldout.append(pool.pop(0))

    remaining: list[dict] = []
    for _, pool in pools:
        remaining.extend(pool)
    rng.shuffle(remaining)

    remaining, dropped = drop_near_duplicates(remaining, heldout)
    if dropped:
        print(f"[freeze] dropped {len(dropped)} LEARN near-dupes of held-out: {dropped[:10]}")

    if len(remaining) < n_validation:
        raise ValueError(
            f"Need ≥{n_validation} LEARN after held-out+dedupe; have {len(remaining)}"
        )
    validation = remaining[:n_validation]
    learn = remaining[n_validation:]
    return (
        [p["id"] for p in heldout],
        [p["id"] for p in learn],
        [p["id"] for p in validation],
    )


def fixture_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_manifest(
    fixture_path: Path = FIXTURE_PATH,
    n_heldout: int = 200,
    n_validation: int = 30,
    seed: int = 42,
) -> dict:
    fixture = json.loads(fixture_path.read_text())
    hard = [p for p in fixture if p.get("difficulty") in ("hard", "extra")]
    if len(hard) < n_heldout + n_validation:
        raise ValueError(
            f"Hard pool {len(hard)} < heldout({n_heldout})+val({n_validation}). "
            "Expand via import probe first."
        )
    held_ids, learn_ids, val_ids = topic_stratified_split(
        hard, n_heldout=n_heldout, n_validation=n_validation, seed=seed
    )
    assert not (set(held_ids) & set(learn_ids))
    assert not (set(held_ids) & set(val_ids))
    assert not (set(learn_ids) & set(val_ids))
    return {
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "fixture_sha256": fixture_sha256(fixture_path),
        "n_hard_total": len(hard),
        "heldout_ids": held_ids,
        "learn_ids": learn_ids,
        "validation_ids": val_ids,
    }


def check_manifest(manifest_path: Path = MANIFEST_PATH, fixture_path: Path = FIXTURE_PATH) -> None:
    m = json.loads(manifest_path.read_text())
    sha = fixture_sha256(fixture_path)
    if m["fixture_sha256"] != sha:
        raise SystemExit(
            f"fixture sha mismatch: manifest={m['fixture_sha256'][:12]}… "
            f"file={sha[:12]}…"
        )
    h, l, v = set(m["heldout_ids"]), set(m["learn_ids"]), set(m["validation_ids"])
    if h & l or h & v or l & v:
        raise SystemExit("disjointness violated")
    print(
        f"[check] ok hard={m['n_hard_total']} heldout={len(h)} "
        f"learn={len(l)} validation={len(v)}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-heldout", type=int, default=200)
    ap.add_argument("--n-validation", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--check", action="store_true", help="validate existing manifest")
    ap.add_argument("--out", type=Path, default=MANIFEST_PATH)
    args = ap.parse_args()
    if args.check:
        check_manifest(args.out)
        return
    m = build_manifest(
        n_heldout=args.n_heldout, n_validation=args.n_validation, seed=args.seed
    )
    args.out.write_text(json.dumps(m, indent=2) + "\n")
    print(
        f"[freeze] wrote {args.out} hard={m['n_hard_total']} "
        f"heldout={len(m['heldout_ids'])} learn={len(m['learn_ids'])} "
        f"validation={len(m['validation_ids'])}"
    )


if __name__ == "__main__":
    main()
