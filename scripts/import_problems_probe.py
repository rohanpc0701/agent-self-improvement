#!/usr/bin/env python3
"""Difficulty probe: live 3B student, k samples @ temp>0; keep pass-rate <= 0.5.

Probing at temperature 0.7 (not 0.0) keeps the temp-0 eval baseline honest:
a problem filtered on a deterministic temp-0 failure would make the WITHOUT
arm 0.000 by construction.
"""
from __future__ import annotations

import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.import_problems import CANDIDATES_PATH, FIXTURE_PATH  # noqa: E402


def select_hard(results: list[tuple[dict, float]], max_keep: int) -> list[dict]:
    """Keep candidates with probe pass-rate <= 0.5, topic-balanced up to max_keep."""
    hard = [(c, r) for c, r in results if r <= 0.5]
    by_topic: dict[str, list[dict]] = defaultdict(list)
    for c, _ in hard:
        by_topic[c["topic"]].append(c)

    kept: list[dict] = []
    # round-robin across topics so scarce topics survive the cap
    pools = sorted(by_topic.items(), key=lambda kv: len(kv[1]))
    while len(kept) < max_keep and any(pool for _, pool in pools):
        for _, pool in pools:
            if pool and len(kept) < max_keep:
                kept.append(pool.pop(0))
    return kept


def probe(k: int = 2, temperature: float = 0.7, max_keep: int = 40) -> None:
    from adapters.coding import generate_code, verify_solution
    from orchestrator import _make_base_config

    candidates = json.loads(CANDIDATES_PATH.read_text())
    config = _make_base_config("difficulty-probe")  # empty few-shots
    results: list[tuple[dict, float]] = []

    for i, p in enumerate(candidates, 1):
        passes = 0
        for _ in range(k):
            text, *_rest = generate_code(
                p["question"],
                config,
                topic=p["topic"],
                use_rules=False,
                temperature=temperature,
            )
            acc, _, _ = verify_solution(text, p)
            passes += int(acc == 1.0)
        rate = passes / k
        results.append((p, rate))
        print(f"  [{i}/{len(candidates)}] {p['id']} pass-rate={rate:.1f}", flush=True)

    kept = select_hard(results, max_keep=max_keep)

    shutil.copy(FIXTURE_PATH, FIXTURE_PATH.with_suffix(".backup.json"))
    fixture = json.loads(FIXTURE_PATH.read_text())
    existing = {p["id"] for p in fixture}
    new = [c for c in kept if c["id"] not in existing]
    fixture.extend(new)
    FIXTURE_PATH.write_text(json.dumps(fixture, indent=1))

    n_hard = sum(1 for p in fixture if p["difficulty"] in ("hard", "extra"))
    print(f"\nappended {len(new)} hard problems → {FIXTURE_PATH}")
    print(f"hard pool now: {n_hard}")
    print("new topics:", dict(Counter(c["topic"] for c in new)))
