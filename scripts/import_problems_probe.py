#!/usr/bin/env python3
"""Difficulty probe: live student, k samples @ temp>0; keep pass-rate <= 0.5.

Supports chunked resume via runs/probe_results.jsonl (--offset/--limit).
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

PROBE_RESULTS_PATH = ROOT / "runs" / "probe_results.jsonl"


def select_hard(results: list[tuple[dict, float]], max_keep: int) -> list[dict]:
    """Keep candidates with probe pass-rate <= 0.5, topic-balanced up to max_keep."""
    hard = [(c, r) for c, r in results if r <= 0.5]
    by_topic: dict[str, list[dict]] = defaultdict(list)
    for c, _ in hard:
        by_topic[c["topic"]].append(c)

    kept: list[dict] = []
    pools = sorted(by_topic.items(), key=lambda kv: len(kv[1]))
    while len(kept) < max_keep and any(pool for _, pool in pools):
        for _, pool in pools:
            if pool and len(kept) < max_keep:
                kept.append(pool.pop(0))
    return kept


def _load_done_ids(path: Path) -> dict[str, float]:
    done: dict[str, float] = {}
    if not path.exists():
        return done
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        done[row["id"]] = float(row["pass_rate"])
    return done


def probe(
    k: int = 2,
    temperature: float = 0.7,
    max_keep: int = 400,
    offset: int = 0,
    limit: int | None = None,
    results_path: Path = PROBE_RESULTS_PATH,
    merge: bool = True,
) -> None:
    from adapters.coding import generate_code, verify_solution
    from orchestrator import _make_base_config

    candidates = json.loads(CANDIDATES_PATH.read_text())
    topic_rank = {
        "dp": 0,
        "graphs": 1,
        "greedy": 2,
        "arithmetic": 3,
        "strings": 4,
        "arrays": 5,
    }
    candidates = sorted(
        candidates, key=lambda p: (topic_rank.get(p["topic"], 9), p["id"])
    )
    end = len(candidates) if limit is None else min(len(candidates), offset + limit)
    chunk = candidates[offset:end]
    print(
        f"[probe] candidates={len(candidates)} chunk=[{offset}:{end}] "
        f"({len(chunk)} to probe) max_keep={max_keep}",
        flush=True,
    )

    results_path.parent.mkdir(parents=True, exist_ok=True)
    done = _load_done_ids(results_path)
    config = _make_base_config("difficulty-probe")
    by_id = {p["id"]: p for p in candidates}

    with results_path.open("a", encoding="utf-8") as out:
        for i, p in enumerate(chunk, 1):
            if p["id"] in done:
                print(
                    f"  [{i}/{len(chunk)}] {p['id']} skip (resume) "
                    f"pass-rate={done[p['id']]:.1f}",
                    flush=True,
                )
                continue
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
            done[p["id"]] = rate
            out.write(json.dumps({"id": p["id"], "pass_rate": rate, "topic": p["topic"]}) + "\n")
            out.flush()
            print(f"  [{i}/{len(chunk)}] {p['id']} pass-rate={rate:.1f}", flush=True)

    if not merge:
        print(f"[probe] chunk done; results → {results_path} (merge skipped)")
        return

    # Merge all recorded rates (full candidate set) into fixture.
    results: list[tuple[dict, float]] = []
    for pid, rate in done.items():
        if pid in by_id:
            results.append((by_id[pid], rate))
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
