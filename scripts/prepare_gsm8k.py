#!/usr/bin/env python3
"""Expand fixtures/gsm8k_subset.json from HuggingFace openai/gsm8k.

Keeps existing IDs; appends new easy/hard stratified items up to target counts.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "fixtures" / "gsm8k_subset.json"

_TOPIC_KW = [
    ("finance", ["dollar", "cost", "$", "price", "profit", "earn", "pay", "salary", "budget"]),
    ("geometry", ["mile", "meter", "distance", "area", "perimeter", "triangle", "square"]),
    ("algebra", ["twice", "half as", "times as", "variable", "equation", "remaining"]),
    ("logic", ["either", "neither", "only if", "unless"]),
    ("arithmetic", ["how many", "total", "altogether", "sum", "add", "subtract"]),
]


def _label_topic(q: str) -> str:
    low = q.lower()
    for topic, kws in _TOPIC_KW:
        if any(k in low for k in kws):
            return topic
    return "arithmetic"


def _parse_answer(raw: str) -> str:
    # GSM8K solutions end with #### <answer>
    m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", raw)
    if not m:
        raise ValueError(f"no #### answer in: {raw[:80]!r}")
    return m.group(1).replace(",", "")


def _difficulty(question: str, answer: str) -> str:
    """Cheap heuristic: longer multi-step / larger numbers → hard."""
    n_sents = max(1, question.count(".") + question.count("?"))
    try:
        mag = abs(float(answer))
    except ValueError:
        mag = 0.0
    if n_sents >= 4 or mag >= 1000 or "percent" in question.lower() or "%" in question:
        return "hard"
    if n_sents >= 3 or mag >= 100:
        return "hard"
    return "easy"


def prepare(n_easy: int = 100, n_hard: int = 150, seed: int = 42) -> None:
    from datasets import load_dataset

    existing = json.loads(FIXTURE.read_text()) if FIXTURE.exists() else []
    by_q = {p["question"].strip(): p for p in existing}
    kept = list(existing)

    ds = load_dataset("openai/gsm8k", "main", split="train")
    rows = list(ds)
    rng_idx = list(range(len(rows)))
    import random

    random.Random(seed).shuffle(rng_idx)

    n_easy_have = sum(1 for p in kept if p["difficulty"] == "easy")
    n_hard_have = sum(1 for p in kept if p["difficulty"] in ("hard", "extra"))
    easy_i = hard_i = 0

    for i in rng_idx:
        if n_easy_have >= n_easy and n_hard_have >= n_hard:
            break
        row = rows[i]
        q = row["question"].strip()
        if q in by_q:
            continue
        try:
            ans = _parse_answer(row["answer"])
        except ValueError:
            continue
        diff = _difficulty(q, ans)
        if diff == "easy" and n_easy_have >= n_easy:
            continue
        if diff == "hard" and n_hard_have >= n_hard:
            continue
        if diff == "easy":
            easy_i += 1
            pid = f"gsm8k_hf_e{easy_i:03d}"
            n_easy_have += 1
        else:
            hard_i += 1
            pid = f"gsm8k_hf_h{hard_i:03d}"
            n_hard_have += 1
        item = {
            "id": pid,
            "question": q,
            "answer": ans,
            "topic": _label_topic(q),
            "difficulty": diff,
        }
        kept.append(item)
        by_q[q] = item

    FIXTURE.write_text(json.dumps(kept, indent=1) + "\n")
    from collections import Counter

    c = Counter(p["difficulty"] for p in kept)
    print(f"wrote {len(kept)} problems → {FIXTURE}")
    print("difficulty:", dict(c))
    print("topics:", dict(Counter(p["topic"] for p in kept)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-easy", type=int, default=100)
    ap.add_argument("--n-hard", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    prepare(n_easy=args.n_easy, n_hard=args.n_hard, seed=args.seed)


if __name__ == "__main__":
    main()
