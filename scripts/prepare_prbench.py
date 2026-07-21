#!/usr/bin/env python3
"""Load PRBench (ScaleAI/PRBench) finance split, keep single-turn Corporate
Finance tasks, cache to fixtures/prbench_corpfin.json.

Each cached task: {id, topic, question (prompt_0), rubric: [criteria]} where each
criterion is {description, weight_class, weight}. Weight sign convention: positive
classes reward, *detrimental classes penalize (stored as negative).
Source: ScaleAI/PRBench (finance split), CC-BY. Rubrics are Scale's own.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "fixtures" / "prbench_corpfin.json"

_WEIGHT_FIELDS = {
    "critically_important_weight": 1,
    "important_weight": 1,
    "slightly_important_weight": 1,
    "slightly_detrimental_weight": -1,
    "detrimental_weight": -1,
    "critically_detrimental_weight": -1,
}


def _criterion(raw: dict) -> dict | None:
    a = raw.get("annotations", raw)
    desc = (a.get("criteria_description") or "").strip()
    if not desc:
        return None
    weight = 0.0
    for field, sign in _WEIGHT_FIELDS.items():
        v = a.get(field)
        if v is not None:
            weight = sign * abs(float(v))
            break
    return {
        "description": desc,
        "weight_class": a.get("weight_class", ""),
        "weight": weight,
        "category": a.get("criteria_category", ""),
    }


def _turns(row: dict) -> int:
    return sum(1 for k in row if k.startswith("prompt_") and row.get(k))


def main() -> None:
    from datasets import load_dataset

    ds = load_dataset("ScaleAI/PRBench", split="finance")
    tasks = []
    for row in ds:
        if str(row.get("topic")) != "Corporate Finance":
            continue
        if _turns(row) != 1:
            continue  # single-turn only (our loop is single-shot)
        crits = [c for c in (_criterion(x) for x in (row["rubric"] or [])) if c]
        if not crits or not (row.get("prompt_0") or "").strip():
            continue
        tasks.append({
            "id": str(row.get("task")),
            "topic": "Corporate Finance",
            "question": row["prompt_0"].strip(),
            "rubric": crits,
        })
    OUT.write_text(json.dumps(
        {"_source": "ScaleAI/PRBench finance / topic=Corporate Finance / single-turn",
         "n": len(tasks), "items": tasks}, indent=1))
    # summary
    from collections import Counter
    ncrit = [len(t["rubric"]) for t in tasks]
    wclass = Counter(c["weight_class"] for t in tasks for c in t["rubric"])
    print(f"wrote {len(tasks)} single-turn Corporate Finance tasks -> {OUT}")
    print(f"criteria/task: min={min(ncrit)} max={max(ncrit)} mean={sum(ncrit)/len(ncrit):.1f}")
    print("weight classes:", dict(wclass))
    print("max possible score (sum positive weights) sample:",
          [round(sum(c['weight'] for c in t['rubric'] if c['weight'] > 0), 1) for t in tasks[:5]])


if __name__ == "__main__":
    main()
