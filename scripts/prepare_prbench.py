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


def _n_prompts(row: dict) -> int:
    return sum(1 for k in row if k.startswith("prompt_") and row.get(k))


def _conversation(row: dict) -> tuple[list[dict], str]:
    """Build [{role,content}...] history + the final prompt the student answers.

    single-turn → [user p0], final = p0
    multi-turn  → [user p0, assistant r0, user p1, assistant r1, ..., user p_last]
                  (prior assistant responses are the dataset's baseline model =
                  fixed conversation context). final = last prompt.
    """
    turns: list[dict] = []
    i = 0
    final = ""
    while (p := row.get(f"prompt_{i}")):
        p = str(p).strip()
        turns.append({"role": "user", "content": p})
        final = p
        r = row.get(f"response_{i}")
        if r and str(r).strip():
            turns.append({"role": "assistant", "content": str(r).strip()})
        i += 1
    return turns, final


def main() -> None:
    from datasets import load_dataset

    ds = load_dataset("ScaleAI/PRBench", split="finance")
    tasks = []
    for row in ds:
        if str(row.get("topic")) != "Corporate Finance":
            continue
        crits = [c for c in (_criterion(x) for x in (row["rubric"] or [])) if c]
        turns, final = _conversation(row)
        if not crits or not final:
            continue
        tasks.append({
            "id": str(row.get("task")),
            "topic": "Corporate Finance",
            "turns": turns,
            "final_prompt": final,
            # `question` kept for single-turn compatibility (= final prompt).
            "question": final,
            "n_turns": sum(1 for t in turns if t["role"] == "user"),
            "rubric": crits,
        })
    OUT.write_text(json.dumps(
        {"_source": "ScaleAI/PRBench finance / topic=Corporate Finance / single+multi-turn",
         "n": len(tasks), "items": tasks}, indent=1))
    from collections import Counter
    ncrit = [len(t["rubric"]) for t in tasks]
    turn_dist = Counter(t["n_turns"] for t in tasks)
    print(f"wrote {len(tasks)} Corporate Finance tasks -> {OUT}")
    print(f"turn distribution: {dict(sorted(turn_dist.items()))}")
    print(f"criteria/task: min={min(ncrit)} max={max(ncrit)} mean={sum(ncrit)/len(ncrit):.1f}")


if __name__ == "__main__":
    main()
