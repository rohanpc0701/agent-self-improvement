#!/usr/bin/env python3
"""BUILD contrastive frozen memory on PRBench Corp-Finance TRAIN tasks.

Per train task: Fable answers (A1), DeepSeek answers → GPT-5.2 grades → mistakes (A2),
Fable(A1, A2, mistakes) → ONE transferable lesson. Collect the lowest-scoring tasks'
lessons (where the student struggled most). Freeze to runs/prbench_memory.json.
Ungated (no uplift filter). Resumable via runs/prbench_memory_build.jsonl.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adapters import prbench as pr  # noqa: E402
from contracts.schemas import AgentConfig  # noqa: E402
from harness import agent  # noqa: E402

BUILD = ROOT / "runs" / "prbench_memory_build.jsonl"
MEMORY = ROOT / "runs" / "prbench_memory.json"


def _done() -> dict:
    d = {}
    if BUILD.exists():
        for line in BUILD.read_text().splitlines():
            r = json.loads(line)
            d[r["tid"]] = r
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--student-model", default="deepseek/deepseek-v4-pro")
    ap.add_argument("--n-train", type=int, default=12, help="train tasks to process")
    ap.add_argument("--max-items", type=int, default=10, help="lessons to freeze (lowest student scores)")
    ap.add_argument("--kind", default="playbook", choices=["playbook", "trap", "skeleton"])
    args = ap.parse_args()

    man = pr.load_manifest()
    train = man["train_ids"][: args.n_train]
    done = _done()
    BUILD.parent.mkdir(exist_ok=True)

    for i, tid in enumerate(train, 1):
        if tid in done:
            continue
        task = pr.get_task(tid)
        cfg = AgentConfig(config_id="build", model=args.student_model, few_shot_examples=[])
        try:
            agent._client = None
            teacher_ans = pr.answer_teacher_alone(task)               # A1
            student_ans, _ = pr.generate_answer(task, cfg)            # A2
            g = pr.score_answer(tid, student_ans)                     # grade → mistakes
            item = pr.build_memory_item(task, student_ans, teacher_ans, g["missed"], kind=args.kind)
            rec = {"tid": tid, "student_score": g["normalized"],
                   "n_missed": len(g["missed"]),
                   "lesson": item.correct_output, "lesson_chars": len(item.correct_output)}
            with open(BUILD, "a") as f:
                f.write(json.dumps(rec) + "\n")
            done[tid] = rec
            print(f"[{i}/{len(train)}] {tid[:8]} student={g['normalized']:.0f} "
                  f"missed={len(g['missed'])} lesson={len(item.correct_output)}ch", flush=True)
        except Exception as e:
            print(f"[{i}/{len(train)}] {tid[:8]} FAIL {str(e)[:70]}", flush=True)

    # freeze the lowest-scoring tasks' lessons (student struggled most = most to learn)
    recs = sorted(done.values(), key=lambda r: r["student_score"])[: args.max_items]
    items = [{"question": f"[FINANCE_PLAYBOOK] {pr._TOPIC}", "correct_output": r["lesson"],
              "domain_id": pr._TOPIC, "source": "tracelift"} for r in recs]
    MEMORY.write_text(json.dumps({"version": 1, "n": len(items), "items": items,
                                  "meta": {"gated": False, "kind": args.kind,
                                           "built_from": [r["tid"] for r in recs]}}, indent=1))
    print(f"\nFROZE {len(items)} lessons → {MEMORY}")


if __name__ == "__main__":
    main()
