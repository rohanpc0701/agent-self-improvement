#!/usr/bin/env python3
"""Build fixtures/demo_continuous.jsonl — two drift/correction cycles for the viewer.

Extends the recorded demo_events.jsonl with a second degraded trough, drift,
correction, and recovery block (no API calls). Run from repo root:

  python3 scripts/build_demo_continuous.py
"""
from __future__ import annotations

import copy
import json
import sys
import time
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from contracts.eventlog import Event, read_events
from contracts.schemas import CorrectionAction, DriftEvent, TelemetryRecord

SRC = REPO / "fixtures" / "demo_events.jsonl"
OUT = REPO / "fixtures" / "demo_continuous.jsonl"


def _write_event(f, etype: str, data: dict) -> None:
    env = Event(type=etype, ts=time.time(), data=data)
    f.write(env.model_dump_json() + "\n")


def main() -> None:
    events = read_events(SRC)
    if not events:
        print(f"No events in {SRC}", file=sys.stderr)
        sys.exit(1)

    first_corr: CorrectionAction | None = None
    first_drift: DriftEvent | None = None
    recovery_telemetry: list[TelemetryRecord] = []
    saw_correction = False

    for ev in events:
        if isinstance(ev, CorrectionAction):
            first_corr = ev
            saw_correction = True
        elif isinstance(ev, DriftEvent) and first_drift is None:
            first_drift = ev
        elif isinstance(ev, TelemetryRecord):
            if saw_correction:
                recovery_telemetry.append(ev)

    if not first_corr or not first_drift or len(recovery_telemetry) < 20:
        print("demo_events.jsonl missing correction/recovery block", file=sys.stderr)
        sys.exit(1)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(SRC) as fin, open(OUT, "w") as fout:
        for line in fin:
            fout.write(line)

        # Second cycle: clone recovery telemetry with a mid-run accuracy dip then climb
        n = min(45, len(recovery_telemetry))
        block = recovery_telemetry[:n]
        t0 = time.time()
        for i, rec in enumerate(block):
            dup = copy.deepcopy(rec)
            dup.run_id = f"{rec.run_id}_c2_{uuid.uuid4().hex[:6]}"
            dup.timestamp = t0 + i * 0.5
            if i < n // 3:
                dup.execution_accuracy = 0.0
                dup.query_valid = i % 4 != 0
            elif i < 2 * n // 3:
                dup.execution_accuracy = 0.0
            else:
                dup.execution_accuracy = 1.0 if i % 2 == 0 else 0.0
            _write_event(fout, "telemetry", dup.model_dump(mode="json"))

        drift2 = first_drift.model_copy(
            update={
                "detected_at": time.time(),
                "severity": round(first_drift.severity + 0.02, 3),
                "window_mean": 0.15,
            }
        )
        _write_event(fout, "drift", drift2.model_dump(mode="json"))

        subset = first_corr.new_few_shot_examples[: min(8, len(first_corr.new_few_shot_examples))]
        corr2 = CorrectionAction(
            triggered_by=drift2.channel,
            new_few_shot_examples=subset,
            rationale=(
                "Second continuous cycle: additional same-topic examples after "
                f"re-drift (severity={drift2.severity:.3f})."
            ),
        )
        _write_event(fout, "correction", corr2.model_dump(mode="json"))

        for i, rec in enumerate(block):
            dup = copy.deepcopy(rec)
            dup.run_id = f"{rec.run_id}_rec2_{uuid.uuid4().hex[:6]}"
            dup.timestamp = t0 + n + i * 0.5
            dup.execution_accuracy = 1.0 if i % 3 != 0 else 0.0
            _write_event(fout, "telemetry", dup.model_dump(mode="json"))

    total_lines = sum(1 for _ in open(OUT))
    print(f"Wrote {OUT} ({total_lines} lines, 2 drift/correction cycles)")


if __name__ == "__main__":
    main()
