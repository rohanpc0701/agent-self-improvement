#!/usr/bin/env bash
# Staged TraceLift LIVE loop (OpenRouter). Resumeable; stops on gate stop-rule.
# Train until TARGET_FAILURES, then candidates, then uplift gate (val-n/k cost-aware).
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=runs/finance_tracelift_live.log
TARGET_FAILURES="${TARGET_FAILURES:-20}"
VAL_N="${FINANCE_UPLIFT_VAL_N:-4}"
K="${FINANCE_UPLIFT_K:-2}"
CHUNK_S="${CHUNK_S:-540}"

echo "=== LOOP_START $(date -u +%Y-%m-%dT%H:%M:%SZ) pid=$$ target_fail=$TARGET_FAILURES val_n=$VAL_N k=$K ===" | tee -a "$LOG"

fail_count() {
  python3 - <<'PY'
import json
from pathlib import Path
rows=[json.loads(l) for l in Path("runs/finance_tracelift_state.jsonl").read_text().splitlines() if l.strip()] if Path("runs/finance_tracelift_state.jsonl").exists() else []
fails={r["key"] for r in rows if r.get("kind")=="train_grade" and r.get("ok") and r.get("failure")}
print(len(fails))
PY
}

cand_pending() {
  python3 - <<'PY'
import json
from pathlib import Path
rows=[json.loads(l) for l in Path("runs/finance_tracelift_state.jsonl").read_text().splitlines() if l.strip()] if Path("runs/finance_tracelift_state.jsonl").exists() else []
fails={r["key"] for r in rows if r.get("kind")=="train_grade" and r.get("ok") and r.get("failure")}
done={r["qid"] for r in rows if r.get("kind")=="candidate" and r.get("ok")}
# each failure wants 3 kinds; approximate pending as failures lacking any cand
need=0
for q in fails:
    kinds={r["mem_kind"] for r in rows if r.get("kind")=="candidate" and r.get("qid")==q and r.get("ok")}
    need += max(0, 3-len(kinds))
print(need)
PY
}

has_stop() {
  python3 -c 'import json; from pathlib import Path; p=Path("runs/finance_tracelift_state.jsonl"); rows=[json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []; raise SystemExit(0 if any(r.get("kind")=="stop" for r in rows) else 1)'
}

snapshot() {
  python3 - <<'PY' | tee -a "$LOG"
import json
from collections import Counter
from pathlib import Path
p=Path("runs/finance_tracelift_state.jsonl")
rows=[json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []
print("state", dict(Counter(r.get("kind") for r in rows)))
ups=[r for r in rows if r.get("kind")=="uplift" and r.get("ok")]
print("uplift_ok", len(ups), "admitted", sum(1 for r in ups if r.get("admitted")))
stops=[r for r in rows if r.get("kind")=="stop"]
if stops:
  print("STOP_SEEN", stops[-1].get("reason"))
mem=Path("runs/finance_tracelift_memory.json")
if mem.exists():
  d=json.loads(mem.read_text())
  print("memory_n", d.get("n"), "dry_run", (d.get("meta") or {}).get("dry_run"))
PY
}

for i in $(seq 1 200); do
  echo "=== TRACECHUNK $i $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$LOG"
  snapshot
  if has_stop; then
    echo "=== STOPPED $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$LOG"
    PYTHONUNBUFFERED=1 bash scripts/use_openrouter_finance.sh scripts/finance_tracelift.py --phase freeze --resume >>"$LOG" 2>&1 || true
    break
  fi

  FC=$(fail_count)
  CP=$(cand_pending)
  echo "fails=$FC cand_kinds_pending=$CP" | tee -a "$LOG"

  if [[ "$FC" -lt "$TARGET_FAILURES" ]]; then
    echo "[phase] train" | tee -a "$LOG"
    PYTHONUNBUFFERED=1 bash scripts/use_openrouter_finance.sh scripts/finance_tracelift.py \
      --phase train --resume --time-budget-s "$CHUNK_S" --max-new 4 \
      --target-failures "$TARGET_FAILURES" \
      --student-model qwen/qwen3.6-27b \
      >>"$LOG" 2>&1 || true
    continue
  fi

  if [[ "$CP" -gt 0 ]]; then
    echo "[phase] candidates" | tee -a "$LOG"
    PYTHONUNBUFFERED=1 bash scripts/use_openrouter_finance.sh scripts/finance_tracelift.py \
      --phase candidates --resume --time-budget-s "$CHUNK_S" --max-new 6 \
      --student-model qwen/qwen3.6-27b \
      >>"$LOG" 2>&1 || true
    continue
  fi

  echo "[phase] gate val_n=$VAL_N k=$K" | tee -a "$LOG"
  PYTHONUNBUFFERED=1 bash scripts/use_openrouter_finance.sh scripts/finance_tracelift.py \
    --phase gate --resume --time-budget-s "$CHUNK_S" --max-new 2 \
    --val-n "$VAL_N" --k "$K" --min-u 1.0 \
    --student-model qwen/qwen3.6-27b \
    >>"$LOG" 2>&1 || true
done

echo "=== LOOP_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$LOG"
