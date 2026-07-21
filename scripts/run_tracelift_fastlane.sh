#!/usr/bin/env bash
# Fast-lane TraceLift build on OpenRouter — first-signal scope.
# Reasoning disabled (reasoning.enabled=false, committed in adapters/finance.py),
# student answers capped for speed, tight gating slice. Both eval arms share the
# same STUDENT_MAX_TOKENS cap so GAP (A4-A1) stays fair.
#
# Run from repo root. Resumable — re-run to continue after any interruption.
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; source .env; set +a

OR="https://openrouter.ai/api/v1"
export AGENT_BASE_URL="$OR" AGENT_API_KEY="$OPENROUTER_API_KEY"
export STUDENT_MODEL="qwen/qwen3.6-27b"
export TEACHER_BASE_URL="$OR" TEACHER_API_KEY="$OPENROUTER_API_KEY"
export TEACHER_MODEL="z-ai/glm-5.2" TEACHER_MAX_TOKENS="4000"
export JUDGE_BASE_URL="$OR" JUDGE_API_KEY="$OPENROUTER_API_KEY"
export JUDGE_MODEL="openai/gpt-5.2"
export AGENT_TIMEOUT_S="180"
export STUDENT_MAX_TOKENS="3500"

LOG=runs/finance_tracelift_fastlane.log
echo "=== fast-lane build $(date) ===" | tee -a "$LOG"
# Fast-lane scope: <=10 memory items, gate on 10 validation Qs, K=1.
# Loops resumable chunks until the build freezes (stopping rule) or scope is met.
for i in $(seq 1 40); do
  python3 scripts/finance_tracelift.py \
    --student-model qwen/qwen3.6-27b \
    --max-new 10 --val-n 10 --k 1 \
    --resume --time-budget-s 480 >> "$LOG" 2>&1 || true
  if grep -qiE "FROZEN|build complete|stopping rule|memory frozen" "$LOG"; then
    echo "=== frozen — build done ===" | tee -a "$LOG"; break
  fi
done
echo "=== fast-lane end $(date) ===" | tee -a "$LOG"
