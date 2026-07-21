#!/usr/bin/env bash
# FinancePro-Bench on OpenRouter (CTO-mandated platform).
# Wires all three roles — student / teacher / judge — to OpenRouter, then
# execs a finance script (default: finance_baselines.py).
#
# Usage:
#   bash scripts/use_openrouter_finance.sh --mode headroom --resume
#   bash scripts/use_openrouter_finance.sh scripts/finance_tracelift.py --phase all --resume
#   bash scripts/use_openrouter_finance.sh scripts/finance_eval.py --arm a4 --resume
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -f .env ]]; then
  set -a; source .env; set +a
fi
: "${OPENROUTER_API_KEY:?Add OPENROUTER_API_KEY to .env}"

OR_BASE="https://openrouter.ai/api/v1"

# Student (agent)
export AGENT_BASE_URL="$OR_BASE"
export AGENT_API_KEY="$OPENROUTER_API_KEY"
export STUDENT_MODEL="${STUDENT_MODEL:-qwen/qwen3.6-27b}"

# Teacher (CTO: GLM 5.2 — heavy reasoner, needs large token budget or content
# comes back empty because it truncates mid-thinking).
export TEACHER_BASE_URL="$OR_BASE"
export TEACHER_API_KEY="$OPENROUTER_API_KEY"
export TEACHER_MODEL="${TEACHER_MODEL:-z-ai/glm-5.2}"
export TEACHER_MAX_TOKENS="${TEACHER_MAX_TOKENS:-4000}"

# Judge (must differ from teacher — asserted in code)
export JUDGE_BASE_URL="$OR_BASE"
export JUDGE_API_KEY="$OPENROUTER_API_KEY"
export JUDGE_MODEL="${JUDGE_MODEL:-openai/gpt-5.2}"

export AGENT_TIMEOUT_S="${AGENT_TIMEOUT_S:-120}"

echo "== finance on OpenRouter =="
echo "  student : $STUDENT_MODEL"
echo "  teacher : $TEACHER_MODEL"
echo "  judge   : $JUDGE_MODEL"
echo

if [[ "${1:-}" == *.py ]] || [[ "${1:-}" == scripts/* ]]; then
  script="$1"
  shift
  python3 "$script" "$@"
else
  python3 scripts/finance_baselines.py "$@"
fi
