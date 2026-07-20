#!/usr/bin/env bash
# GSM8K uplift-gated memory on OpenRouter.
# Usage:
#   1. OPENROUTER_API_KEY in .env
#   2. bash scripts/use_openrouter_gsm8k.sh band|tracelift|ablate
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

: "${OPENROUTER_API_KEY:?Add OPENROUTER_API_KEY to .env first}"

export AGENT_BASE_URL="https://openrouter.ai/api/v1"
# Prefer in-band student on GSM8K hard (~0.3–0.6 unaided). 7B Qwen saturates (~0.93).
# Measured: meta-llama/llama-3.2-3b-instruct ≈ 0.47 on 30 held-out hard.
export AGENT_MODEL="${OR_AGENT_MODEL:-meta-llama/llama-3.2-3b-instruct}"

export TEACHER_USE_OPENROUTER=1
export TEACHER_USE_PRIME=0
export TEACHER_BASE_URL="https://openrouter.ai/api/v1"
export TEACHER_API_KEY="$OPENROUTER_API_KEY"
export TEACHER_MODEL="${OR_TEACHER_MODEL:-qwen/qwen3-coder-plus}"
export DISTILL_MODEL="${DISTILL_MODEL:-$TEACHER_MODEL}"

export UPLIFT_GATE="${UPLIFT_GATE:-1}"
# k=1 + 8 cand × 6 val ≈ 96 student calls; raise UPLIFT_K=3 for tighter estimate
export UPLIFT_K="${UPLIFT_K:-1}"
export UPLIFT_MAX_CANDIDATES="${UPLIFT_MAX_CANDIDATES:-8}"
export UPLIFT_VAL_N="${UPLIFT_VAL_N:-6}"
export MEMORY_MAX_TOTAL="${MEMORY_MAX_TOTAL:-5}"
export MEMORY_MAX_PER_DB="${MEMORY_MAX_PER_DB:-5}"
export AGENT_USE_RULES=0
export OPENROUTER_APP_TITLE="${OPENROUTER_APP_TITLE:-agent-self-improvement}"

step="${1:-band}"

echo "== openrouter gsm8k =="
echo "  student: $AGENT_MODEL"
echo "  teacher: $TEACHER_MODEL"
echo "  UPLIFT_GATE=$UPLIFT_GATE UPLIFT_K=$UPLIFT_K MEMORY_MAX_TOTAL=$MEMORY_MAX_TOTAL"
echo "  step=$step"
echo

case "$step" in
  prepare)
    pip3 install -q datasets
    python3 scripts/prepare_gsm8k.py --n-easy 100 --n-hard 150
    ;;
  band)
    # Bare held-out hard accuracy — must land ~0.3–0.6 before TraceLift claims
    python3 orchestrator.py --adapter gsm8k --dry-run-heldout --full
    ;;
  tracelift)
    export MEMORY_MAX_TOTAL=5
    python3 orchestrator.py --adapter gsm8k --hard-curriculum --fresh \
      --n-learn "${N_LEARN:-80}" --n-heldout "${N_HELDOUT:-30}" \
      --heldout-frac "${HELDOUT_FRAC:-0.5}"
    ;;
  ablate)
    # Requires a prior tracelift/learn CorrectionAction in events.jsonl
    python3 orchestrator.py --adapter gsm8k --ablation none,examples \
      --n-heldout "${N_HELDOUT:-30}" --heldout-frac "${HELDOUT_FRAC:-0.5}"
    ;;
  *)
    echo "Usage: $0 {prepare|band|tracelift|ablate}"
    exit 1
    ;;
esac
