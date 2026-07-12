#!/usr/bin/env bash
# Coding-adapter measurement checklist (Phase 1).
# Run from repo root. Student: local Ollama by default, or Prime via
# scripts/use_prime_student.sh. Teacher (--full): MINIMAX_API_KEY.
set -euo pipefail
cd "$(dirname "$0")/.."

# Caller overrides (e.g. use_prime_student.sh) must win over .env Ollama defaults.
_caller_base="${AGENT_BASE_URL-}"
_caller_model="${AGENT_MODEL-}"
_caller_rules="${AGENT_USE_RULES-}"

if [[ -f .env ]]; then
  # shellcheck disable=SC1091
  set -a
  source .env
  set +a
fi

export AGENT_BASE_URL="${_caller_base:-${AGENT_BASE_URL:-http://localhost:11434/v1}}"
export AGENT_MODEL="${_caller_model:-${AGENT_MODEL:-qwen2.5:1.5b-instruct}}"
export AGENT_USE_RULES="${_caller_rules:-${AGENT_USE_RULES:-0}}"

step="${1:-all}"

echo "== coding eval =="
echo "  AGENT_BASE_URL=$AGENT_BASE_URL"
echo "  AGENT_MODEL=$AGENT_MODEL"
echo "  AGENT_USE_RULES=$AGENT_USE_RULES"
echo "  step=$step"
echo

case "$step" in
  probe)
    echo "[1/1] Cheap with/without probe (~22 calls) ..."
    python3 orchestrator.py --adapter coding --probe
    ;;
  degraded)
    echo "[1/1] Dry-run degraded pool (detector headroom) ..."
    python3 orchestrator.py --adapter coding --dry-run-degraded
    ;;
  heldout)
    echo "[1/1] Dry-run held-out (WITHOUT examples) ..."
    python3 orchestrator.py --adapter coding --dry-run-heldout
    ;;
  full)
    echo "[1/1] Full loop --fresh ..."
    python3 orchestrator.py --adapter coding --full --fresh
    ;;
  significance)
    echo "[1/1] McNemar paired test (requires prior --full) ..."
    python3 orchestrator.py --adapter coding --significance
    ;;
  all)
    echo "[1/4] Probe ..."
    python3 orchestrator.py --adapter coding --probe
    echo
    echo "[2/4] Dry-run degraded ..."
    python3 orchestrator.py --adapter coding --dry-run-degraded
    echo
    echo "[3/4] Full loop ..."
    python3 orchestrator.py --adapter coding --full --fresh
    echo
    echo "[4/4] Significance ..."
    python3 orchestrator.py --adapter coding --significance
    echo
    echo "Done. Copy hard-bucket WITHOUT / WITH / Δ into README Results table."
    ;;
  *)
    echo "Usage: $0 {probe|degraded|heldout|full|significance|all}"
    exit 1
    ;;
esac
