#!/usr/bin/env bash
# Coding harness on OpenRouter: wide model catalog for student + teacher.
# Usage (from repo root):
#   1. Add OPENROUTER_API_KEY=... to .env
#   2. bash scripts/use_openrouter_student.sh list|smoke|curriculum|ablate|...
#
# Defaults follow the ablation decision tree: mid-size coder student,
# stronger teacher. Override with OR_AGENT_MODEL / OR_TEACHER_MODEL.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

: "${OPENROUTER_API_KEY:?Add OPENROUTER_API_KEY to .env first}"

# Force OpenRouter — ignore AGENT_* from .env (often local Ollama / Prime).
export AGENT_BASE_URL="https://openrouter.ai/api/v1"
# Mid-size default: capacity probe showed 3B is the floor; promote ~7–14B coder.
export AGENT_MODEL="${OR_AGENT_MODEL:-qwen/qwen3-coder}"

# Teacher on OpenRouter (same key). Stronger default for verified repairs.
export TEACHER_USE_OPENROUTER=1
export TEACHER_USE_PRIME=0
export TEACHER_BASE_URL="https://openrouter.ai/api/v1"
export TEACHER_API_KEY="$OPENROUTER_API_KEY"
export TEACHER_MODEL="${OR_TEACHER_MODEL:-qwen/qwen3-coder-plus}"
export DISTILL_MODEL="${DISTILL_MODEL:-$TEACHER_MODEL}"

export AGENT_USE_RULES="${OR_USE_RULES:-1}"
export OPENROUTER_APP_TITLE="${OPENROUTER_APP_TITLE:-agent-self-improvement}"

step="${1:-smoke}"

echo "== openrouter coding harness =="
echo "  student: $AGENT_MODEL @ $AGENT_BASE_URL"
echo "  teacher: $TEACHER_MODEL @ OpenRouter"
echo "  AGENT_USE_RULES=$AGENT_USE_RULES"
echo "  step=$step"
echo

case "$step" in
  list)
    python3 - <<'PY'
import os
from openai import OpenAI
c = OpenAI(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url=os.environ["AGENT_BASE_URL"],
    default_headers={"X-Title": os.environ.get("OPENROUTER_APP_TITLE", "agent-self-improvement")},
)
ids = sorted(m.id for m in c.models.list().data)
keys = ("coder", "code", "7b", "8b", "14b", "32b", "qwen3", "llama-3.1", "llama-3.3", "deepseek", "gemini")
hits = [i for i in ids if any(k in i.lower() for k in keys)]
print(f"{len(ids)} models; coding / mid-size candidates:")
for i in (hits[:60] or ids[:40]):
    print(" ", i)
PY
    ;;
  smoke)
    python3 - <<'PY'
import os, time
from openai import OpenAI
from correction.provider import teacher_client_and_model
from harness.sandbox import execution_accuracy

c = OpenAI(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url=os.environ["AGENT_BASE_URL"],
    default_headers={"X-Title": os.environ.get("OPENROUTER_APP_TITLE", "agent-self-improvement")},
)
model = os.environ["AGENT_MODEL"]
t0 = time.time()
r = c.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Return only: def add(a, b):\\n    return a + b"}],
    temperature=0,
    max_tokens=64,
)
print(f"student ok model={model} latency={time.time()-t0:.1f}s")
print((r.choices[0].message.content or "")[:120])

from adapters.coding import _index, _teacher_repair
p = _index()["h_climb_stairs"]
broken = "def climb_stairs(n):\n    return n\n"
t0 = time.time()
code = _teacher_repair(p, broken)
client, tmodel = teacher_client_and_model()
print(f"teacher endpoint model={tmodel}")
if not code:
    raise SystemExit("teacher returned no code — pick a stronger OR_TEACHER_MODEL")
acc, _, err = execution_accuracy(code, p["function_name"], p["tests"])
print(f"teacher repair latency={time.time()-t0:.1f}s acc={acc} err={err!r}")
print(code[:200])
if acc != 1.0:
    raise SystemExit("teacher repair failed unit tests — pick a stronger OR_TEACHER_MODEL")
print("smoke ok")
PY
    ;;
  probe)
    AGENT_USE_RULES=0 bash scripts/run_coding_eval.sh probe
    ;;
  degraded)
    bash scripts/run_coding_eval.sh degraded
    ;;
  full)
    bash scripts/run_coding_eval.sh full
    ;;
  compare)
    python3 orchestrator.py --adapter coding --compare-teacher --full
    ;;
  full-compare)
    COMPARE_TEACHER=1 bash scripts/run_coding_eval.sh full
    ;;
  curriculum)
    python3 orchestrator.py --adapter coding --hard-curriculum \
      --n-learn "${N_LEARN:-100}" \
      --n-heldout "${N_HELDOUT:-40}" \
      --heldout-frac "${HELDOUT_FRAC:-0.5}"
    ;;
  ablate)
    capacity_model="${OR_CAPACITY_MODEL:-$AGENT_MODEL}"
    echo "== 1/3 learn phase (fresh) =="
    python3 orchestrator.py --adapter coding --hard-curriculum --fresh \
      --n-learn "${N_LEARN:-100}" --n-heldout "${N_HELDOUT:-30}" \
      --heldout-frac "${HELDOUT_FRAC:-0.5}"
    echo "== 2/3 ablation arms (student: $AGENT_MODEL) =="
    python3 orchestrator.py --adapter coding --ablation all \
      --n-heldout "${N_HELDOUT:-30}" --heldout-frac "${HELDOUT_FRAC:-0.5}"
    echo "== 3/3 capacity probe (student: $capacity_model) =="
    AGENT_MODEL="$capacity_model" \
      python3 orchestrator.py --adapter coding --ablation none,both \
      --n-heldout "${N_HELDOUT:-30}" --heldout-frac "${HELDOUT_FRAC:-0.5}"
    ;;
  *)
    echo "Usage: $0 {list|smoke|probe|degraded|full|compare|full-compare|curriculum|ablate}"
    exit 1
    ;;
esac
