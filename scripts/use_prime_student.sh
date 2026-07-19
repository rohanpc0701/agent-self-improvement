#!/usr/bin/env bash
# Coding harness on Prime: cheap student + stronger teacher + KG on recovery.
# Usage (from repo root):
#   1. Add PRIME_API_KEY=... to .env
#   2. bash scripts/use_prime_student.sh list|smoke|probe|full
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

: "${PRIME_API_KEY:?Add PRIME_API_KEY to .env first}"

# Force Prime for the student — ignore AGENT_* from .env (often local Ollama).
export AGENT_BASE_URL="https://api.pinference.ai/api/v1"
export AGENT_MODEL="${PRIME_AGENT_MODEL:-meta-llama/Llama-3.2-3B-Instruct}"

# Teacher on Prime (MiniMax direct is optional; often out of balance).
# Override: PRIME_TEACHER_MODEL=qwen/qwen3-8b bash scripts/use_prime_student.sh full
export TEACHER_USE_PRIME=1
export TEACHER_BASE_URL="https://api.pinference.ai/api/v1"
export TEACHER_API_KEY="$PRIME_API_KEY"
export TEACHER_MODEL="${PRIME_TEACHER_MODEL:-minimax/minimax-m2.5}"
# Distill (trap,fix) on the same endpoint unless overridden
export DISTILL_MODEL="${DISTILL_MODEL:-$TEACHER_MODEL}"

# Thesis measurement: examples + KG on recovery (probe forces rules off below)
export AGENT_USE_RULES="${PRIME_USE_RULES:-1}"

step="${1:-smoke}"

echo "== prime coding harness =="
echo "  student: $AGENT_MODEL @ $AGENT_BASE_URL"
echo "  teacher: $TEACHER_MODEL @ Prime"
echo "  AGENT_USE_RULES=$AGENT_USE_RULES"
echo "  step=$step"
echo

case "$step" in
  list)
    python3 - <<'PY'
import os
from openai import OpenAI
c = OpenAI(api_key=os.environ["PRIME_API_KEY"], base_url=os.environ["AGENT_BASE_URL"])
ids = sorted(m.id for m in c.models.list().data)
cheap = [i for i in ids if any(k in i.lower() for k in ("1b","1.5b","2b","3b","7b","8b","mini","small","3.2-3b","3.2-1b","m2.5","m2.7","m3"))]
print(f"{len(ids)} models; candidates:")
for i in cheap[:50] or ids[:40]:
    print(" ", i)
PY
    ;;
  smoke)
    python3 - <<'PY'
import os, time
from openai import OpenAI
from correction.provider import teacher_client_and_model
from harness.sandbox import extract_python_code, execution_accuracy

# student smoke
c = OpenAI(api_key=os.environ["PRIME_API_KEY"], base_url=os.environ["AGENT_BASE_URL"])
model = os.environ["AGENT_MODEL"]
t0 = time.time()
r = c.chat.completions.create(
    model=model,
    messages=[{"role":"user","content":"Return only: def add(a, b):\\n    return a + b"}],
    temperature=0,
    max_tokens=64,
)
print(f"student ok model={model} latency={time.time()-t0:.1f}s")
print((r.choices[0].message.content or "")[:120])

# teacher smoke — climb_stairs repair
from adapters.coding import _index, _teacher_repair
p = _index()["h_climb_stairs"]
broken = "def climb_stairs(n):\n    return n\n"
t0 = time.time()
code = _teacher_repair(p, broken)
client, tmodel = teacher_client_and_model()
print(f"teacher endpoint model={tmodel}")
if not code:
    raise SystemExit("teacher returned no code")
acc, _, err = execution_accuracy(code, p["function_name"], p["tests"])
print(f"teacher repair latency={time.time()-t0:.1f}s acc={acc} err={err!r}")
print(code[:200])
if acc != 1.0:
    raise SystemExit("teacher repair failed unit tests — pick a stronger PRIME_TEACHER_MODEL")
print("smoke ok")
PY
    ;;
  probe)
    # probe is WITH/WITHOUT gold examples — keep rules off for clean delta
    AGENT_USE_RULES=0 bash scripts/run_coding_eval.sh probe
    ;;
  degraded)
    bash scripts/run_coding_eval.sh degraded
    ;;
  full)
    bash scripts/run_coding_eval.sh full
    ;;
  compare)
    # Requires a prior full/curriculum run (CorrectionAction in events.jsonl)
    python3 orchestrator.py --adapter coding --compare-teacher --full
    ;;
  full-compare)
    # Full loop then student+memory vs unaided teacher on held-out
    COMPARE_TEACHER=1 bash scripts/run_coding_eval.sh full
    ;;
  curriculum)
    # Eval pipeline: easy warmup → 100 hard LEARN → KG → vs teacher on new hard
    # Override: N_LEARN=120 N_HELDOUT=40 bash scripts/use_prime_student.sh curriculum
    python3 orchestrator.py --adapter coding --hard-curriculum \
      --n-learn "${N_LEARN:-100}" \
      --n-heldout "${N_HELDOUT:-40}" \
      --heldout-frac "${HELDOUT_FRAC:-0.5}"
    ;;
  ablate)
    # Diagnostics: learn once, then frozen-memory ablation + capacity probe.
    #   Arms (3B): none / examples / rules / both  — same held-out Qs
    #   Capacity (7B): none / both
    # Override: PRIME_CAPACITY_MODEL=meta-llama/Llama-3.1-8B-Instruct \
    #   N_LEARN=100 N_HELDOUT=30 bash scripts/use_prime_student.sh ablate
    capacity_model="${PRIME_CAPACITY_MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct}"

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
