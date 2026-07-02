#!/usr/bin/env bash
# Replay the recorded self-improvement run in the viewer (no API key, no Ollama).
set -euo pipefail
cd "$(dirname "$0")/.."
export VIEWER_LOG="${VIEWER_LOG:-fixtures/demo_events.jsonl}"
echo "Viewer: http://127.0.0.1:8011  (log: $VIEWER_LOG)"
exec python3 -m uvicorn viewer.app:app --port 8011
