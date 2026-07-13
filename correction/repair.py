"""ReAct repair loop: teacher model fixes a broken SQL query.

Uses the stronger MiniMax tier (TEACHER_MODEL env var, default MiniMax-M3) via the
OpenAI-compatible API — same stack as correction/teacher.py and harness/agent.py.
Only called on confirmed failures — off the hot path.
"""
from __future__ import annotations

import logging
import os
import re
import sqlite3
from pathlib import Path
from typing import Optional

from openai import OpenAI

from .contracts import FailedRun

log = logging.getLogger(__name__)

_MINIMAX_BASE_URL = "https://api.minimax.io/v1"
_TEACHER_MODEL = os.environ.get("TEACHER_MODEL", "MiniMax-M3")
_MAX_ITERS = 3

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        key = os.environ.get("MINIMAX_API_KEY")
        if not key:
            raise RuntimeError("MINIMAX_API_KEY is not set — repair teacher calls will fail.")
        _client = OpenAI(api_key=key, base_url=_MINIMAX_BASE_URL)
    return _client


def repair(failed: FailedRun, db_path: Optional[Path] = None) -> str:
    """Run a ReAct loop (max 3 iters) and return the best corrected SQL."""
    best_sql = failed.broken_output
    history: list[str] = []

    for iteration in range(1, _MAX_ITERS + 1):
        prompt = _build_prompt(failed, iteration, history)
        try:
            sql_candidate = _extract_sql(_call_model(prompt))
        except Exception as e:
            log.warning("repair: model call failed on iter %d for run %s: %s",
                        iteration, failed.run_id, e)
            break

        if not sql_candidate:
            continue

        best_sql = sql_candidate
        if db_path and db_path.exists():
            rows, error = _execute(sql_candidate, db_path)
            if error is None and _match(rows, failed.expected_result):
                return best_sql
            history.append(f"iter{iteration}: {sql_candidate[:80]} -> err={error}, rows={rows[:2]}")
        else:
            # No live DB — accept the first syntactically plausible candidate
            if sql_candidate.strip().upper().startswith("SELECT"):
                return best_sql

    if best_sql == failed.broken_output:
        log.warning("repair: could not improve SQL for run %s (returning broken_sql unchanged)",
                    failed.run_id)
    return best_sql


# ── internals ─────────────────────────────────────────────────────────────────

def _strip_think(text: str) -> str:
    """Remove <think>...</think> reasoning blocks (MiniMax M-series)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL).strip()
    return text


def _call_model(prompt: str) -> str:
    client = _get_client()
    response = client.chat.completions.create(
        model=_TEACHER_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    raw = response.choices[0].message.content or ""
    return _strip_think(raw).strip()


def _build_prompt(failed: FailedRun, iteration: int, history: list[str]) -> str:
    prev = ("Previous attempts:\n" + "\n".join(history[-2:]) + "\n\n") if history else ""
    exp = str(failed.expected_result[:3]) if failed.expected_result else "unknown"
    obs = str(failed.observed_result[:3]) if failed.observed_result else "none"
    return f"""You are a SQL expert repairing a broken query (ReAct style).

Schema:
{_schema_text(failed.schema)}

Question: {failed.question}
Broken SQL: {failed.broken_output}
Execution error: {failed.execution_error or "none"}
Expected result (sample): {exp}
Observed result (sample): {obs}

{prev}Iteration {iteration}/{_MAX_ITERS}. Reply ONLY in this format:
Thought: <one sentence on what went wrong>
SQL: <corrected SQL, single line, no markdown fences>"""


def _extract_sql(response: str) -> str:
    for line in response.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("SQL:"):
            return stripped.split(":", 1)[1].strip()
    for line in response.splitlines():
        if line.strip().upper().startswith("SELECT"):
            return line.strip()
    return ""


def _execute(sql: str, db_path: Path) -> tuple[list, str | None]:
    try:
        conn = sqlite3.connect(str(db_path))
        rows = conn.cursor().execute(sql).fetchall()
        conn.close()
        return rows, None
    except Exception as e:
        return [], str(e)


def _match(observed: list, expected: list | None) -> bool:
    if expected is None:
        return False
    return set(map(tuple, observed)) == set(map(tuple, expected))


def _schema_text(schema: dict) -> str:
    return "\n".join(
        f"  {table}({', '.join(cols) if isinstance(cols, list) else cols})"
        for table, cols in schema.items()
    )
