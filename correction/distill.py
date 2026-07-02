"""Distill broken vs fixed SQL into a reusable CorrectionRule.

Uses a cheap model call (DISTILL_MODEL env var, default MiniMax-M2.7-highspeed)
via the OpenAI-compatible MiniMax API — same stack as the rest of the repo.
Prompts for JSON only; parses defensively with a fallback.
"""
from __future__ import annotations

import json
import os
import re
import uuid

import logging

from openai import OpenAI

from .contracts import FailedRun, CorrectionRule

log = logging.getLogger(__name__)

_MINIMAX_BASE_URL = "https://api.minimax.io/v1"
_DISTILL_MODEL = os.environ.get("DISTILL_MODEL", "MiniMax-M2.7-highspeed")

_client: OpenAI | None = None


def distill(failed: FailedRun, fixed_sql: str) -> CorrectionRule:
    """Diff broken vs fixed SQL and return a CorrectionRule (scope='db')."""
    rule_id = f"rule:{failed.db_id}:{uuid.uuid4().hex[:8]}"

    try:
        parsed = _call_model(failed, fixed_sql)
        applies_to = [
            f"schema:{failed.db_id}:{a}" if not a.startswith("schema:") else a
            for a in parsed.get("applies_to", [])
        ]
        return CorrectionRule(
            id=rule_id,
            scope="db",
            db_id=failed.db_id,
            trap=parsed["trap"],
            fix=parsed["fix"],
            trigger=parsed["trigger"],
            applies_to=applies_to,
            source="react_repair",
            seen_dbs=[failed.db_id],
        )
    except Exception as e:
        log.warning("distill: model call failed for run %s, using fallback rule: %s",
                    failed.run_id, e)
        return _fallback(rule_id, failed, fixed_sql)


# ── internals ─────────────────────────────────────────────────────────────────

def _get_client() -> OpenAI:
    global _client
    if _client is None:
        key = os.environ.get("MINIMAX_API_KEY")
        if not key:
            raise RuntimeError("MINIMAX_API_KEY is not set — distill model calls will fail.")
        _client = OpenAI(api_key=key, base_url=_MINIMAX_BASE_URL)
    return _client


def _strip_think(text: str) -> str:
    """Remove <think>...</think> reasoning blocks (MiniMax M-series)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL).strip()
    return text


def _call_model(failed: FailedRun, fixed_sql: str) -> dict:
    schema = "\n".join(
        f"  {t}({', '.join(c) if isinstance(c, list) else c})"
        for t, c in failed.schema.items()
    )
    prompt = f"""Analyze these two SQL queries for the same question and extract a reusable correction rule.

Question: {failed.question}
Schema:
{schema}

Broken SQL: {failed.broken_sql}
Fixed SQL:  {fixed_sql}
Execution error: {failed.execution_error or "none"}

Output JSON ONLY — no prose, no markdown fences:
{{
  "trap": "<specific mistake pattern the agent made>",
  "fix": "<specific correction to apply>",
  "trigger": "<1-3 keywords from the question or schema that signal this trap>",
  "applies_to": ["<table_or_column_name>"]
}}"""

    client = _get_client()
    response = client.chat.completions.create(
        model=_DISTILL_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    text = _strip_think(response.choices[0].message.content or "").strip()

    # Strip accidental markdown fences
    if text.startswith("```"):
        lines = text.splitlines()
        start = 1
        end = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
        text = "\n".join(lines[start:end])

    return json.loads(text)


def _fallback(rule_id: str, failed: FailedRun, fixed_sql: str) -> CorrectionRule:
    tables = list(failed.schema.keys())
    trigger = next((t for t in tables if t.lower() in fixed_sql.lower()), failed.db_id)
    return CorrectionRule(
        id=rule_id,
        scope="db",
        db_id=failed.db_id,
        trap=f"Incorrect SQL for question about {trigger}",
        fix=f"Use: {fixed_sql[:200]}",
        trigger=trigger,
        applies_to=[f"schema:{failed.db_id}:{trigger}"],
        source="react_repair",
        seen_dbs=[failed.db_id],
    )
