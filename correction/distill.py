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
_DEFAULT_DISTILL_MODEL = "MiniMax-M2.7-highspeed"

_client: OpenAI | None = None
_client_model: str | None = None


def distill(failed: FailedRun, fixed_sql: str) -> CorrectionRule:
    """Diff broken vs fixed SQL and return a CorrectionRule (scope='db')."""
    return _distill_with_prompt(failed, fixed_sql, domain="sql")


def distill_code(failed: FailedRun, fixed_code: str) -> CorrectionRule:
    """Diff broken vs fixed Python and return a topic-scoped CorrectionRule."""
    return _distill_with_prompt(failed, fixed_code, domain="code")


def _distill_with_prompt(
    failed: FailedRun, fixed: str, *, domain: str
) -> CorrectionRule:
    rule_id = f"rule:{failed.domain_id}:{uuid.uuid4().hex[:8]}"
    try:
        parsed = _call_model(failed, fixed, domain=domain)
        applies_to = [
            f"schema:{failed.domain_id}:{a}" if not a.startswith("schema:") else a
            for a in parsed.get("applies_to", [])
        ]
        return CorrectionRule(
            id=rule_id,
            scope="db",
            db_id=failed.domain_id,
            trap=parsed["trap"],
            fix=parsed["fix"],
            trigger=parsed["trigger"],
            applies_to=applies_to,
            source="react_repair",
            seen_dbs=[failed.domain_id],
        )
    except Exception as e:
        log.warning(
            "distill: model call failed for run %s, using fallback rule: %s",
            failed.run_id,
            e,
        )
        return _fallback(rule_id, failed, fixed, domain=domain)


# ── internals ─────────────────────────────────────────────────────────────────

def _get_client() -> tuple[OpenAI, str]:
    """Reuse teacher endpoint resolution so distill works on Prime when MiniMax is dry."""
    global _client, _client_model
    if _client is None:
        from correction.provider import teacher_client_and_model

        _client, default_model = teacher_client_and_model()
        _client_model = os.environ.get("DISTILL_MODEL") or default_model
    assert _client_model is not None
    return _client, _client_model


def _strip_think(text: str) -> str:
    """Remove <think>...</think> reasoning blocks (MiniMax M-series)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL).strip()
    return text


def _call_model(failed: FailedRun, fixed: str, *, domain: str = "sql") -> dict:
    schema = "\n".join(
        f"  {t}({', '.join(c) if isinstance(c, list) else c})"
        for t, c in failed.schema.items()
    )
    if domain == "code":
        prompt = f"""Analyze these two Python solutions for the same problem and extract a reusable correction rule.

Problem: {failed.question}
Topic / tags:
{schema or failed.domain_id}

Broken code:
{failed.broken_output}

Fixed code:
{fixed}

Runtime / test error: {failed.execution_error or "none"}

Output JSON ONLY — no prose, no markdown fences:
{{
  "trap": "<specific mistake pattern the agent made>",
  "fix": "<specific correction to apply>",
  "trigger": "<1-3 keywords from the problem that signal this trap>",
  "applies_to": ["<topic_or_algorithm_cue>"]
}}"""
    else:
        prompt = f"""Analyze these two SQL queries for the same question and extract a reusable correction rule.

Question: {failed.question}
Schema:
{schema}

Broken SQL: {failed.broken_output}
Fixed SQL:  {fixed}
Execution error: {failed.execution_error or "none"}

Output JSON ONLY — no prose, no markdown fences:
{{
  "trap": "<specific mistake pattern the agent made>",
  "fix": "<specific correction to apply>",
  "trigger": "<1-3 keywords from the question or schema that signal this trap>",
  "applies_to": ["<table_or_column_name>"]
}}"""

    client, model = _get_client()
    response = client.chat.completions.create(
        model=model,
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


def _fallback(
    rule_id: str, failed: FailedRun, fixed: str, *, domain: str = "sql"
) -> CorrectionRule:
    tables = list(failed.schema.keys())
    trigger = next((t for t in tables if t.lower() in fixed.lower()), failed.domain_id)
    if domain == "code":
        trap = f"Incorrect algorithm / edge-case handling for {trigger}"
        fix = f"Prefer a solution structured like: {fixed[:240]}"
    else:
        trap = f"Incorrect SQL for question about {trigger}"
        fix = f"Use: {fixed[:200]}"
    return CorrectionRule(
        id=rule_id,
        scope="db",
        db_id=failed.domain_id,
        trap=trap,
        fix=fix,
        trigger=trigger,
        applies_to=[f"schema:{failed.domain_id}:{trigger}"],
        source="react_repair",
        seen_dbs=[failed.domain_id],
    )
