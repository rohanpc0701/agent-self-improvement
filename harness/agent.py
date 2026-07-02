"""Text-to-SQL student agent.

KEY: the prompt MUST include config.few_shot_examples — that growing list is how the
agent recovers after correction feeds it learned examples.

- generate_sql(question, schema_text, config) -> (sql, tokens, latency_ms, reasoning)

Providers (student / hot path):
  default -> MiniMax cloud API (MINIMAX_API_KEY required)
  local   -> any OpenAI-compatible server, e.g. Ollama:
             export AGENT_BASE_URL=http://localhost:11434/v1
             then set AgentConfig.model to e.g. "qwen2.5:1.5b-instruct".
             No API key needed — the weak student runs free and local;
             only the teacher (correction path) calls the cloud.

Teacher (correction stage) stays on MiniMax-M3 via correction/teacher.py.
"""
from __future__ import annotations

import os
import re
import time

from openai import OpenAI

from contracts.schemas import AgentConfig, FewShotExample

_MINIMAX_BASE_URL = "https://api.minimax.io/v1"

_client: OpenAI | None = None


class MissingCredentialsError(RuntimeError):
    """Raised when MINIMAX_API_KEY is not set — fail fast instead of emitting
    error-SQL telemetry that silently pollutes the drift stream."""


def _base_url() -> str:
    return os.environ.get("AGENT_BASE_URL", _MINIMAX_BASE_URL)


def _is_local() -> bool:
    url = _base_url()
    return "localhost" in url or "127.0.0.1" in url


def require_api_key() -> None:
    """Call at startup so a missing key stops the run loudly, before any
    telemetry is written. Without this, every run becomes a fake '-- error'
    record that the evaluator can score as valid/correct by accident.

    Skipped for local providers (AGENT_BASE_URL pointing at localhost) —
    Ollama and friends don't need credentials."""
    if _is_local():
        return
    if not os.environ.get("MINIMAX_API_KEY"):
        raise MissingCredentialsError(
            "MINIMAX_API_KEY is not set. Export it in THIS shell before running:\n"
            "    export MINIMAX_API_KEY=sk-...\n"
            "then re-run. (A key set in another terminal does not carry over.)\n"
            "Or run the student locally with no key:\n"
            "    export AGENT_BASE_URL=http://localhost:11434/v1  # Ollama"
        )


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        require_api_key()
        _client = OpenAI(
            api_key=os.environ.get("MINIMAX_API_KEY", "ollama"),  # local servers ignore the key
            base_url=_base_url(),
        )
    return _client


_SYSTEM = (
    "You are a SQL expert. Given a database schema and a question, "
    "write a single valid SQL SELECT statement. "
    "Return ONLY the SQL, no markdown fences, no explanation."
)


def _correction_rules_block(db_id: str, question: str) -> str:
    """Query the correction knowledge graph for rules learned from past failures.

    Second feedback channel alongside few-shot examples: correction distills each
    repaired failure into a (trap, fix) rule attached to schema nodes; this read
    hook splices matching rules into the prompt. Returns "" when the graph is
    empty/absent so the harness never depends on the correction stage being run.
    """
    if not db_id:
        return ""
    try:
        from correction.inject import build_context, format_prompt_block
        return format_prompt_block(build_context(db_id, question))
    except Exception:
        return ""


def _build_prompt(
    question: str,
    schema: str,
    examples: list[FewShotExample],
    db_id: str = "",
    use_rules: bool = True,
) -> str:
    parts = [f"Schema:\n{schema}"]
    if examples:
        # Only show examples for the same database — cross-schema SQL is pure noise
        # because it references tables/columns that don't exist in the current schema.
        relevant = (
            [e for e in examples if not e.db_id or e.db_id == db_id]
            if db_id else examples
        )
        if relevant:
            shots = "\n\n".join(
                f"Q: {e.question}\nSQL: {e.correct_sql}" for e in relevant[:16]
            )
            parts.append(f"Few-shot examples:\n{shots}")
    if use_rules:
        rules_block = _correction_rules_block(db_id, question)
        if rules_block:
            parts.append(rules_block)
    parts.append(f"Question: {question}\nSQL:")
    return "\n\n".join(parts)


def _strip_think(text: str) -> str:
    """Remove <think>...</think> reasoning blocks (MiniMax M-series reasoning models).
    Also handles unclosed <think> (model cut off mid-reasoning — no valid SQL follows).
    """
    # closed block: strip the think section, keep what follows
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # unclosed block: model was cut off mid-think — strip from <think> to end
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL).strip()
    return text


def _strip_fences(text: str) -> str:
    text = text.strip()
    m = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else text


def _clean_response(text: str) -> str:
    return _strip_fences(_strip_think(text))


def _extract_think(text: str) -> str:
    """Return the raw content of the first <think>...</think> block, or empty string."""
    m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    return m.group(1).strip() if m else ""


def generate_sql(
    question: str,
    schema: str,
    config: AgentConfig,
    db_id: str = "",
    use_rules: bool = True,
) -> tuple[str, int, float, str]:
    """Returns (sql, tokens, latency_ms, reasoning). sql may be an error comment on failure.

    use_rules=False disables knowledge-graph rule injection — required for
    contamination-free WITHOUT-examples measurement passes (dry-run/significance).
    """
    t0 = time.time()
    try:
        client = _get_client()
        prompt = _build_prompt(
            question, schema, config.few_shot_examples, db_id=db_id, use_rules=use_rules
        )
        response = client.chat.completions.create(
            model=config.model,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        raw = response.choices[0].message.content or ""
        reasoning = _extract_think(raw)
        sql = _clean_response(raw)
        tokens = response.usage.total_tokens if response.usage else 0
    except Exception as e:
        sql = f"-- error: {e}"
        reasoning = ""
        tokens = 0
    latency_ms = (time.time() - t0) * 1000
    return sql, tokens, latency_ms, reasoning
