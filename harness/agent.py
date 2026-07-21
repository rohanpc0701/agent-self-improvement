"""Text-to-SQL student agent.

KEY: the prompt MUST include config.few_shot_examples — that growing list is how the
agent recovers after correction feeds it learned examples.

- generate_sql(question, schema_text, config) -> (sql, tokens, latency_ms, reasoning)

Providers (student / hot path):
  default -> MiniMax cloud API (MINIMAX_API_KEY required)
  local   -> any OpenAI-compatible server, e.g. Ollama:
             export AGENT_BASE_URL=http://localhost:11434/v1
  prime   -> Prime Intellect Inference (OpenAI-compatible):
             export AGENT_BASE_URL=https://api.pinference.ai/api/v1
             export PRIME_API_KEY=...
             export AGENT_MODEL=<cheap open model id>
  openrouter -> OpenRouter (wide model catalog):
             export AGENT_BASE_URL=https://openrouter.ai/api/v1
             export OPENROUTER_API_KEY=...
             export AGENT_MODEL=<provider/model id>

Teacher (correction stage) stays on MiniMax-M3 via correction/teacher.py
unless TEACHER_* overrides are set there.
"""
from __future__ import annotations

import os
import random
import re
import time

from openai import OpenAI

from contracts.schemas import AgentConfig, FewShotExample

_RETRYABLE_STATUS = (408, 429, 500, 502, 503, 504)


class ProviderPinError(RuntimeError):
    """OpenRouter served a request from a provider other than the pinned one.

    Reproducibility depends on one provider/serving-config; if drift is detected
    we abort loudly rather than continue with contaminated data.
    """


def _pinned_model() -> str:
    # Which model slug must be provider-pinned (reproducibility target).
    return os.environ.get("OPENROUTER_PIN_MODEL", "deepseek/deepseek-v4-pro").strip()


def _provider_order() -> list[str]:
    # Ordered allow-list. First = primary; the rest = permitted fallbacks (only
    # these, nothing else). Comma-separated, e.g. "fireworks,together".
    raw = os.environ.get("OPENROUTER_PROVIDER_ORDER", "fireworks")
    return [p.strip() for p in raw.split(",") if p.strip()]


# How many pinned requests were served by a NON-primary (backup) provider.
_provider_fallback_count = 0


def provider_fallback_count() -> int:
    """Number of pinned requests served by a backup provider (not the primary).

    Runs should report this — any value > 0 means some results came from a
    different serving config than the primary provider.
    """
    return _provider_fallback_count


def reset_provider_fallback_count() -> None:
    global _provider_fallback_count
    _provider_fallback_count = 0


def openrouter_provider_pin(model: str) -> dict:
    """Return the OpenRouter `provider` routing object for a pinned model, else {}.

    Constrains the model to an ORDERED allow-list of providers and nothing else
    (allow_fallbacks=false = never route outside the list). Config-driven:
      OPENROUTER_PROVIDER_ORDER  comma list, primary first (default "fireworks,together")
      OPENROUTER_PROVIDER_QUANT  optional precision lock, e.g. "fp8" (unset = don't pin)
      OPENROUTER_PIN_MODEL       model slug to pin (default deepseek/deepseek-v4-pro)
    Only applies on the OpenRouter base and only for the pinned model, so other
    models (judge, teacher) are untouched.
    """
    if not _is_openrouter() or (model or "").strip() != _pinned_model():
        return {}
    order = _provider_order()
    if not order:
        return {}
    provider: dict = {
        "order": order,             # try primary, then permitted backups, in order
        "allow_fallbacks": False,   # NEVER route to a provider outside this list
        "require_parameters": True,  # exclude providers that ignore our sampling params
    }
    quant = os.environ.get("OPENROUTER_PROVIDER_QUANT", "").strip()
    if quant:
        provider["quantizations"] = [quant]
    return {"provider": provider}


def _assert_provider(resp, model: str) -> None:
    """Verify the served provider is in our allow-list; warn loudly on fallback.

    - primary (order[0])         → OK, silent
    - a permitted backup         → OK but LOUD warning + fallback counter bumped
    - anything outside the list  → ProviderPinError (real drift, abort)
    """
    pin = openrouter_provider_pin(model)
    if not pin:
        return
    order = [p.lower() for p in pin["provider"]["order"]]
    primary = order[0]
    used = getattr(resp, "provider", None) or (
        (getattr(resp, "model_extra", None) or {}).get("provider")
    )
    if used is None:
        raise ProviderPinError(
            f"pinned model {model!r} but response had no provider field to verify "
            f"(expected {primary!r})"
        )
    used_l = str(used).strip().lower()
    if used_l == primary:
        return
    if used_l in order:
        global _provider_fallback_count
        _provider_fallback_count += 1
        print(
            f"⚠️ PROVIDER FALLBACK: {model} served by {used!r} "
            f"(primary {pin['provider']['order'][0]!r} unavailable) — "
            f"result from a backup serving config [fallbacks so far: "
            f"{_provider_fallback_count}]",
            file=__import__("sys").stderr,
            flush=True,
        )
        return
    raise ProviderPinError(
        f"provider drift: {model!r} served by {used!r}, outside allow-list "
        f"{pin['provider']['order']} — aborting to avoid contaminated data"
    )


def _chat_with_retry(client, *, max_retries: int = 5, **kwargs):
    """Retry transient API failures (429/5xx/connection) with exponential backoff.

    A single throttled call must not kill a multi-hundred-call pipeline run.
    For a provider-pinned model, injects the `provider` routing object into every
    attempt (so retries reuse the SAME provider, never a fallback) and asserts the
    served provider matches the pin.
    """
    model = kwargs.get("model", "")
    pin = openrouter_provider_pin(model)
    if pin:
        # Re-apply on every call so a caller that dropped extra_body (e.g. an
        # empty-content retry) can never fall off the pinned provider.
        extra = dict(kwargs.get("extra_body") or {})
        extra.update(pin)
        kwargs["extra_body"] = extra

    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(**kwargs)
            _assert_provider(resp, model)
            return resp
        except ProviderPinError:
            raise  # drift is never retryable — abort loudly
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            retryable = status in _RETRYABLE_STATUS or exc.__class__.__name__ in (
                "APIConnectionError",
                "APITimeoutError",
            )
            if not retryable or attempt == max_retries:
                raise
            delay = min(2.0 * 2**attempt, 60.0) + random.uniform(0, 1)
            print(
                f"  [retry] {status or exc.__class__.__name__} — attempt "
                f"{attempt + 1}/{max_retries}, sleeping {delay:.1f}s "
                f"(same provider, no fallback)",
                flush=True,
            )
            time.sleep(delay)

_MINIMAX_BASE_URL = "https://api.minimax.io/v1"
_PRIME_BASE_URL = "https://api.pinference.ai/api/v1"
_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

_client: OpenAI | None = None


class MissingCredentialsError(RuntimeError):
    """Raised when required API credentials are missing — fail fast instead of
    emitting error-SQL telemetry that silently pollutes the drift stream."""


def _base_url() -> str:
    return os.environ.get("AGENT_BASE_URL", _MINIMAX_BASE_URL)


def _is_local() -> bool:
    url = _base_url()
    return "localhost" in url or "127.0.0.1" in url


def _is_prime() -> bool:
    url = _base_url().lower()
    return "pinference.ai" in url or "primeintellect" in url


def _is_openrouter() -> bool:
    return "openrouter.ai" in _base_url().lower()


def _api_key() -> str:
    """Pick the credential that matches the configured student endpoint."""
    if _is_local():
        return (
            os.environ.get("OPENROUTER_API_KEY")
            or os.environ.get("PRIME_API_KEY")
            or os.environ.get("MINIMAX_API_KEY")
            or "ollama"
        )
    if _is_openrouter():
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise MissingCredentialsError(
                "OPENROUTER_API_KEY is not set. Add it to .env or export it:\n"
                "    export OPENROUTER_API_KEY=...\n"
                "    export AGENT_BASE_URL=https://openrouter.ai/api/v1\n"
                "    export AGENT_MODEL=<provider/model-id>"
            )
        return key
    if _is_prime():
        key = os.environ.get("PRIME_API_KEY") or os.environ.get("PRIME_INTELLECT_API_KEY")
        if not key:
            raise MissingCredentialsError(
                "PRIME_API_KEY is not set. Add it to .env or export it:\n"
                "    export PRIME_API_KEY=...\n"
                "    export AGENT_BASE_URL=https://api.pinference.ai/api/v1\n"
                "    export AGENT_MODEL=<cheap-model-id>"
            )
        return key
    key = os.environ.get("MINIMAX_API_KEY")
    if not key:
        raise MissingCredentialsError(
            "MINIMAX_API_KEY is not set. Export it in THIS shell before running:\n"
            "    export MINIMAX_API_KEY=sk-...\n"
            "Or use OpenRouter:\n"
            "    export AGENT_BASE_URL=https://openrouter.ai/api/v1\n"
            "    export OPENROUTER_API_KEY=...\n"
            "Or use Prime Intellect:\n"
            "    export AGENT_BASE_URL=https://api.pinference.ai/api/v1\n"
            "    export PRIME_API_KEY=...\n"
            "Or run the student locally:\n"
            "    export AGENT_BASE_URL=http://localhost:11434/v1"
        )
    return key


def require_api_key() -> None:
    """Call at startup so a missing key stops the run loudly, before any
    telemetry is written. Without this, every run becomes a fake '-- error'
    record that the detector / eval can mis-score.

    Skipped for local providers (AGENT_BASE_URL pointing at localhost).
    """
    if _is_local():
        return
    _api_key()  # raises MissingCredentialsError if missing


def _prime_team_headers() -> dict[str, str]:
    """Optional team billing header. Personal accounts leave PRIME_TEAM_ID unset."""
    team = (os.environ.get("PRIME_TEAM_ID") or os.environ.get("PRIME_TEAM") or "").strip()
    if team:
        return {"X-Prime-Team-ID": team}
    return {}


def _client_kwargs() -> dict:
    kwargs: dict = {
        "api_key": _api_key(),
        "base_url": _base_url(),
        # Avoid indefinite hangs on flaky inference endpoints (probe/eval).
        "timeout": float(os.environ.get("AGENT_TIMEOUT_S", "90")),
    }
    headers: dict[str, str] = {}
    if _is_openrouter():
        # Optional OpenRouter ranking headers (safe defaults).
        referer = os.environ.get("OPENROUTER_HTTP_REFERER", "").strip()
        title = os.environ.get("OPENROUTER_APP_TITLE", "agent-self-improvement").strip()
        if referer:
            headers["HTTP-Referer"] = referer
        if title:
            headers["X-Title"] = title
    if _is_prime():
        headers.update(_prime_team_headers())
    if headers:
        kwargs["default_headers"] = headers
    return kwargs


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        require_api_key()
        _client = OpenAI(**_client_kwargs())
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
            [e for e in examples if not e.domain_id or e.domain_id == db_id]
            if db_id else examples
        )
        if relevant:
            shots = "\n\n".join(
                f"Q: {e.question}\nSQL: {e.correct_output}" for e in relevant[:16]
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
