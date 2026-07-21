"""Shared OpenAI-compatible client resolution for teacher / distill calls.

Student hot-path stays in harness/agent.py. Correction-side models (teacher,
distill, repair) resolve here so domains can point teacher at Prime while the
student stays on a cheap model — or fall back to MiniMax when credited.
"""
from __future__ import annotations

import os

from openai import OpenAI

_MINIMAX_BASE = "https://api.minimax.io/v1"
_PRIME_BASE = "https://api.pinference.ai/api/v1"
_OPENROUTER_BASE = "https://openrouter.ai/api/v1"
_DEFAULT_MINIMAX_TEACHER = "MiniMax-M3"
_DEFAULT_PRIME_TEACHER = "minimax/minimax-m2.5"
_DEFAULT_OPENROUTER_TEACHER = "qwen/qwen3-coder"


def _default_model_for_base(base: str, model: str) -> str:
    if model:
        return model
    b = base.lower()
    if "openrouter.ai" in b:
        return _DEFAULT_OPENROUTER_TEACHER
    if "pinference" in b or "primeintellect" in b:
        return _DEFAULT_PRIME_TEACHER
    return _DEFAULT_MINIMAX_TEACHER


def _prime_team_headers() -> dict[str, str]:
    team = (os.environ.get("PRIME_TEAM_ID") or os.environ.get("PRIME_TEAM") or "").strip()
    return {"X-Prime-Team-ID": team} if team else {}


def _openrouter_client(api_key: str, base_url: str = _OPENROUTER_BASE) -> OpenAI:
    headers = {}
    referer = os.environ.get("OPENROUTER_HTTP_REFERER", "").strip()
    title = os.environ.get("OPENROUTER_APP_TITLE", "agent-self-improvement").strip()
    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-Title"] = title
    kwargs: dict = {"api_key": api_key, "base_url": base_url}
    if headers:
        kwargs["default_headers"] = headers
    return OpenAI(**kwargs)


def _prime_client(api_key: str, base_url: str = _PRIME_BASE) -> OpenAI:
    headers = _prime_team_headers()
    kwargs: dict = {"api_key": api_key, "base_url": base_url}
    if headers:
        kwargs["default_headers"] = headers
    return OpenAI(**kwargs)


def teacher_client_and_model() -> tuple[OpenAI, str]:
    """Return (client, model_id) for teacher generation.

    Resolution order:
      1. TEACHER_BASE_URL + TEACHER_API_KEY (explicit override)
      2. OpenRouter when TEACHER_USE_OPENROUTER=1 or TEACHER_BASE_URL is OpenRouter
      3. Prime (PRIME_API_KEY) when TEACHER_USE_PRIME=1 or TEACHER_BASE_URL is Prime
      4. MiniMax via MINIMAX_API_KEY
    """
    explicit_base = (os.environ.get("TEACHER_BASE_URL") or "").strip()
    explicit_key = (os.environ.get("TEACHER_API_KEY") or "").strip()
    model = (os.environ.get("TEACHER_MODEL") or "").strip()

    if explicit_base and explicit_key:
        if "openrouter.ai" in explicit_base.lower():
            return _openrouter_client(explicit_key, explicit_base), _default_model_for_base(
                explicit_base, model
            )
        if "pinference" in explicit_base.lower() or "primeintellect" in explicit_base.lower():
            return _prime_client(explicit_key, explicit_base), _default_model_for_base(
                explicit_base, model
            )
        return OpenAI(api_key=explicit_key, base_url=explicit_base), _default_model_for_base(
            explicit_base, model
        )

    use_openrouter = (
        os.environ.get("TEACHER_USE_OPENROUTER", "").strip() in ("1", "true", "yes")
        or "openrouter.ai" in explicit_base.lower()
    )
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    if use_openrouter and openrouter_key:
        return _openrouter_client(openrouter_key), (
            model or _DEFAULT_OPENROUTER_TEACHER
        )

    use_prime = (
        os.environ.get("TEACHER_USE_PRIME", "").strip() in ("1", "true", "yes")
        or "pinference" in explicit_base.lower()
    )
    prime_key = os.environ.get("PRIME_API_KEY") or os.environ.get("PRIME_INTELLECT_API_KEY")
    if use_prime and prime_key:
        return _prime_client(prime_key), (model or _DEFAULT_PRIME_TEACHER)

    key = os.environ.get("MINIMAX_API_KEY")
    if not key:
        raise RuntimeError(
            "No teacher credentials. Set TEACHER_BASE_URL+TEACHER_API_KEY, "
            "or TEACHER_USE_OPENROUTER=1 with OPENROUTER_API_KEY, "
            "or TEACHER_USE_PRIME=1 with PRIME_API_KEY, or MINIMAX_API_KEY."
        )
    return OpenAI(api_key=key, base_url=_MINIMAX_BASE), (
        model or _DEFAULT_MINIMAX_TEACHER
    )
