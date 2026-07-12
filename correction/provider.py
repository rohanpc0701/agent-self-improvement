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
_DEFAULT_MINIMAX_TEACHER = "MiniMax-M3"
_DEFAULT_PRIME_TEACHER = "minimax/minimax-m2.5"


def teacher_client_and_model() -> tuple[OpenAI, str]:
    """Return (client, model_id) for teacher generation.

    Resolution order:
      1. TEACHER_BASE_URL + TEACHER_API_KEY (explicit override)
      2. Prime (PRIME_API_KEY) when TEACHER_USE_PRIME=1 or TEACHER_BASE_URL is Prime
      3. MiniMax via MINIMAX_API_KEY
    """
    explicit_base = (os.environ.get("TEACHER_BASE_URL") or "").strip()
    explicit_key = (os.environ.get("TEACHER_API_KEY") or "").strip()
    model = (os.environ.get("TEACHER_MODEL") or "").strip()

    if explicit_base and explicit_key:
        return OpenAI(api_key=explicit_key, base_url=explicit_base), (
            model or _DEFAULT_PRIME_TEACHER
            if "pinference" in explicit_base
            else model or _DEFAULT_MINIMAX_TEACHER
        )

    use_prime = (
        os.environ.get("TEACHER_USE_PRIME", "").strip() in ("1", "true", "yes")
        or "pinference" in explicit_base.lower()
    )
    prime_key = os.environ.get("PRIME_API_KEY") or os.environ.get("PRIME_INTELLECT_API_KEY")
    if use_prime and prime_key:
        return OpenAI(api_key=prime_key, base_url=_PRIME_BASE), (
            model or _DEFAULT_PRIME_TEACHER
        )

    key = os.environ.get("MINIMAX_API_KEY")
    if not key:
        raise RuntimeError(
            "No teacher credentials. Set TEACHER_BASE_URL+TEACHER_API_KEY, "
            "or TEACHER_USE_PRIME=1 with PRIME_API_KEY, or MINIMAX_API_KEY."
        )
    return OpenAI(api_key=key, base_url=_MINIMAX_BASE), (
        model or _DEFAULT_MINIMAX_TEACHER
    )
