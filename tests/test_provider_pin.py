"""Hermetic tests for OpenRouter single-provider pinning (reproducibility)."""
from __future__ import annotations

import pytest

from harness import agent
from harness.agent import (
    ProviderPinError,
    _chat_with_retry,
    openrouter_provider_pin,
    provider_fallback_count,
    reset_provider_fallback_count,
)

OR = "https://openrouter.ai/api/v1"
PINNED = "deepseek/deepseek-v4-pro"


@pytest.fixture(autouse=True)
def _openrouter_env(monkeypatch):
    monkeypatch.setenv("AGENT_BASE_URL", OR)
    monkeypatch.setenv("OPENROUTER_PIN_MODEL", PINNED)
    monkeypatch.setenv("OPENROUTER_PROVIDER_ORDER", "fireworks,together")
    monkeypatch.delenv("OPENROUTER_PROVIDER_QUANT", raising=False)
    reset_provider_fallback_count()


class _Resp:
    def __init__(self, provider):
        self.provider = provider
        self.choices = []


class _Client:
    """Fake OpenAI client capturing the request and returning a chosen provider."""

    def __init__(self, served_provider, record):
        self._served = served_provider
        self._record = record

        class _Completions:
            def create(_self, **kw):
                record.append(kw)
                return _Resp(self._served)

        class _Chat:
            completions = _Completions()

        self.chat = _Chat()


def test_pin_block_shape():
    pin = openrouter_provider_pin(PINNED)
    assert pin == {
        "provider": {
            "order": ["fireworks", "together"],
            "allow_fallbacks": False,
            "require_parameters": True,
        }
    }


def test_pin_only_for_target_model():
    assert openrouter_provider_pin("openai/gpt-5.2") == {}
    assert openrouter_provider_pin(PINNED) != {}


def test_quant_pinned_when_env_set(monkeypatch):
    monkeypatch.setenv("OPENROUTER_PROVIDER_QUANT", "fp8")
    assert openrouter_provider_pin(PINNED)["provider"]["quantizations"] == ["fp8"]


def test_pin_injected_into_request():
    record = []
    client = _Client("fireworks", record)
    _chat_with_retry(client, model=PINNED, messages=[])
    assert record[0]["extra_body"]["provider"]["order"] == ["fireworks", "together"]
    assert record[0]["extra_body"]["provider"]["allow_fallbacks"] is False


def test_pin_reapplied_even_if_caller_clears_extra_body():
    record = []
    client = _Client("fireworks", record)
    # caller passes extra_body without provider (e.g. an empty-content retry)
    _chat_with_retry(client, model=PINNED, messages=[], extra_body={"reasoning": {"enabled": False}})
    eb = record[0]["extra_body"]
    assert eb["provider"]["order"] == ["fireworks", "together"]  # pin re-added
    assert eb["reasoning"] == {"enabled": False}                 # caller's field preserved


def test_primary_provider_silent_no_fallback_count():
    client = _Client("fireworks", [])
    _chat_with_retry(client, model=PINNED, messages=[])
    assert provider_fallback_count() == 0


def test_backup_provider_warns_and_counts_not_raises(capsys):
    client = _Client("together", [])  # permitted backup
    _chat_with_retry(client, model=PINNED, messages=[])  # must NOT raise
    assert provider_fallback_count() == 1
    assert "PROVIDER FALLBACK" in capsys.readouterr().err


def test_provider_outside_allowlist_raises():
    client = _Client("deepinfra", [])  # NOT in fireworks,together
    with pytest.raises(ProviderPinError, match="outside allow-list"):
        _chat_with_retry(client, model=PINNED, messages=[])


def test_missing_provider_field_raises():
    client = _Client(None, [])  # OpenRouter didn't report a provider
    with pytest.raises(ProviderPinError, match="no provider field"):
        _chat_with_retry(client, model=PINNED, messages=[])


def test_non_pinned_model_not_asserted():
    # A different model served by anyone must not trigger the pin assertion.
    client = _Client("anything", [])
    resp = _chat_with_retry(client, model="openai/gpt-5.2", messages=[])
    assert resp.provider == "anything"
