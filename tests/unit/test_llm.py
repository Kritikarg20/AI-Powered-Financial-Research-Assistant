"""Unit tests for fin_ai.core.llm — error handling around Mistral/LiteLLM."""

from __future__ import annotations

import sys
import types

import pytest

from fin_ai.core import llm as llm_module
from fin_ai.core.exceptions import LLMError


def test_chat_raises_when_api_key_missing(monkeypatch):
    monkeypatch.setattr(llm_module, "MISTRAL_API_KEY", "")
    with pytest.raises(LLMError, match="not configured"):
        llm_module.chat([{"role": "user", "content": "hi"}])


def test_is_configured_reflects_api_key(monkeypatch):
    monkeypatch.setattr(llm_module, "MISTRAL_API_KEY", "")
    assert llm_module.is_configured() is False
    monkeypatch.setattr(llm_module, "MISTRAL_API_KEY", "sk-test")
    assert llm_module.is_configured() is True


class _FakeLiteLLMModule(types.SimpleNamespace):
    """Minimal fake of the litellm module for testing chat()."""


def _install_fake_litellm(monkeypatch, completion_fn, exceptions_ns=None):
    fake = _FakeLiteLLMModule()
    fake.completion = completion_fn

    class _Exceptions:
        class AuthenticationError(Exception):
            pass

        class PermissionDeniedError(Exception):
            pass

        class NotFoundError(Exception):
            pass

        class BadRequestError(Exception):
            pass

        class ContextWindowExceededError(BadRequestError):
            pass

        class RateLimitError(Exception):
            pass

        class Timeout(Exception):
            pass

        class APIConnectionError(Exception):
            pass

        class ServiceUnavailableError(Exception):
            pass

        class InternalServerError(Exception):
            pass

        class BadGatewayError(Exception):
            pass

    fake.exceptions = exceptions_ns or _Exceptions
    monkeypatch.setitem(sys.modules, "litellm", fake)
    return fake


def test_chat_success_returns_content(monkeypatch):
    monkeypatch.setattr(llm_module, "MISTRAL_API_KEY", "sk-test")

    def fake_completion(**kwargs):
        return {"choices": [{"message": {"content": "Revenue grew 10%."}}]}

    _install_fake_litellm(monkeypatch, fake_completion)
    result = llm_module.chat([{"role": "user", "content": "Summarize revenue."}])
    assert result == "Revenue grew 10%."


def test_chat_maps_authentication_error(monkeypatch):
    monkeypatch.setattr(llm_module, "MISTRAL_API_KEY", "sk-bad")
    fake = _install_fake_litellm(monkeypatch, None)

    def fake_completion(**kwargs):
        raise fake.exceptions.AuthenticationError("bad key")

    fake.completion = fake_completion
    with pytest.raises(LLMError, match="rejected the API key"):
        llm_module.chat([{"role": "user", "content": "hi"}])


def test_chat_maps_rate_limit_error(monkeypatch):
    monkeypatch.setattr(llm_module, "MISTRAL_API_KEY", "sk-test")
    fake = _install_fake_litellm(monkeypatch, None)

    def fake_completion(**kwargs):
        raise fake.exceptions.RateLimitError("slow down")

    fake.completion = fake_completion
    with pytest.raises(LLMError, match="rate limit"):
        llm_module.chat([{"role": "user", "content": "hi"}])


def test_chat_maps_permission_denied_error(monkeypatch):
    monkeypatch.setattr(llm_module, "MISTRAL_API_KEY", "sk-test")
    fake = _install_fake_litellm(monkeypatch, None)

    def fake_completion(**kwargs):
        raise fake.exceptions.PermissionDeniedError("no access to this model")

    fake.completion = fake_completion
    with pytest.raises(LLMError, match="does not have access"):
        llm_module.chat([{"role": "user", "content": "hi"}])


def test_chat_maps_not_found_error(monkeypatch):
    monkeypatch.setattr(llm_module, "MISTRAL_API_KEY", "sk-test")
    fake = _install_fake_litellm(monkeypatch, None)

    def fake_completion(**kwargs):
        raise fake.exceptions.NotFoundError("unknown model")

    fake.completion = fake_completion
    with pytest.raises(LLMError, match="does not recognize the model"):
        llm_module.chat([{"role": "user", "content": "hi"}])


def test_chat_maps_context_window_exceeded_error(monkeypatch):
    """ContextWindowExceededError is a BadRequestError subclass in real
    litellm; this checks the more specific handler fires, not the generic
    BadRequestError one below it."""
    monkeypatch.setattr(llm_module, "MISTRAL_API_KEY", "sk-test")
    fake = _install_fake_litellm(monkeypatch, None)

    def fake_completion(**kwargs):
        raise fake.exceptions.ContextWindowExceededError("too many tokens")

    fake.completion = fake_completion
    with pytest.raises(LLMError, match="too long for Mistral AI's context window"):
        llm_module.chat([{"role": "user", "content": "hi"}])


def test_chat_maps_bad_request_error(monkeypatch):
    monkeypatch.setattr(llm_module, "MISTRAL_API_KEY", "sk-test")
    fake = _install_fake_litellm(monkeypatch, None)

    def fake_completion(**kwargs):
        raise fake.exceptions.BadRequestError("malformed payload")

    fake.completion = fake_completion
    with pytest.raises(LLMError, match="rejected the request as malformed"):
        llm_module.chat([{"role": "user", "content": "hi"}])


@pytest.mark.parametrize(
    "exc_name",
    ["ServiceUnavailableError", "InternalServerError", "BadGatewayError"],
)
def test_chat_maps_provider_outage_errors(monkeypatch, exc_name):
    monkeypatch.setattr(llm_module, "MISTRAL_API_KEY", "sk-test")
    fake = _install_fake_litellm(monkeypatch, None)

    def fake_completion(**kwargs):
        raise getattr(fake.exceptions, exc_name)("down for maintenance")

    fake.completion = fake_completion
    with pytest.raises(LLMError, match="temporarily unavailable"):
        llm_module.chat([{"role": "user", "content": "hi"}])


def test_chat_maps_generic_error_without_traceback_leak(monkeypatch):
    monkeypatch.setattr(llm_module, "MISTRAL_API_KEY", "sk-test")

    def fake_completion(**kwargs):
        raise ValueError("some internal detail that should not reach the user")

    _install_fake_litellm(monkeypatch, fake_completion)
    with pytest.raises(LLMError) as exc_info:
        llm_module.chat([{"role": "user", "content": "hi"}])
    message = str(exc_info.value).lower()
    assert "could not process the request" in message
    # The exception's own class name should now be surfaced on screen (not
    # just in the log) so an unanticipated failure is diagnosable from the
    # UI alone.
    assert "valueerror" in message


def test_chat_raises_on_empty_response(monkeypatch):
    monkeypatch.setattr(llm_module, "MISTRAL_API_KEY", "sk-test")

    def fake_completion(**kwargs):
        return {"choices": [{"message": {"content": "   "}}]}

    _install_fake_litellm(monkeypatch, fake_completion)
    with pytest.raises(LLMError, match="empty response"):
        llm_module.chat([{"role": "user", "content": "hi"}])


def test_chat_passes_mistral_prefixed_model_to_litellm(monkeypatch):
    """Regression test for a real bug: litellm.completion() requires a
    "<provider>/<model>" string (e.g. "mistral/mistral-large-latest").
    A bare model name like "mistral-large-latest" makes the REAL litellm
    raise BadRequestError("LLM Provider NOT provided") before any network
    call is made — every fake_completion() test above uses a fully faked
    litellm module, so none of them would have caught this. This test
    inspects the actual `model=` kwarg chat() builds, so it fails if that
    prefix is ever dropped again.
    """
    monkeypatch.setattr(llm_module, "MISTRAL_API_KEY", "sk-test")
    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return {"choices": [{"message": {"content": "ok"}}]}

    _install_fake_litellm(monkeypatch, fake_completion)
    llm_module.chat([{"role": "user", "content": "hi"}])
    assert captured["model"].startswith("mistral/"), (
        f"model={captured['model']!r} is missing the LiteLLM provider "
        "prefix; the real litellm.completion() would raise "
        "BadRequestError for this value."
    )


def test_litellm_model_resolves_with_real_litellm():
    """Boundary check with the REAL litellm (no fake module installed):
    proves LITELLM_MODEL is a string litellm.get_llm_provider() actually
    accepts. This is the exact call that failed in production before the
    "mistral/" prefix fix — it needs no network access or API key.
    """
    litellm = pytest.importorskip("litellm")
    from fin_ai.config.fin_ai import LITELLM_MODEL

    model, provider, _key, _base = litellm.get_llm_provider(LITELLM_MODEL)
    assert provider == "mistral"
