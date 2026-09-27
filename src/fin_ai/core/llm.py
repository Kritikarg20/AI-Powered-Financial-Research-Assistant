"""
LLM interface — Mistral AI via LiteLLM.

Mistral is the single primary LLM provider for this project. LiteLLM gives
a uniform ``completion()`` call and normalizes provider-specific errors,
which this module converts into friendly, actionable messages so the
Streamlit UI never has to show a raw traceback.
"""

from __future__ import annotations

import logging

from fin_ai.config.fin_ai import LITELLM_MODEL, MISTRAL_API_KEY, MISTRAL_BASE_URL
from fin_ai.core.exceptions import LLMError

logger = logging.getLogger(__name__)


def is_configured() -> bool:
    """Whether a Mistral API key is present in the environment."""
    return bool(MISTRAL_API_KEY)


def chat(
    messages: list[dict],
    *,
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 2000,
) -> str:
    """Send a chat completion request to Mistral via LiteLLM.

    Returns the assistant's text response. Raises :class:`LLMError` with a
    user-friendly message on any failure (missing key, timeout, rate
    limit, malformed response, etc.) — never a raw exception.
    """
    if not MISTRAL_API_KEY:
        raise LLMError(
            "Mistral API key is not configured. Set MISTRAL_API_KEY in your "
            ".env file (see .env.example), then restart the app."
        )

    try:
        import litellm
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise LLMError(
            "The 'litellm' package is required. Install project dependencies "
            "with `pip install -r requirements.txt`."
        ) from exc

    resolved_model = model or LITELLM_MODEL
    if not resolved_model.startswith("mistral/"):
        # Guard against a caller-supplied `model=` without the LiteLLM
        # provider prefix hitting the same BadRequestError.
        resolved_model = f"mistral/{resolved_model}"

    kwargs = dict(
        model=resolved_model,
        messages=messages,
        api_key=MISTRAL_API_KEY,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=60,
    )
    if MISTRAL_BASE_URL:
        kwargs["api_base"] = MISTRAL_BASE_URL

    try:
        response = litellm.completion(**kwargs)
    except litellm.exceptions.AuthenticationError as exc:
        raise LLMError(
            "Mistral AI rejected the API key. Please check MISTRAL_API_KEY in "
            "your .env file."
        ) from exc
    except litellm.exceptions.PermissionDeniedError as exc:
        raise LLMError(
            f"Mistral AI rejected the request for model '{resolved_model}': "
            "your API key does not have access to it. Check your Mistral "
            "account/plan or try a different MISTRAL_MODEL."
        ) from exc
    except litellm.exceptions.NotFoundError as exc:
        raise LLMError(
            f"Mistral AI does not recognize the model '{resolved_model}'. "
            "Check MISTRAL_MODEL in your .env file against the model names "
            "listed at https://docs.mistral.ai/getting-started/models/."
        ) from exc
    except litellm.exceptions.ContextWindowExceededError as exc:
        raise LLMError(
            "The request was too long for Mistral AI's context window. Try "
            "asking about fewer documents at once or a shorter question."
        ) from exc
    except litellm.exceptions.BadRequestError as exc:
        logger.exception("Mistral AI rejected the request as malformed")
        raise LLMError(
            f"Mistral AI rejected the request as malformed: {exc}"
        ) from exc
    except litellm.exceptions.RateLimitError as exc:
        raise LLMError(
            "Mistral AI rate limit reached. Please wait a moment and try again."
        ) from exc
    except litellm.exceptions.Timeout as exc:
        raise LLMError(
            "The request to Mistral AI timed out. Please try again."
        ) from exc
    except litellm.exceptions.APIConnectionError as exc:
        raise LLMError(
            "Could not reach the Mistral AI API. Please check your internet "
            "connection and try again."
        ) from exc
    except (
        litellm.exceptions.ServiceUnavailableError,
        litellm.exceptions.InternalServerError,
        litellm.exceptions.BadGatewayError,
    ) as exc:
        raise LLMError(
            "Mistral AI's service is temporarily unavailable. Please try "
            "again in a moment."
        ) from exc
    except Exception as exc:
        # A genuinely unanticipated failure. Log the full traceback (see
        # logs/fin_ai.log) and surface the exception's own class/message on
        # screen too, so this is diagnosable from the UI alone rather than
        # a dead-end "check your configuration" with no next step.
        logger.exception("Unexpected error calling Mistral via LiteLLM")
        raise LLMError(
            f"Mistral AI could not process the request ({type(exc).__name__}: "
            f"{exc}). See logs/fin_ai.log for the full error."
        ) from exc

    try:
        content = response["choices"][0]["message"]["content"]
    except Exception as exc:
        raise LLMError("Mistral AI returned an unexpected/malformed response.") from exc

    if not content or not content.strip():
        raise LLMError("Mistral AI returned an empty response. Please try again.")

    return content.strip()
