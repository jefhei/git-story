"""OpenAI backend adapter — implements :class:`LLMBackend` for OpenAI models.

Uses the official ``openai`` SDK to call the chat completions endpoint.
Supports both plain-text and JSON-mode responses.

Token counting is handled via ``tiktoken`` when available, falling back
to a ~4-char-per-token heuristic.

Example::

    from git_story.backends.openai import OpenAIBackend

    backend = OpenAIBackend(api_key=\"sk-...\")
    resp = backend.complete(LLMRequest(
        system_prompt=\"You are a helpful assistant.\",
        user_prompt=\"Summarize this diff...\",
        model=\"gpt-4o\",
        response_format=\"json\",
    ))
"""

from __future__ import annotations

from typing import Dict, Optional

from openai import APIError, APIStatusError, AuthenticationError, BadRequestError, RateLimitError, OpenAI

from git_story.backends import register_backend
from git_story.llm import (
    LLMAuthenticationError,
    LLMBackend,
    LLMContextLengthError,
    LLMError,
    LLMRateLimitError,
    LLMRequest,
    LLMResponse,
    TokenUsage,
)

# ── TikToken (optional, with graceful fallback) ───────────────────────────────

_ENCODING_CACHE: Dict[str, object] = {}


def _get_encoding(model: str) -> object | None:
    """Return a tiktoken encoding for the given model, or ``None``."""
    if model in _ENCODING_CACHE:
        return _ENCODING_CACHE[model]

    try:
        import tiktoken
    except ImportError:
        _ENCODING_CACHE[model] = None
        return None

    try:
        encoding = tiktoken.encoding_for_model(model)
        _ENCODING_CACHE[model] = encoding
        return encoding
    except KeyError:
        # Unknown model — try cl100k_base as a fallback (used by GPT-4, GPT-3.5).
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            _ENCODING_CACHE[model] = encoding
            return encoding
        except Exception:
            _ENCODING_CACHE[model] = None
            return None


def _count_tokens_tiktoken(text: str, model: str) -> int | None:
    """Count tokens using tiktoken, returning ``None`` on failure."""
    encoding = _get_encoding(model)
    if encoding is None:
        return None
    try:
        return len(encoding.encode(text))
    except Exception:
        return None


# ── Backend implementation ────────────────────────────────────────────────────


class OpenAIBackend(LLMBackend):
    """LLM backend for OpenAI models.

    Args:
        api_key: OpenAI API key. If ``None``, the ``OPENAI_API_KEY``
            environment variable is used (via the OpenAI SDK default).
        organization: Optional OpenAI organization ID.
        base_url: Optional custom API base URL (for proxies or Azure).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        self._client = OpenAI(
            api_key=api_key,
            organization=organization,
            base_url=base_url,
        )

    @property
    def name(self) -> str:
        """Human-readable backend name."""
        return "openai"

    def complete(self, request: LLMRequest) -> LLMResponse:
        """Send a completion request to the OpenAI chat completions API.

        Args:
            request: The populated request parameters.

        Returns:
            The API response wrapped in an :class:`LLMResponse`.

        Raises:
            LLMAuthenticationError: If the API key is invalid.
            LLMRateLimitError: If the API rate limit is exceeded.
            LLMContextLengthError: If the prompt exceeds the context window.
            LLMError: For any other API or network failure.
        """
        kwargs: dict = {
            "model": request.model,
            "messages": [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_prompt},
            ],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }

        if request.response_format == "json":
            kwargs["response_format"] = {"type": "json_object"}

        try:
            response = self._client.chat.completions.create(**kwargs)
        except AuthenticationError as exc:
            raise LLMAuthenticationError(
                f"OpenAI authentication failed: {exc}"
            ) from exc
        except RateLimitError as exc:
            raise LLMRateLimitError(
                f"OpenAI rate limit exceeded: {exc}"
            ) from exc
        except BadRequestError as exc:
            message = str(exc).lower()
            if "context_length_exceeded" in message or "maximum context" in message:
                raise LLMContextLengthError(
                    f"OpenAI context length exceeded: {exc}"
                ) from exc
            raise LLMError(f"OpenAI bad request: {exc}") from exc
        except APIStatusError as exc:
            # Catch-all for other status errors (server errors, etc.)
            raise LLMError(f"OpenAI API error (HTTP {exc.status_code}): {exc}") from exc
        except APIError as exc:
            # Network / connection / timeout errors
            raise LLMError(f"OpenAI API request failed: {exc}") from exc
        except Exception as exc:
            # Anything else (including SDK-internal errors)
            raise LLMError(f"Unexpected OpenAI error: {exc}") from exc

        # Extract the text response
        choice = response.choices[0]
        content = choice.message.content or ""

        # Extract usage
        usage: Optional[TokenUsage] = None
        if response.usage is not None:
            usage = TokenUsage(
                prompt_tokens=response.usage.prompt_tokens or 0,
                completion_tokens=response.usage.completion_tokens or 0,
                total_tokens=response.usage.total_tokens or 0,
            )

        return LLMResponse(
            content=content,
            model=response.model,
            usage=usage,
        )

    def count_tokens(self, text: str) -> int:
        """Estimate the token count of *text*.

        Uses tiktoken (for accurate model-specific counts) when available,
        falling back to a ~4-char-per-token heuristic.
        """
        # Use the default model if we don't have a specific one to key on.
        # ``gpt-4o`` uses the ``o200k_base`` encoding which is a reasonable
        # default for most modern OpenAI models.
        tik_count = _count_tokens_tiktoken(text, "gpt-4o")
        if tik_count is not None:
            return tik_count
        # Fallback heuristic: ~4 chars per token
        return max(1, len(text) // 4)


# ── Auto-register ─────────────────────────────────────────────────────────────

register_backend("openai", OpenAIBackend)
