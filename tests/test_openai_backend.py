"""Tests for the OpenAI backend adapter.

Tests cover:
- Backend construction and registration
- Token counting (tiktoken and fallback)
- Request construction (including JSON mode)
- Error mapping (auth, rate limit, context length, network)
- LLMResponse extraction (content, model, usage)
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional
from unittest.mock import ANY, MagicMock, PropertyMock, patch

import pytest

from git_story.backends import get_backend_class, list_backends
from git_story.backends.openai import OpenAIBackend
from git_story.llm import (
    LLMAuthenticationError,
    LLMContextLengthError,
    LLMError,
    LLMRateLimitError,
    LLMRequest,
    LLMResponse,
    TokenUsage,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def backend() -> OpenAIBackend:
    """An OpenAIBackend instance with a fake API key."""
    return OpenAIBackend(api_key="sk-test-fake-key")


@pytest.fixture
def minimal_request() -> LLMRequest:
    """A minimal LLMRequest for testing."""
    return LLMRequest(
        system_prompt="You are a helpful assistant.",
        user_prompt="Summarize these git commits.",
        model="gpt-4o",
    )


# ── Mock OpenAI response helpers ──────────────────────────────────────────────


def _mock_openai_choice(content: str, finish_reason: str = "stop") -> MagicMock:
    """Create a mock OpenAI Choice."""
    message = MagicMock()
    message.content = content
    message.refusal = None
    choice = MagicMock()
    choice.message = message
    choice.finish_reason = finish_reason
    choice.index = 0
    return choice


def _mock_openai_usage(
    prompt_tokens: int = 50,
    completion_tokens: int = 100,
    total_tokens: int = 150,
) -> MagicMock:
    """Create a mock OpenAI CompletionUsage object."""
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    usage.total_tokens = total_tokens
    return usage


def _mock_openai_response(
    content: str = "Generated changelog.",
    model: str = "gpt-4o",
    usage: Optional[MagicMock] = None,
) -> MagicMock:
    """Create a mock OpenAI chat.completions.create response."""
    response = MagicMock(spec=["choices", "model", "usage"])
    response.choices = [_mock_openai_choice(content)]
    response.model = model
    response.usage = usage
    return response


# ── Construction and registration ─────────────────────────────────────────────


class TestOpenAIBackendConstruction:
    """OpenAIBackend construction and registration."""

    def test_register_openai(self) -> None:
        """OpenAI backend is registered under 'openai' key."""
        backends = list_backends()
        assert "openai" in backends
        assert backends["openai"] is OpenAIBackend

    def test_get_backend_class(self) -> None:
        """get_backend_class returns OpenAIBackend."""
        cls = get_backend_class("openai")
        assert cls is OpenAIBackend

    def test_constructor_defaults(self) -> None:
        """OpenAIBackend can be constructed with just an API key."""
        backend = OpenAIBackend(api_key="sk-test")
        assert backend.name == "openai"

    def test_constructor_with_org(self) -> None:
        """OpenAIBackend accepts optional org and base_url."""
        backend = OpenAIBackend(
            api_key="sk-test",
            organization="org-123",
            base_url="https://api.openai.com/v1",
        )
        assert backend._client.organization == "org-123"
        assert str(backend._client.base_url) == "https://api.openai.com/v1/"


# ── Properties ────────────────────────────────────────────────────────────────


class TestOpenAIBackendProperties:
    """OpenAIBackend property behaviour."""

    def test_name(self, backend: OpenAIBackend) -> None:
        """name returns 'openai'."""
        assert backend.name == "openai"

    def test_repr(self, backend: OpenAIBackend) -> None:
        """__repr__ includes class name and backend name."""
        rep = repr(backend)
        assert "OpenAIBackend" in rep
        assert "openai" in rep


# ── Token counting ────────────────────────────────────────────────────────────


class TestOpenAITokenCounting:
    """OpenAIBackend.count_tokens behaviour."""

    def test_count_tokens_returns_positive_int(self, backend: OpenAIBackend) -> None:
        """count_tokens returns a positive integer for non-empty text."""
        count = backend.count_tokens("Hello, world!")
        assert isinstance(count, int)
        assert count > 0

    def test_count_tokens_empty_string(self, backend: OpenAIBackend) -> None:
        """count_tokens handles empty string gracefully."""
        count = backend.count_tokens("")
        assert isinstance(count, int)
        assert count >= 0

    def test_count_tokens_larger_text(self, backend: OpenAIBackend) -> None:
        """Longer text produces more tokens."""
        small = backend.count_tokens("Short text.")
        large = backend.count_tokens("A much longer text that should have significantly more tokens than the short one.")
        assert large >= small

    def test_count_tokens_fallback_heuristic(self) -> None:
        """Fallback heuristic is used when tiktoken is unavailable."""
        with patch("git_story.backends.openai._count_tokens_tiktoken", return_value=None):
            backend = OpenAIBackend(api_key="sk-test")
            count = backend.count_tokens("Hello world")
            # Heuristic: len("Hello world") // 4 = 2
            assert count == 2


# ── Token estimation ──────────────────────────────────────────────────────────


class TestOpenAITokenEstimation:
    """estimate_request_tokens helper."""

    def test_estimate_returns_positive_int(self, backend: OpenAIBackend) -> None:
        """estimate_request_tokens returns a positive integer."""
        est = backend.estimate_request_tokens(
            system_prompt="Be concise.",
            user_prompt="Diff content here.",
            max_tokens=1024,
        )
        assert isinstance(est, int)
        assert est > 0

    def test_estimate_scales(self, backend: OpenAIBackend) -> None:
        """Larger prompts produce larger estimates."""
        small = backend.estimate_request_tokens("Short", "Small", max_tokens=512)
        large = backend.estimate_request_tokens(
            "A long system prompt with many words for testing purposes",
            "A very long user prompt that contains significantly more words for testing the scaling behaviour",
            max_tokens=512,
        )
        assert large > small


# ── Complete (success) ────────────────────────────────────────────────────────


class TestOpenAICompleteSuccess:
    """OpenAIBackend.complete — success cases."""

    def test_complete_returns_llm_response(self, backend: OpenAIBackend, minimal_request: LLMRequest) -> None:
        """complete() returns an LLMResponse."""
        mock_response = _mock_openai_response(
            content="Generated changelog content.",
            usage=_mock_openai_usage(50, 100, 150),
        )
        with patch.object(backend._client.chat.completions, "create", return_value=mock_response):
            resp = backend.complete(minimal_request)

        assert isinstance(resp, LLMResponse)
        assert resp.content == "Generated changelog content."
        assert resp.model == "gpt-4o"

    def test_complete_returns_usage(self, backend: OpenAIBackend, minimal_request: LLMRequest) -> None:
        """complete() returns token usage."""
        mock_response = _mock_openai_response(
            content="Content.",
            usage=_mock_openai_usage(prompt_tokens=55, completion_tokens=99, total_tokens=154),
        )
        with patch.object(backend._client.chat.completions, "create", return_value=mock_response):
            resp = backend.complete(minimal_request)

        assert resp.usage is not None
        assert resp.usage.prompt_tokens == 55
        assert resp.usage.completion_tokens == 99
        assert resp.usage.total_tokens == 154

    def test_complete_no_usage(self, backend: OpenAIBackend, minimal_request: LLMRequest) -> None:
        """complete() handles response with no usage data."""
        mock_response = _mock_openai_response(content="Content.", usage=None)
        with patch.object(backend._client.chat.completions, "create", return_value=mock_response):
            resp = backend.complete(minimal_request)
        assert resp.usage is None

    def test_complete_empty_content(self, backend: OpenAIBackend, minimal_request: LLMRequest) -> None:
        """complete() handles empty content."""
        mock_response = _mock_openai_response(content="", usage=_mock_openai_usage())
        with patch.object(backend._client.chat.completions, "create", return_value=mock_response):
            resp = backend.complete(minimal_request)
        assert resp.content == ""

    def test_complete_sends_correct_params(self, backend: OpenAIBackend) -> None:
        """complete() sends the right parameters to the API."""
        req = LLMRequest(
            system_prompt="You are helpful.",
            user_prompt="Summarize.",
            model="gpt-4o",
            temperature=0.5,
            max_tokens=2048,
        )
        mock_response = _mock_openai_response(content="Summary.", usage=_mock_openai_usage())
        with patch.object(backend._client.chat.completions, "create", return_value=mock_response) as mock_create:
            backend.complete(req)

        mock_create.assert_called_once_with(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Summarize."},
            ],
            temperature=0.5,
            max_tokens=2048,
        )

    def test_complete_json_mode(self, backend: OpenAIBackend) -> None:
        """complete() sends response_format when JSON mode is requested."""
        req = LLMRequest(
            system_prompt="Return JSON.",
            user_prompt="Classify these commits.",
            model="gpt-4o",
            response_format="json",
        )
        mock_response = _mock_openai_response(
            content=json.dumps({"themes": []}),
            usage=_mock_openai_usage(),
        )
        with patch.object(backend._client.chat.completions, "create", return_value=mock_response) as mock_create:
            backend.complete(req)

        mock_create.assert_called_once_with(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Return JSON."},
                {"role": "user", "content": "Classify these commits."},
            ],
            temperature=0.3,
            max_tokens=4096,
            response_format={"type": "json_object"},
        )


# ── Complete (error mapping) ──────────────────────────────────────────────────


class TestOpenAICompleteErrors:
    """OpenAIBackend.complete — error mapping."""

    def test_auth_error(self, backend: OpenAIBackend, minimal_request: LLMRequest) -> None:
        """AuthenticationError maps to LLMAuthenticationError."""
        from openai import AuthenticationError as OpenAIAuthError

        mock_response = MagicMock()
        mock_response.status_code = 401
        with patch.object(
            backend._client.chat.completions,
            "create",
            side_effect=OpenAIAuthError(
                "Incorrect API key provided",
                response=mock_response,
                body=None,
            ),
        ):
            with pytest.raises(LLMAuthenticationError, match="OpenAI authentication failed"):
                backend.complete(minimal_request)

    def test_rate_limit_error(self, backend: OpenAIBackend, minimal_request: LLMRequest) -> None:
        """RateLimitError maps to LLMRateLimitError."""
        from openai import RateLimitError as OpenAIRateLimitError

        mock_response = MagicMock()
        mock_response.status_code = 429
        with patch.object(
            backend._client.chat.completions,
            "create",
            side_effect=OpenAIRateLimitError(
                "Rate limit exceeded",
                response=mock_response,
                body=None,
            ),
        ):
            with pytest.raises(LLMRateLimitError, match="OpenAI rate limit exceeded"):
                backend.complete(minimal_request)

    def test_context_length_error(self, backend: OpenAIBackend, minimal_request: LLMRequest) -> None:
        """BadRequestError with context_length_exceeded maps to LLMContextLengthError."""
        from openai import BadRequestError as OpenAIBadRequestError

        with patch.object(
            backend._client.chat.completions,
            "create",
            side_effect=OpenAIBadRequestError(
                "This model's maximum context length is 128000 tokens",
                response=MagicMock(),
                body=None,
            ),
        ):
            with pytest.raises(LLMContextLengthError, match="OpenAI context length exceeded"):
                backend.complete(minimal_request)

    def test_context_length_error_alt_message(self, backend: OpenAIBackend, minimal_request: LLMRequest) -> None:
        """BadRequestError with 'maximum context' in message maps to LLMContextLengthError."""
        from openai import BadRequestError as OpenAIBadRequestError

        with patch.object(
            backend._client.chat.completions,
            "create",
            side_effect=OpenAIBadRequestError(
                "This request exceeds the maximum context length of 8192 tokens",
                response=MagicMock(),
                body=None,
            ),
        ):
            with pytest.raises(LLMContextLengthError, match="OpenAI context length exceeded"):
                backend.complete(minimal_request)

    def test_other_bad_request(self, backend: OpenAIBackend, minimal_request: LLMRequest) -> None:
        """Other BadRequestErrors (not context length) map to LLMError."""
        from openai import BadRequestError as OpenAIBadRequestError

        with patch.object(
            backend._client.chat.completions,
            "create",
            side_effect=OpenAIBadRequestError(
                "Unknown parameter: invalid_param",
                response=MagicMock(),
                body=None,
            ),
        ):
            with pytest.raises(LLMError, match="OpenAI bad request"):
                backend.complete(minimal_request)

    def test_api_status_error(self, backend: OpenAIBackend, minimal_request: LLMRequest) -> None:
        """APIStatusError (other HTTP errors) maps to LLMError."""
        from openai import APIStatusError as OpenAIStatusError

        with patch.object(
            backend._client.chat.completions,
            "create",
            side_effect=OpenAIStatusError(
                "Internal server error",
                response=MagicMock(),
                body=None,
            ),
        ):
            with pytest.raises(LLMError, match="OpenAI API error"):
                backend.complete(minimal_request)

    def test_api_error(self, backend: OpenAIBackend, minimal_request: LLMRequest) -> None:
        """APIError (network/connection) maps to LLMError."""
        from openai import APIError as OpenAIAPIError

        with patch.object(
            backend._client.chat.completions,
            "create",
            side_effect=OpenAIAPIError(
                "Connection timeout",
                request=MagicMock(),
                body=None,
            ),
        ):
            with pytest.raises(LLMError, match="OpenAI API request failed"):
                backend.complete(minimal_request)

    def test_unexpected_error(self, backend: OpenAIBackend, minimal_request: LLMRequest) -> None:
        """Unexpected exceptions map to LLMError."""
        with patch.object(
            backend._client.chat.completions,
            "create",
            side_effect=ValueError("Something broke internally"),
        ):
            with pytest.raises(LLMError, match="Unexpected OpenAI error"):
                backend.complete(minimal_request)


# ── Backend registry tests ────────────────────────────────────────────────────


class TestBackendRegistry:
    """Backend registration and factory."""

    def test_get_unknown_backend(self) -> None:
        """get_backend_class raises ValueError for unknown providers."""
        from git_story.backends import get_backend_class

        with pytest.raises(ValueError, match="Unknown provider"):
            get_backend_class("nonexistent-provider")

    def test_list_backends_includes_openai(self) -> None:
        """list_backends returns dict with openai."""
        backends = list_backends()
        assert "openai" in backends

    def test_get_backend_factory(self) -> None:
        """get_backend creates an OpenAI instance."""
        from git_story.backends import get_backend

        backend = get_backend("openai", api_key="sk-test")
        assert isinstance(backend, OpenAIBackend)
        assert backend.name == "openai"
