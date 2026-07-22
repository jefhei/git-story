"""Tests for the LLM backend abstraction layer.

Verifies that:
- Data classes (LLMRequest, LLMResponse, TokenUsage) work correctly.
- LLMBackend ABC cannot be instantiated directly.
- Subclasses must implement all abstract methods.
- estimate_request_tokens helper works on concrete subclasses.
- Exception hierarchy is correct.
"""

from __future__ import annotations

from typing import Dict, Optional

import pytest

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


# ── Data class tests ──────────────────────────────────────────────────────────


class TestLLMRequest:
    """LLMRequest dataclass construction and defaults."""

    def test_minimal_construction(self) -> None:
        """A request can be built with only required fields."""
        req = LLMRequest(system_prompt="You are helpful.", user_prompt="Hello", model="gpt-4o")
        assert req.system_prompt == "You are helpful."
        assert req.user_prompt == "Hello"
        assert req.model == "gpt-4o"
        assert req.temperature == 0.3  # default
        assert req.max_tokens == 4096  # default
        assert req.response_format is None  # default

    def test_json_response_format(self) -> None:
        """JSON mode is selected via response_format='json'."""
        req = LLMRequest(
            system_prompt="You are helpful.",
            user_prompt="Hello",
            model="gpt-4o",
            response_format="json",
        )
        assert req.response_format == "json"

    def test_explicit_temperature(self) -> None:
        """Temperature can be overridden."""
        req = LLMRequest(
            system_prompt="You are helpful.",
            user_prompt="Hello",
            model="gpt-4o",
            temperature=0.8,
        )
        assert req.temperature == 0.8

    def test_frozen_immutable(self) -> None:
        """LLMRequest instances are frozen (immutable)."""
        req = LLMRequest(system_prompt="A", user_prompt="B", model="m")
        with pytest.raises(AttributeError):
            req.system_prompt = "changed"  # type: ignore[misc]


class TestLLMResponse:
    """LLMResponse dataclass construction."""

    def test_minimal_construction(self) -> None:
        """A response requires only content and model."""
        resp = LLMResponse(content="Hello back!", model="gpt-4o")
        assert resp.content == "Hello back!"
        assert resp.model == "gpt-4o"
        assert resp.usage is None

    def test_with_usage(self) -> None:
        """Usage statistics can be attached to a response."""
        usage = TokenUsage(prompt_tokens=50, completion_tokens=30, total_tokens=80)
        resp = LLMResponse(content="Hello back!", model="gpt-4o", usage=usage)
        assert resp.usage is not None
        assert resp.usage.prompt_tokens == 50
        assert resp.usage.total_tokens == 80

    def test_frozen_immutable(self) -> None:
        """LLMResponse instances are frozen."""
        resp = LLMResponse(content="Hi", model="m")
        with pytest.raises(AttributeError):
            resp.content = "changed"  # type: ignore[misc]


class TestTokenUsage:
    """TokenUsage dataclass and from_dict helper."""

    def test_default_all_zero(self) -> None:
        """Default TokenUsage has all fields at 0."""
        u = TokenUsage()
        assert u.prompt_tokens == 0
        assert u.completion_tokens == 0
        assert u.total_tokens == 0

    def test_from_dict_full(self) -> None:
        """from_dict builds a TokenUsage from a complete dict."""
        u = TokenUsage.from_dict({"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30})
        assert u.prompt_tokens == 10
        assert u.completion_tokens == 20
        assert u.total_tokens == 30

    def test_from_dict_missing_total(self) -> None:
        """from_dict computes total when it's missing from the dict."""
        u = TokenUsage.from_dict({"prompt_tokens": 15, "completion_tokens": 25})
        assert u.total_tokens == 40

    def test_from_dict_empty(self) -> None:
        """from_dict handles an empty dict gracefully."""
        u = TokenUsage.from_dict({})
        assert u.prompt_tokens == 0
        assert u.completion_tokens == 0
        assert u.total_tokens == 0

    def test_from_dict_none_values(self) -> None:
        """from_dict handles None values in dict gracefully."""
        u = TokenUsage.from_dict(
            {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}  # type: ignore[dict-item]
        )
        assert u.prompt_tokens == 0
        assert u.completion_tokens == 0
        assert u.total_tokens == 0


# ── Exception hierarchy tests ─────────────────────────────────────────────────


class TestLLMExceptions:
    """LLM exception hierarchy correctness."""

    def test_all_are_llm_error_subclasses(self) -> None:
        """All custom exceptions inherit from LLMError."""
        assert issubclass(LLMAuthenticationError, LLMError)
        assert issubclass(LLMRateLimitError, LLMError)
        assert issubclass(LLMContextLengthError, LLMError)

    def test_can_catch_generic_llm_error(self) -> None:
        """A generic except LLMError catches all backend exceptions."""
        for exc in [LLMAuthenticationError("bad key"), LLMRateLimitError("too fast"), LLMContextLengthError("too long")]:
            assert isinstance(exc, LLMError)

    def test_exception_message_preserved(self) -> None:
        """Exception messages are preserved."""
        msg = "Custom error message"
        assert str(LLMError(msg)) == msg
        assert str(LLMAuthenticationError(msg)) == msg


# ── Abstract backend tests ────────────────────────────────────────────────────


class TestLLMBackendABC:
    """LLMBackend abstract base class enforcement."""

    def test_cannot_instantiate_abc_directly(self) -> None:
        """LLMBackend cannot be instantiated directly."""
        with pytest.raises(TypeError, match="abstract"):
            LLMBackend()  # type: ignore[abstract]

    def test_must_implement_abstract_methods(self) -> None:
        """Subclass missing abstract methods cannot be instantiated."""
        # Missing complete()
        class IncompleteBackend(LLMBackend):
            @property
            def name(self) -> str:
                return "incomplete"

            def count_tokens(self, text: str) -> int:
                return len(text) // 4

        with pytest.raises(TypeError, match="abstract"):
            IncompleteBackend()  # type: ignore[abstract]


# ── Concrete subclass fixture ─────────────────────────────────────────────────


class FakeBackend(LLMBackend):
    """A minimal concrete backend for testing interface behaviour."""

    def __init__(self, name: str = "fake") -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def complete(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(
            content=f"Response to: {request.user_prompt[:40]}",
            model=request.model,
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )

    def count_tokens(self, text: str) -> int:
        # Crude: 1 token per word
        return len(text.split())


class TestConcreteBackend:
    """A concrete LLMBackend subclass works correctly."""

    @pytest.fixture
    def backend(self) -> FakeBackend:
        return FakeBackend()

    def test_name_property(self, backend: FakeBackend) -> None:
        """The name property returns the configured name."""
        assert backend.name == "fake"

    def test_custom_name(self) -> None:
        """A custom name can be set at construction."""
        b = FakeBackend(name="custom-backend")
        assert b.name == "custom-backend"

    def test_complete_returns_response(self, backend: FakeBackend) -> None:
        """complete() returns an LLMResponse."""
        req = LLMRequest(system_prompt="Be concise.", user_prompt="Summarize this diff", model="gpt-4o")
        resp = backend.complete(req)
        assert isinstance(resp, LLMResponse)
        assert "Summarize this diff" in resp.content

    def test_complete_returns_usage(self, backend: FakeBackend) -> None:
        """complete() returns usage statistics."""
        req = LLMRequest(system_prompt="", user_prompt="Hi", model="m")
        resp = backend.complete(req)
        assert resp.usage is not None
        assert resp.usage.prompt_tokens == 10

    def test_count_tokens_returns_int(self, backend: FakeBackend) -> None:
        """count_tokens returns an integer estimate."""
        assert backend.count_tokens("hello world") == 2
        assert backend.count_tokens("") == 0

    def test_repr(self, backend: FakeBackend) -> None:
        """__repr__ includes the class name and backend name."""
        rep = repr(backend)
        assert "FakeBackend" in rep
        assert "fake" in rep


class TestEstimateRequestTokens:
    """estimate_request_tokens helper on concrete backends."""

    @pytest.fixture
    def backend(self) -> FakeBackend:
        return FakeBackend()

    def test_estimate_returns_positive_integer(self, backend: FakeBackend) -> None:
        """estimate_request_tokens returns a positive integer."""
        total = backend.estimate_request_tokens(
            system_prompt="You are an expert summarizer.",
            user_prompt="Here is a git diff with changes to several files.",
            max_tokens=1024,
        )
        assert isinstance(total, int)
        assert total > 0

    def test_estimate_includes_overhead(self, backend: FakeBackend) -> None:
        """estimate includes overhead in the total."""
        # Small prompts + max_tokens
        total = backend.estimate_request_tokens(
            system_prompt="Hi",
            user_prompt="Bye",
            max_tokens=256,
        )
        # system tokens (2) + user tokens (1) + max_tokens (256) + overhead (4) + response_overhead (200)
        assert total >= 2 + 1 + 256

    def test_estimate_scales_with_prompt_size(self, backend: FakeBackend) -> None:
        """Larger prompts produce larger estimates."""
        small = backend.estimate_request_tokens(
            system_prompt="Short",
            user_prompt="Small",
            max_tokens=512,
        )
        large = backend.estimate_request_tokens(
            system_prompt="A long system prompt with many words for testing purposes",
            user_prompt="A very long user prompt that contains significantly more tokens and should result in a larger estimate overall.",
            max_tokens=512,
        )
        assert large > small
