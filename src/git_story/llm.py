"""LLM backend abstraction layer — pluggable interface for AI providers.

Defines the abstract base class :class:`LLMBackend` and supporting data
classes (:class:`LLMRequest`, :class:`LLMResponse`, :class:`TokenUsage`)
that all concrete backends (OpenAI, Anthropic, Ollama) must implement.

The interface is deliberately small and synchronous for MVP simplicity.
Streaming and async variants can be added later as optional extensions.
"""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from typing import Dict, List, Optional


# ── Data classes ──────────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class LLMRequest:
    """A completion request to an LLM backend.

    Attributes:
        system_prompt: System-level instruction defining the assistant's role.
        user_prompt: The primary input containing the data to process.
        model: Model identifier (e.g. ``"gpt-4o"``, ``"claude-sonnet-4-20250514"``).
        temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative).
        max_tokens: Maximum tokens in the response.
        response_format: Requested output format (``"json"`` for JSON mode,
            or ``None`` for plain text).
    """

    system_prompt: str
    user_prompt: str
    model: str
    temperature: float = 0.3
    max_tokens: int = 4096
    response_format: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class TokenUsage:
    """Token usage statistics returned by an LLM backend.

    Attributes:
        prompt_tokens: Number of tokens in the prompt.
        completion_tokens: Number of tokens in the generated response.
        total_tokens: Sum of prompt and completion tokens.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @classmethod
    def from_dict(cls, data: Dict[str, int]) -> TokenUsage:
        """Build a ``TokenUsage`` from a raw dict (e.g. from an API response).

        Falls back to 0 for any missing keys so that partial provider
        responses don't crash the tool.
        """
        return cls(
            prompt_tokens=data.get("prompt_tokens", 0) or 0,
            completion_tokens=data.get("completion_tokens", 0) or 0,
            total_tokens=data.get("total_tokens", 0)
            or (data.get("prompt_tokens", 0) or 0) + (data.get("completion_tokens", 0) or 0),
        )


@dataclasses.dataclass(frozen=True)
class LLMResponse:
    """A structured response from an LLM backend.

    Attributes:
        content: The generated text content.
        model: The model that produced the response.
        usage: Token usage statistics, if reported by the provider.
    """

    content: str
    model: str
    usage: Optional[TokenUsage] = None


# ── Exceptions ────────────────────────────────────────────────────────────────


class LLMError(Exception):
    """Base exception for all LLM backend errors.

    Subclasses may define more specific error types for rate limits,
    authentication failures, context-length exceeded, etc.
    """


class LLMAuthenticationError(LLMError):
    """Raised when the API key is invalid, missing, or expired."""


class LLMRateLimitError(LLMError):
    """Raised when the API rate limit has been exceeded."""


class LLMContextLengthError(LLMError):
    """Raised when the total token count exceeds the model's context window."""


# ── Abstract backend ──────────────────────────────────────────────────────────


class LLMBackend(ABC):
    """Abstract interface for LLM backends.

    Every concrete backend (OpenAI, Anthropic, Ollama) must implement:

    * :meth:`complete` — send a prompt and get a response back.
    * :meth:`count_tokens` — estimate how many tokens a text string uses.
    * :attr:`name` — a human-readable provider name.

    Example usage::

        class MyBackend(LLMBackend):
            @property
            def name(self) -> str:
                return "my-backend"

            def complete(self, request: LLMRequest) -> LLMResponse:
                # ... implement API call ...
                ...

            def count_tokens(self, text: str) -> int:
                # ... estimate tokens ...
                return len(text) // 4  # crude heuristic
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable backend identifier (e.g. ``"openai"``, ``"anthropic"``)."""

    @abstractmethod
    def complete(self, request: LLMRequest) -> LLMResponse:
        """Send a completion request to the LLM backend.

        This is a synchronous, blocking call. Subclasses should implement
        appropriate retry logic and error mapping to the exception hierarchy
        defined in this module.

        Args:
            request: The populated request parameters.

        Returns:
            The backend's response wrapped in an :class:`LLMResponse`.

        Raises:
            LLMAuthenticationError: If the API key is invalid.
            LLMRateLimitError: If the rate limit is exceeded.
            LLMContextLengthError: If the prompt exceeds the context window.
            LLMError: For any other API or network failure.
        """

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Estimate the number of tokens in *text*.

        Concrete backends should use the most accurate method available
        (e.g., ``tiktoken`` for OpenAI, or the Anthropic tokeniser).
        If no library is available, a heuristic of ~4 characters per token
        is an acceptable fallback.

        Args:
            text: The text to estimate.

        Returns:
            An integer token count (may be approximate).
        """

    def estimate_request_tokens(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        estimated_response_tokens_per_message: int = 200,
    ) -> int:
        """Estimate the total tokens a request will consume.

        Useful for pre-filtering requests that would exceed a model's
        context window before sending them to the API.

        The estimate is::

            system_tokens + user_tokens + max_tokens + overhead

        where *overhead* accounts for chat template markup (~4 tokens
        for message framing).

        Args:
            system_prompt: The system prompt text.
            user_prompt: The user prompt text.
            max_tokens: The ``max_tokens`` parameter of the request.
            estimated_response_tokens_per_message: Token overhead for
                the response message formatting (default 200 is a safe
                upper bound for most chat APIs).

        Returns:
            Estimated total token consumption.
        """
        system_tokens = self.count_tokens(system_prompt)
        user_tokens = self.count_tokens(user_prompt)
        overhead = 4  # role markers and formatting
        return system_tokens + user_tokens + max_tokens + overhead + estimated_response_tokens_per_message

    def __repr__(self) -> str:
        return f"<{type(self).__name__}(name='{self.name}')>"
