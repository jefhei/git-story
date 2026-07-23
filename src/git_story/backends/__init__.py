"""Backend adapters for LLM providers.

Each submodule in this package implements :class:`git_story.llm.LLMBackend`
for a specific provider (OpenAI, Anthropic, Ollama).

The :func:`get_backend` factory function resolves a provider name
to the corresponding backend class.
"""

from __future__ import annotations

from typing import Dict, Optional, Type

from git_story.llm import LLMBackend

_backend_registry: Dict[str, Type[LLMBackend]] = {}


def register_backend(name: str, cls: Type[LLMBackend]) -> None:
    """Register a backend class under a provider name.

    Args:
        name: Provider identifier (e.g. ``"openai"``, ``"anthropic"``).
        cls: The backend class (must be a concrete subclass of :class:`LLMBackend`).
    """
    _backend_registry[name] = cls


def get_backend_class(name: str) -> Type[LLMBackend]:
    """Return the backend class registered for *name*.

    Args:
        name: Provider identifier.

    Returns:
        The corresponding :class:`LLMBackend` subclass.

    Raises:
        ValueError: If no backend is registered for *name*.
    """
    cls = _backend_registry.get(name)
    if cls is None:
        available = ", ".join(sorted(_backend_registry))
        raise ValueError(
            f"Unknown provider: '{name}'. "
            f"Available providers: {available}"
        )
    return cls


def list_backends() -> Dict[str, Type[LLMBackend]]:
    """Return a copy of the registered backend map."""
    return dict(_backend_registry)


# ── Lazy imports ─────────────────────────────────────────────────────────────

def get_backend(provider: str, **kwargs) -> LLMBackend:
    """Instantiate a backend for the given provider.

    This is the preferred way to create a backend — it lazily imports
    the concrete class and handles construction.

    Args:
        provider: Provider identifier (e.g. ``"openai"``).
        **kwargs: Keyword arguments passed to the backend constructor
            (typically ``api_key=...``).

    Returns:
        An instantiated :class:`LLMBackend`.

    Raises:
        ValueError: If *provider* is not a registered backend.
    """
    cls = get_backend_class(provider)
    return cls(**kwargs)
