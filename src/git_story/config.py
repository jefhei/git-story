"""Configuration management with layered resolution.

Resolution order (highest to lowest priority):

1. **CLI flags** — passed in from the Click command (highest priority)
2. **Environment variables** — ``GIT_STORY_*`` namespace
3. **Config file** — ``.git-story.toml``, ``~/.config/git-story/config.toml``,
   or ``~/.git-story.toml`` (first found wins)
4. **Defaults** — hard-coded defaults (lowest priority)

Auth validation is provided via :func:`check_auth`, which verifies that
the selected provider's API key is available before an LLM call is made.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

try:
    import tomllib
except ImportError:  # Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]


# ── Config dataclass ────────────────────────────────────────────────────────


@dataclass
class Config:
    """Resolved configuration for git-story.

    All fields have sensible defaults so the tool works out of the box
    for basic usage.  API keys are never set via CLI flags — they are
    read from environment variables or the config file.
    """

    # CLI-controllable settings
    provider: str = "openai"
    model: str = ""
    style: str = "plain-markdown"
    output: Optional[str] = None
    verbose: bool = False

    # API keys (loaded from env vars or config file, never from CLI flags)
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None


DEFAULT_CONFIG = Config()

# Config file search paths (in order of precedence — first found wins).
# Local config (.git-story.toml in CWD) takes priority over user-global
# config.
CONFIG_FILE_CANDIDATES: List[Path] = [
    Path(".git-story.toml"),
    Path("~/.config/git-story/config.toml").expanduser(),
    Path("~/.git-story.toml").expanduser(),
]

# Environment variable prefix for git-story-specific vars.
ENV_PREFIX = "GIT_STORY_"

# API key env vars per provider (preferred var listed first).
API_KEY_ENV_VARS: Dict[str, List[str]] = {
    "openai": [f"{ENV_PREFIX}OPENAI_API_KEY", "OPENAI_API_KEY"],
    "anthropic": [f"{ENV_PREFIX}ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY"],
}

# Default model per provider when none is explicitly set.
PROVIDER_DEFAULT_MODELS: Dict[str, str] = {
    "openai": "gpt-4o",
    "anthropic": "claude-sonnet-4-20250514",
    "ollama": "",
}

# Mapping of TOML config-file keys to Config attribute names with optional
# type coercion.
_TOML_COERCIONS: Dict[str, Callable[[object], object]] = {
    "provider": str,
    "model": str,
    "style": str,
    "output": lambda v: str(v) if v else None,
    "verbose": bool,
}


# ── Helpers ─────────────────────────────────────────────────────────────────


def _find_config_file() -> Optional[Path]:
    """Return the first existing config file from the search paths."""
    for path in CONFIG_FILE_CANDIDATES:
        if path.exists():
            return path
    return None


def _load_toml(path: Path) -> dict:
    """Parse a TOML file and return the dict contents.

    Returns an empty dict on any parse error or permission issue so that
    a broken config file never crashes the tool.
    """
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except (FileNotFoundError, PermissionError, tomllib.TOMLDecodeError):
        return {}


def _get_env(key: str) -> Optional[str]:
    """Read an environment variable, returning ``None`` if unset or empty."""
    val = os.environ.get(key)
    return val if val else None


def _resolve_api_key(provider: str) -> Optional[str]:
    """Check known env var names for a provider's API key.

    Returns the first non-empty value found, or ``None``.
    """
    for var in API_KEY_ENV_VARS.get(provider, []):
        key = _get_env(var)
        if key:
            return key
    return None


# ── Main resolution ─────────────────────────────────────────────────────────


def resolve_config(
    *,
    cli_provider: Optional[str] = None,
    cli_model: Optional[str] = None,
    cli_style: Optional[str] = None,
    cli_output: Optional[str] = None,
    cli_verbose: bool = False,
    cli_config: Optional[str] = None,
) -> Config:
    """Build a resolved :class:`Config` by layering sources.

    Parameters are keyword-only to prevent accidental ordering mistakes.
    All ``cli_*`` parameters correspond to CLI flags (``None`` means the
    flag was not provided, so the next-priority source is used).

    Resolution order: CLI flags > env vars > config file > defaults.
    """
    cfg = Config()

    # 1. Defaults are already baked into the Config() constructor.

    # 2. Config file (TOML)
    config_path: Optional[Path] = None
    if cli_config is not None:
        config_path = Path(cli_config)
    else:
        config_path = _find_config_file()

    if config_path is not None and config_path.exists():
        toml_data = _load_toml(config_path)
        # Support both top-level keys and a [git-story] section
        if "git-story" in toml_data:
            toml_data = toml_data["git-story"]
        if isinstance(toml_data, dict):
            for key, coercer in _TOML_COERCIONS.items():
                if key in toml_data:
                    setattr(cfg, key, coercer(toml_data[key]))
            # API keys from config file (non-standard keys)
            if "openai_api_key" in toml_data:
                cfg.openai_api_key = str(toml_data["openai_api_key"])
            if "anthropic_api_key" in toml_data:
                cfg.anthropic_api_key = str(toml_data["anthropic_api_key"])

    # 3. Environment variables (GIT_STORY_* namespace)
    _apply_env_overrides(cfg)

    # 4. CLI flags (highest priority)
    if cli_provider is not None:
        cfg.provider = cli_provider
    if cli_model is not None:
        cfg.model = cli_model
    if cli_style is not None:
        cfg.style = cli_style
    if cli_output is not None:
        cfg.output = cli_output
    cfg.verbose = cli_verbose  # flags always win

    # --- Post-resolution ---

    # Fill in a default model for the selected provider if none was set.
    if not cfg.model:
        cfg.model = PROVIDER_DEFAULT_MODELS.get(cfg.provider, "")

    # Resolve API keys from environment (highest priority for secrets).
    resolved_key = _resolve_api_key(cfg.provider)
    if cfg.provider == "openai":
        cfg.openai_api_key = resolved_key or cfg.openai_api_key
    elif cfg.provider == "anthropic":
        cfg.anthropic_api_key = resolved_key or cfg.anthropic_api_key

    return cfg


def _apply_env_overrides(cfg: Config) -> None:
    """Override config fields from ``GIT_STORY_*`` environment variables.

    Mutates *cfg* in place.
    """
    env_provider = _get_env(f"{ENV_PREFIX}PROVIDER")
    if env_provider:
        cfg.provider = env_provider
    env_model = _get_env(f"{ENV_PREFIX}MODEL")
    if env_model:
        cfg.model = env_model
    env_style = _get_env(f"{ENV_PREFIX}STYLE")
    if env_style:
        cfg.style = env_style


# ── Auth validation ─────────────────────────────────────────────────────────


def check_auth(cfg: Config) -> Optional[str]:
    """Validate that the selected provider's API key is available.

    Args:
        cfg: A resolved :class:`Config` instance.

    Returns:
        An error message string if the key is missing, or ``None`` if
        everything looks fine.
    """
    if cfg.provider == "openai" and not cfg.openai_api_key:
        return (
            "OpenAI API key not found.\n\n"
            "Set the OPENAI_API_KEY or GIT_STORY_OPENAI_API_KEY "
            "environment variable, or add openai_api_key to your\n"
            "config file (~/.config/git-story/config.toml)."
        )
    if cfg.provider == "anthropic" and not cfg.anthropic_api_key:
        return (
            "Anthropic API key not found.\n\n"
            "Set the ANTHROPIC_API_KEY or GIT_STORY_ANTHROPIC_API_KEY "
            "environment variable, or add anthropic_api_key to your\n"
            "config file (~/.config/git-story/config.toml)."
        )
    if cfg.provider == "ollama":
        # Ollama runs locally — no API key required.
        return None
    return None
