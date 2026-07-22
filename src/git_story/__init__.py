"""git-story — Transform git history into PR-ready prose using LLMs."""

from git_story.config import Config, resolve_config, check_auth
from git_story.git_reader import (
    CommitInfo,
    CommitsResult,
    get_commit_info,
    get_commits_with_diffs,
    _EMPTY_RANGE_WARNING,
)
from git_story.llm import (
    LLMBackend,
    LLMError,
    LLMAuthenticationError,
    LLMRateLimitError,
    LLMContextLengthError,
    LLMRequest,
    LLMResponse,
    TokenUsage,
)

__version__ = "0.1.0"
__all__ = [
    "__version__",
    "Config",
    "CommitInfo",
    "CommitsResult",
    "check_auth",
    "get_commit_info",
    "get_commits_with_diffs",
    "resolve_config",
    "_EMPTY_RANGE_WARNING",
    "LLMBackend",
    "LLMError",
    "LLMAuthenticationError",
    "LLMRateLimitError",
    "LLMContextLengthError",
    "LLMRequest",
    "LLMResponse",
    "TokenUsage",
]
