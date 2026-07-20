"""git-story — Transform git history into PR-ready prose using LLMs."""

from git_story.config import Config, resolve_config, check_auth
from git_story.git_reader import CommitInfo, CommitsResult, get_commits_with_diffs

__version__ = "0.1.0"
__all__ = [
    "__version__",
    "Config",
    "CommitInfo",
    "CommitsResult",
    "check_auth",
    "get_commits_with_diffs",
    "resolve_config",
]
