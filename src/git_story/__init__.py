"""git-story — Transform git history into PR-ready prose using LLMs."""

from git_story.git_reader import CommitInfo, CommitsResult, get_commits_with_diffs

__version__ = "0.1.0"
__all__ = [
    "__version__",
    "CommitInfo",
    "CommitsResult",
    "get_commits_with_diffs",
]
