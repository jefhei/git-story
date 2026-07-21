"""Git commit range parsing and diff extraction via GitPython.

Core module for reading git commits and diffs from a local repository.
Supports revision range syntax (e.g., ``main..feature``) and single refs.
"""

from __future__ import annotations

import dataclasses
import datetime
import logging
from typing import List

import git

logger = logging.getLogger(__name__)


# ── CommitInfo --------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class CommitInfo:
    """Structured information about a single commit."""

    sha: str
    author_name: str
    author_email: str
    authored_at: datetime.datetime
    message: str
    message_summary: str
    diff: str
    parents: List[str]
    is_merge: bool = False


@dataclasses.dataclass(frozen=True)
class CommitsResult:
    """Result of parsing a revision range — the range string, list of
    commits, and any non-fatal errors encountered.

    Attributes:
        range: The original revision range string.
        commits: Ordered list of commits (newest first).
        errors: Fatal or per-commit errors encountered.
        warnings: Non-fatal warnings (e.g. empty range, fixup commits).
    """

    range: str
    commits: List[CommitInfo]
    errors: List[str]
    warnings: List[str] = dataclasses.field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.commits)


def open_repo(path: str = ".") -> git.Repo:
    """Open a git repository at *path* (walking up parent directories).

    Args:
        path: Directory to start searching from.

    Returns:
        An opened ``git.Repo`` instance.

    Raises:
        ValueError: If *path* is not inside a git repository.
    """
    try:
        return git.Repo(path, search_parent_directories=True)
    except git.InvalidGitRepositoryError as exc:
        raise ValueError(f"Not a git repository: {path}") from exc


_EMPTY_RANGE_WARNING = "Empty revision range — no commits found for '{}'. Both ends point to the same commit."


def _validate_range_parts(rev1: str, rev2: str, revision_range: str) -> None:
    """Validate that both sides of a ``..`` range are non-empty and resolvable.

    Raises:
        ValueError: If either side is empty or has an obviously invalid format.
    """
    if not rev1.strip():
        raise ValueError(
            f"Invalid revision range: '{revision_range}' — "
            f"left side of '..' is empty. Use a ref like 'main..{rev2 or 'HEAD'}'."
        )
    if not rev2.strip():
        raise ValueError(
            f"Invalid revision range: '{revision_range}' — "
            f"right side of '..' is empty. Use a ref like '{rev1}..HEAD'."
        )


def parse_range(revision_range: str) -> List[str]:
    """Parse a revision range string into chronologically-ordered commit SHAs.

    Supports:
    - Single ref: ``HEAD``, ``main``, ``v1.0``
    - Range: ``main..feature``, ``HEAD~5..HEAD``

    Args:
        revision_range: A git revision range expression.

    Returns:
        List of commit hex SHAs in reverse chronological order (newest first).

    Raises:
        ValueError: If the revision range string is empty, malformed, or
            cannot be resolved.
    """
    if not revision_range or not revision_range.strip():
        raise ValueError("Revision range is empty. Provide a ref like 'HEAD' or 'main..feature'.")

    repo = open_repo()

    # --- Single revision ---------------------------------------------------
    if ".." not in revision_range:
        try:
            commit = repo.commit(revision_range)
            return [commit.hexsha]
        except (git.BadName, git.GitCommandError, ValueError) as exc:
            raise ValueError(f"Invalid revision: {revision_range}") from exc

    # --- Range <rev1>..<rev2> ----------------------------------------------
    parts = revision_range.split("..", 1)
    rev1, rev2 = parts[0], parts[1]

    _validate_range_parts(rev1, rev2, revision_range)

    try:
        # Validate both refs exist before iterating
        repo.commit(rev1)
        repo.commit(rev2)
        commits = list(repo.iter_commits(f"{rev1}..{rev2}"))
    except (git.BadName, git.GitCommandError) as exc:
        raise ValueError(f"Invalid revision range: {revision_range}") from exc

    return [c.hexsha for c in commits]


def get_commit_info(sha_or_ref: str) -> CommitInfo:
    """Build a ``CommitInfo`` for a single commit referenced by SHA or ref name.

    Includes the full diff against the parent (or against the empty tree for
    root commits).

    Args:
        sha_or_ref: Commit SHA, branch name, tag, or any ref-like expression.

    Returns:
        A ``CommitInfo`` dataclass instance.

    Raises:
        ValueError: If *sha_or_ref* cannot be resolved.
    """
    repo = open_repo()
    try:
        commit = repo.commit(sha_or_ref)
    except (git.BadName, git.GitCommandError, ValueError) as exc:
        raise ValueError(f"Invalid commit: {sha_or_ref}") from exc

    # Diff strategy: use combined diff for merge commits, otherwise
    # diff against first parent (or empty tree for root commit).
    try:
        if len(commit.parents) > 1:
            # Merge commit: show combined diff using --cc
            # This shows the changes that differ from ALL parents,
            # giving a cleaner view of what the merge actually introduced.
            diff = repo.git.diff_tree("--cc", "-p", commit.hexsha)
        elif commit.parents:
            diff = repo.git.diff(commit.parents[0].hexsha, commit.hexsha)
        else:
            diff = repo.git.diff("--root", commit.hexsha)
    except git.GitCommandError:
        diff = ""

    return CommitInfo(
        sha=commit.hexsha,
        author_name=commit.author.name,
        author_email=commit.author.email,
        authored_at=commit.authored_datetime,
        message=commit.message.strip(),
        message_summary=commit.summary or commit.message.split("\n")[0].strip(),
        diff=diff,
        parents=[p.hexsha for p in commit.parents],
        is_merge=len(commit.parents) > 1,
    )


def get_commits_with_diffs(revision_range: str) -> CommitsResult:
    """Return structured commit data (metadata + diff) for a revision range.

    Results are in reverse-chronological order (newest commit first), matching
    ``git log`` default behaviour.

    Args:
        revision_range: A git revision range expression.

    Returns:
        A ``CommitsResult`` with the resolved commits, the original range
        string, and any non-fatal errors.

    Raises:
        ValueError: If the revision range itself is invalid.
    """
    errors: List[str] = []
    warnings: List[str] = []
    shas = parse_range(revision_range)

    if not shas:
        warnings.append(_EMPTY_RANGE_WARNING.format(revision_range))
        return CommitsResult(range=revision_range, commits=[], errors=[], warnings=warnings)

    commits: List[CommitInfo] = []
    for sha in shas:
        try:
            commits.append(get_commit_info(sha))
        except ValueError as exc:
            errors.append(str(exc))
    return CommitsResult(range=revision_range, commits=commits, errors=errors, warnings=warnings)
