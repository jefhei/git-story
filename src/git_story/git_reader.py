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
    commits, and any non-fatal errors encountered."""

    range: str
    commits: List[CommitInfo]
    errors: List[str]

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
        ValueError: If the revision or range cannot be resolved.
    """
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

    try:
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

    # Diff against parent, or --root for the root commit (empty tree).
    try:
        if commit.parents:
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
    shas = parse_range(revision_range)
    commits: List[CommitInfo] = []
    for sha in shas:
        try:
            commits.append(get_commit_info(sha))
        except ValueError as exc:
            errors.append(str(exc))
    return CommitsResult(range=revision_range, commits=commits, errors=errors)
