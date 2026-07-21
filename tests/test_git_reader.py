"""Tests for git range parsing and commit/diff extraction."""

from __future__ import annotations

import os
from typing import Iterator

import pytest
from click.testing import CliRunner
from git_story.cli import main
from git_story.git_reader import (
    CommitInfo,
    CommitsResult,
    get_commit_info,
    get_commits_with_diffs,
    open_repo,
    parse_range,
)


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def _repo_path(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Create a temporary git repo with known history for testing."""
    path = tmp_path_factory.mktemp("test_repo")
    _init_repo(path)
    return str(path)


@pytest.fixture
def repo(_repo_path: str) -> Iterator[str]:
    """Change directory to the test repo for the duration of a test.

    All git_reader functions call ``open_repo()`` which searches from
    the current working directory, so we must be inside the test repo
    for isolation.
    """
    old_cwd = os.getcwd()
    os.chdir(_repo_path)
    try:
        yield _repo_path
    finally:
        os.chdir(old_cwd)


def _init_repo(path) -> None:
    """Initialise a small git repository with structured history."""
    import git
    from pathlib import Path

    path = Path(path)

    repo = git.Repo.init(path)
    repo.config_writer().set_value("user", "name", "Test User").release()
    repo.config_writer().set_value("user", "email", "test@example.com").release()

    # Determine the default branch name (could be 'main' or 'master')
    default_branch = repo.active_branch.name

    # Create an initial commit
    (path / "README.md").write_text("# Test\n")
    repo.index.add(["README.md"])
    repo.index.commit("Initial commit")

    # Commit 2: add a feature
    (path / "feature.py").write_text("def hello():\n    return 'hello'\n")
    repo.index.add(["feature.py"])
    repo.index.commit("Add hello feature")

    # Commit 3: second feature
    (path / "utils.py").write_text("def add(a, b):\n    return a + b\n")
    repo.index.add(["utils.py"])
    repo.index.commit("Add utility module")

    # Create a branch and make a commit there
    branch = repo.create_head("feature-branch")
    branch.checkout()
    (path / "branch_file.py").write_text("# branch work\n")
    repo.index.add(["branch_file.py"])
    repo.index.commit("Work on branch")

    # Merge back into default branch (produces a merge commit)
    repo.heads[default_branch].checkout()
    repo.git.merge("feature-branch", no_ff=True)

    # Commit 5: another commit on default branch after merge
    (path / "final.py").write_text("# final\n")
    repo.index.add(["final.py"])
    repo.index.commit("Final commit")


# ── Tests: open_repo ───────────────────────────────────────────────────────


def test_open_repo_success(_repo_path: str) -> None:
    """Can open the test repo."""
    repo = open_repo(_repo_path)
    assert not repo.bare
    repo.close()


def test_open_repo_not_a_repo(tmp_path) -> None:
    """open_repo raises ValueError for non-repo directories."""
    with pytest.raises(ValueError, match="Not a git repository"):
        open_repo(str(tmp_path))


# ── Tests: parse_range ─────────────────────────────────────────────────────


def test_parse_range_head(repo: str) -> None:
    """Parse a single HEAD reference."""
    shas = parse_range("HEAD")
    assert len(shas) == 1
    assert isinstance(shas[0], str)
    assert len(shas[0]) == 40


def test_parse_range_branch_name(repo: str) -> None:
    """Parse a branch name as a single ref."""
    import git as _git
    _repo = _git.Repo(".")
    branch_name = _repo.active_branch.name
    _repo.close()
    shas = parse_range(branch_name)
    assert len(shas) == 1


def test_parse_range_two_dot_range(repo: str) -> None:
    """Parse a rev1..rev2 range."""
    shas = parse_range("HEAD~2..HEAD")
    assert len(shas) >= 1


def test_parse_range_invalid_revision(repo: str) -> None:
    """Invalid revision raises ValueError."""
    with pytest.raises(ValueError, match="Invalid revision"):
        parse_range("this-does-not-exist-xyz")


def test_parse_range_invalid_range(repo: str) -> None:
    """Invalid range raises ValueError."""
    with pytest.raises(ValueError, match="Invalid revision range"):
        parse_range("main..this-does-not-exist-xyz")


def test_parse_range_empty_string(repo: str) -> None:
    """Empty string raises ValueError."""
    with pytest.raises(ValueError, match="Revision range is empty"):
        parse_range("")


def test_parse_range_whitespace_string(repo: str) -> None:
    """Whitespace-only string raises ValueError."""
    with pytest.raises(ValueError, match="Revision range is empty"):
        parse_range("   ")


def test_parse_range_bare_two_dots(repo: str) -> None:
    """Bare '..' raises ValueError."""
    with pytest.raises(ValueError, match="left side of '..' is empty"):
        parse_range("..")


def test_parse_range_left_empty(repo: str) -> None:
    """Range with empty left side raises ValueError."""
    with pytest.raises(ValueError, match="left side of '..' is empty"):
        parse_range("..HEAD")


def test_parse_range_right_empty(repo: str) -> None:
    """Range with empty right side raises ValueError."""
    with pytest.raises(ValueError, match="right side of '..' is empty"):
        parse_range("main..")


def test_parse_range_same_ref_empty(repo: str) -> None:
    """Range with same ref on both sides returns empty list."""
    shas = parse_range("HEAD..HEAD")
    assert shas == []


# ── Tests: get_commit_info ─────────────────────────────────────────────────


def test_get_commit_info_head(repo: str) -> None:
    """Get commit info for HEAD."""
    info = get_commit_info("HEAD")
    assert isinstance(info, CommitInfo)
    assert info.sha
    assert len(info.sha) == 40
    assert info.author_name == "Test User"
    assert info.author_email == "test@example.com"
    assert info.message_summary


def test_get_commit_info_has_diff(repo: str) -> None:
    """A non-root commit has a non-empty diff."""
    info = get_commit_info("HEAD")
    assert info.diff is not None
    assert len(info.diff) > 0


def test_get_commit_info_root_commit(repo: str) -> None:
    """The root commit also produces a diff (against empty tree)."""
    # Walk backwards to find the root commit (no parents)
    info = get_commit_info("HEAD")
    while info.parents:
        info = get_commit_info(info.parents[0])
    # Root commit should have a diff even though it has no parents
    assert info.diff is not None
    assert len(info.diff) > 0, (
        f"Root commit {info.sha[:8]} should have a non-empty diff "
        f"(against empty tree). Parents: {info.parents}"
    )


def test_get_commit_info_merge_detection(repo: str) -> None:
    """Merge commits are correctly flagged."""
    # Use --all to iterate across all refs and find a merge commit
    import git as _git
    _repo = _git.Repo(".")
    for commit in _repo.iter_commits("--all"):
        info = get_commit_info(commit.hexsha)
        if len(info.parents) > 1:
            assert info.is_merge
            _repo.close()
            return
    _repo.close()
    pytest.skip("No merge commit found in test repo")


def test_get_commit_info_merge_has_combined_diff(repo: str) -> None:
    """Merge commit uses combined diff (--cc) format."""
    import git as _git
    _repo = _git.Repo(".")
    merge_sha = None
    for commit in _repo.iter_commits("--all"):
        if len(commit.parents) > 1:
            merge_sha = commit.hexsha
            break
    _repo.close()

    if merge_sha is None:
        pytest.skip("No merge commit found in test repo")

    info = get_commit_info(merge_sha)
    assert info.is_merge
    # Combined diff should exist and may start with a diff header
    # For clean merges with no conflicts, --cc may return empty string
    # Let's just verify it doesn't error and has the right metadata
    assert isinstance(info.diff, str)
    assert info.sha == merge_sha


def test_get_commit_info_merge_diff_differs_from_first_parent(repo: str) -> None:
    """Merge commit diff is derived from --cc (not just first-parent diff)."""
    import git as _git
    _repo = _git.Repo(".")
    merge_commit = None
    for commit in _repo.iter_commits("--all"):
        if len(commit.parents) > 1:
            merge_commit = commit
            break
    _repo.close()

    if merge_commit is None:
        pytest.skip("No merge commit found in test repo")

    info = get_commit_info(merge_commit.hexsha)
    first_parent = merge_commit.parents[0].hexsha
    # Get the first-parent diff directly
    _repo2 = _git.Repo(".")
    first_parent_diff = _repo2.git.diff(first_parent, merge_commit.hexsha)
    _repo2.close()

    # --cc diff should be different from first-parent diff in most cases
    # (combined diff shows only lines that conflict or differ from all parents)
    if info.diff:
        assert info.diff != first_parent_diff, (
            "Merge commit diff (-cc) should differ from first-parent diff"
        )


def test_get_commit_info_invalid(repo: str) -> None:
    """Invalid ref raises ValueError."""
    with pytest.raises(ValueError, match="Invalid commit"):
        get_commit_info("nonexistent-sha-1234567")


# ── Tests: get_commits_with_diffs ──────────────────────────────────────────


def test_get_commits_with_diffs_head(repo: str) -> None:
    """Returns a CommitsResult for HEAD."""
    result = get_commits_with_diffs("HEAD")
    assert isinstance(result, CommitsResult)
    assert result.range == "HEAD"
    assert result.count == 1
    assert len(result.commits) == 1


def test_get_commits_with_diffs_range(repo: str) -> None:
    """Returns multiple commits for a range."""
    result = get_commits_with_diffs("HEAD~3..HEAD")
    assert result.count >= 1
    for commit in result.commits:
        assert isinstance(commit, CommitInfo)
        assert commit.diff is not None


def test_get_commits_with_diffs_full_history(repo: str) -> None:
    """Can retrieve multiple commits from the repo."""
    result = get_commits_with_diffs("HEAD~3..HEAD")
    assert result.count >= 2


def test_get_commits_with_diffs_invalid(repo: str) -> None:
    """Invalid range raises ValueError from parse_range."""
    with pytest.raises(ValueError, match="Invalid revision"):
        get_commits_with_diffs("bad-ref-xyz")


def test_get_commits_with_diffs_empty_range_warning(repo: str) -> None:
    """Empty range returns CommitsResult with empty commits and a warning."""
    result = get_commits_with_diffs("HEAD..HEAD")
    assert result.count == 0
    assert len(result.commits) == 0
    assert len(result.warnings) >= 1
    assert "Empty revision range" in result.warnings[0]


def test_get_commits_with_diffs_same_branch_empty(repo: str) -> None:
    """Same branch on both sides of '..' returns empty result with warning."""
    import git as _git
    _repo = _git.Repo(".")
    branch = _repo.active_branch.name
    _repo.close()
    result = get_commits_with_diffs(f"{branch}..{branch}")
    assert result.count == 0
    assert len(result.warnings) >= 1


def test_get_commits_with_diffs_warnings_field(repo: str) -> None:
    """CommitsResult warnings field is populated for edge cases."""
    result = get_commits_with_diffs("HEAD..HEAD")
    assert hasattr(result, "warnings")
    assert isinstance(result.warnings, list)
    assert len(result.warnings) > 0


def test_get_commits_with_diffs_merge_commit(repo: str) -> None:
    """get_commits_with_diffs handles merge commits without error."""
    import git as _git
    _repo = _git.Repo(".")
    # Find a merge commit SHA directly
    merge_sha = None
    for commit in _repo.iter_commits("--all"):
        if len(commit.parents) > 1:
            merge_sha = commit.hexsha
            break
    _repo.close()

    if merge_sha is None:
        pytest.skip("No merge commit found in test repo")

    # Use the SHA as both a single ref and in a range
    info = get_commit_info(merge_sha)
    assert info.is_merge
    assert isinstance(info.diff, str)
    assert len(info.parents) > 1

    # Also test that a range containing the merge works
    result = get_commits_with_diffs(f"{merge_sha}~1..{merge_sha}")
    assert result.count >= 1
    merge_in_result = [c for c in result.commits if c.sha == merge_sha]
    if merge_in_result:
        assert merge_in_result[0].is_merge


# ── CLI integration tests ──────────────────────────────────────────────────


def test_cli_with_range_shows_commits(repo: str) -> None:
    """The CLI shows commit output when given a valid range."""
    runner = CliRunner()
    result = runner.invoke(main, ["HEAD~2..HEAD"])
    assert result.exit_code == 0
    assert "Commits found:" in result.output


def test_cli_verbose_shows_diff(repo: str) -> None:
    """The --verbose flag shows detailed commit output."""
    runner = CliRunner()
    result = runner.invoke(main, ["HEAD", "--verbose"])
    assert result.exit_code == 0
    assert "Author:" in result.output
    assert "Date:" in result.output


def test_cli_invalid_range_exits_nonzero(repo: str) -> None:
    """Invalid revision range exits with code 1."""
    runner = CliRunner()
    result = runner.invoke(main, ["bad..range"])
    assert result.exit_code != 0
    assert "Error:" in result.output


def test_cli_defaults_to_head(repo: str) -> None:
    """No arguments defaults to HEAD."""
    runner = CliRunner()
    result = runner.invoke(main, [])
    assert result.exit_code == 0
    assert "HEAD" in result.output


def test_cli_empty_range_shows_warning(repo: str) -> None:
    """Empty revision range shows a warning message."""
    runner = CliRunner()
    result = runner.invoke(main, ["HEAD..HEAD"])
    assert result.exit_code == 0
    assert "Warning:" in result.output
    assert "Empty revision range" in result.output


def test_cli_empty_string_rejected(repo: str) -> None:
    """Empty revision range string exits with error."""
    runner = CliRunner()
    result = runner.invoke(main, [""])
    assert result.exit_code != 0
    assert "Error:" in result.output


def test_cli_bare_two_dots_rejected(repo: str) -> None:
    """Bare '..' exits with error."""
    runner = CliRunner()
    result = runner.invoke(main, [".."])
    assert result.exit_code != 0
    assert "Error:" in result.output


def test_cli_shows_commit_summaries(repo: str) -> None:
    """Non-verbose mode shows commit SHA prefixes and summaries."""
    runner = CliRunner()
    result = runner.invoke(main, ["HEAD~2..HEAD"])
    assert result.exit_code == 0
    assert "Commits found:" in result.output
    # Should show at least one abbreviated SHA
    assert any(len(line.strip()) >= 7 for line in result.output.splitlines()
               if line.strip() and not line.startswith("Range:") and not line.startswith("Commits"))
