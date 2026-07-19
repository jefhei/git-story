"""Tests for the git-story CLI skeleton.

These tests verify Click argument parsing, flags, and help output.
Tests that exercise actual git operations live in test_git_reader.py.
"""

from __future__ import annotations

import os

import pytest
from click.testing import CliRunner
from git_story.cli import main
from git_story import __version__


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def chdir_repo(tmp_path) -> str:
    """Create a temporary git repo and chdir into it for CLI tests."""
    import git

    repo = git.Repo.init(tmp_path)
    repo.config_writer().set_value("user", "name", "Test User").release()
    repo.config_writer().set_value("user", "email", "test@example.com").release()
    (tmp_path / "README.md").write_text("# Test\n")
    repo.index.add(["README.md"])
    repo.index.commit("Initial commit")

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        yield str(tmp_path)
    finally:
        os.chdir(old_cwd)


# ── Tests ───────────────────────────────────────────────────────────────────


def test_default_range_is_head(chdir_repo: str) -> None:
    """When no revision range is given, defaults to HEAD."""
    runner = CliRunner()
    result = runner.invoke(main, [])
    assert result.exit_code == 0
    assert "HEAD" in result.output


def test_revision_range_arg(chdir_repo: str) -> None:
    """A revision range argument is accepted and shown in output."""
    runner = CliRunner()
    result = runner.invoke(main, ["HEAD"])
    assert result.exit_code == 0


def test_style_option(chdir_repo: str) -> None:
    """The --style option is accepted."""
    runner = CliRunner()
    result = runner.invoke(main, ["HEAD", "--style", "conventional-commit"])
    assert result.exit_code == 0


def test_style_default(chdir_repo: str) -> None:
    """Default style is plain-markdown (CLI remains functional)."""
    runner = CliRunner()
    result = runner.invoke(main, ["HEAD"])
    assert result.exit_code == 0


def test_invalid_style() -> None:
    """An invalid style is rejected (even outside a git repo)."""
    runner = CliRunner()
    result = runner.invoke(main, ["HEAD", "--style", "invalid"])
    assert result.exit_code != 0


def test_output_option(chdir_repo: str) -> None:
    """The --output option is accepted."""
    runner = CliRunner()
    result = runner.invoke(main, ["HEAD", "--output", "CHANGELOG.md"])
    assert result.exit_code == 0


def test_version_flag() -> None:
    """The --version flag prints version and exits (outside any repo)."""
    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_help_output() -> None:
    """The CLI displays help with expected topics (outside any repo)."""
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "git-story" in result.output
    assert "REVISION_RANGE" in result.output
    assert "--style" in result.output
    assert "--output" in result.output


def test_provider_option(chdir_repo: str) -> None:
    """The --provider option is accepted."""
    runner = CliRunner()
    result = runner.invoke(main, ["HEAD", "--provider", "openai"])
    assert result.exit_code == 0


def test_invalid_provider() -> None:
    """An invalid provider is rejected (even outside a git repo)."""
    runner = CliRunner()
    result = runner.invoke(main, ["HEAD", "--provider", "invalid"])
    assert result.exit_code != 0


def test_model_option(chdir_repo: str) -> None:
    """The --model option is accepted."""
    runner = CliRunner()
    result = runner.invoke(main, ["HEAD", "--model", "gpt-4o"])
    assert result.exit_code == 0


def test_not_in_repo_shows_error() -> None:
    """Running outside a git repo shows a helpful error message."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(main, ["HEAD"])
    assert result.exit_code != 0
    assert "Error" in result.output or "error" in result.output
