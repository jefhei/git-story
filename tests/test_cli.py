"""Tests for the git-story CLI skeleton."""

from click.testing import CliRunner
from git_story.cli import main
from git_story import __version__


def test_default_range_is_head() -> None:
    """When no revision range is given, defaults to HEAD."""
    runner = CliRunner()
    result = runner.invoke(main, [])
    assert result.exit_code == 0
    assert "HEAD" in result.output


def test_revision_range_arg() -> None:
    """A revision range argument is accepted and shown in output."""
    runner = CliRunner()
    result = runner.invoke(main, ["main..feature"])
    assert result.exit_code == 0
    assert "main..feature" in result.output


def test_style_option() -> None:
    """The --style option is accepted."""
    runner = CliRunner()
    result = runner.invoke(main, ["main..feature", "--style", "conventional-commit"])
    assert result.exit_code == 0
    assert "conventional-commit" in result.output


def test_style_default() -> None:
    """Default style is plain-markdown."""
    runner = CliRunner()
    result = runner.invoke(main, ["main..feature"])
    assert result.exit_code == 0
    assert "plain-markdown" in result.output


def test_invalid_style() -> None:
    """An invalid style is rejected."""
    runner = CliRunner()
    result = runner.invoke(main, ["main..feature", "--style", "invalid"])
    assert result.exit_code != 0


def test_output_option() -> None:
    """The --output option is accepted."""
    runner = CliRunner()
    result = runner.invoke(main, ["main..feature", "--output", "CHANGELOG.md"])
    assert result.exit_code == 0


def test_version_flag() -> None:
    """The --version flag prints version and exits."""
    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_help_output() -> None:
    """The CLI displays help with expected topics."""
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "git-story" in result.output
    assert "REVISION_RANGE" in result.output
    assert "--style" in result.output
    assert "--output" in result.output


def test_provider_option() -> None:
    """The --provider option is accepted."""
    runner = CliRunner()
    result = runner.invoke(main, ["main..feature", "--provider", "openai"])
    assert result.exit_code == 0
    assert "openai" in result.output


def test_invalid_provider() -> None:
    """An invalid provider is rejected."""
    runner = CliRunner()
    result = runner.invoke(main, ["main..feature", "--provider", "invalid"])
    assert result.exit_code != 0


def test_model_option() -> None:
    """The --model option is accepted."""
    runner = CliRunner()
    result = runner.invoke(main, ["main..feature", "--model", "gpt-4o"])
    assert result.exit_code == 0
    assert "gpt-4o" in result.output
