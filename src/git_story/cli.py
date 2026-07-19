"""git-story CLI — transform git history into PR-ready prose.

Usage:
    git-story <rev_range>
    git-story <rev1>..<rev2>
    git-story --version
"""

from __future__ import annotations

import sys

import click

from git_story.git_reader import CommitsResult, get_commits_with_diffs

# ── helper to format a summary line ────────────────────────────────────────


def _summarise(result: CommitsResult) -> str:
    """Return a one-line summary string."""
    parts = [f"Range: {result.range}"]
    parts.append(f"Commits found: {result.count}")
    if result.errors:
        parts.append(f"Errors: {len(result.errors)}")
    return " | ".join(parts)


# ── CLI command ────────────────────────────────────────────────────────────


@click.command()
@click.version_option(version="0.1.0", prog_name="git-story")
@click.argument("revision_range", required=False, default="HEAD")
@click.option(
    "--output",
    "-o",
    type=click.Path(writable=True),
    default=None,
    help="Write output to file instead of stdout.",
)
@click.option(
    "--style",
    type=click.Choice(["plain-markdown", "conventional-commit", "json"]),
    default="plain-markdown",
    help="Output style (default: plain-markdown).",
)
@click.option(
    "--model",
    help="LLM model identifier (e.g. gpt-4o, claude-3-opus).",
    default=None,
)
@click.option(
    "--provider",
    type=click.Choice(["openai", "anthropic", "ollama"]),
    default=None,
    help="LLM provider backend.",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Show detailed output including commit summaries and diffs.",
)
def main(
    revision_range: str,
    output: str | None,
    style: str,
    model: str | None,
    provider: str | None,
    verbose: bool,
) -> None:
    """git-story — Transform git history into PR-ready prose using LLMs.

    REVISION_RANGE is a git revision range like main..feature or
    v1.0..v1.1. If omitted, defaults to HEAD (last commit only).
    """
    try:
        result = get_commits_with_diffs(revision_range)
    except ValueError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    click.echo(_summarise(result))

    if verbose:
        for commit in result.commits:
            click.echo("")
            click.echo(f"Commit: {commit.sha[:8]}")
            click.echo(f"Author: {commit.author_name} <{commit.author_email}>")
            click.echo(f"Date:   {commit.authored_at.isoformat()}")
            if commit.is_merge:
                click.echo("Merge:  yes")
            click.echo(f"")
            click.echo(f"    {commit.message_summary}")
            click.echo(f"")
            if commit.diff:
                # Show first 20 lines of diff as a preview
                diff_lines = commit.diff.splitlines()
                preview = diff_lines[:20]
                for line in preview:
                    click.echo(f"  {line}")
                if len(diff_lines) > 20:
                    click.echo(f"  ... ({len(diff_lines) - 20} more diff lines)")
    else:
        for commit in result.commits:
            merge_tag = " (merge)" if commit.is_merge else ""
            click.echo(f"  {commit.sha[:8]}  {commit.message_summary}{merge_tag}")


if __name__ == "__main__":
    main()
