"""git-story CLI — transform git history into PR-ready prose.

Usage:
    git-story <rev_range>
    git-story <rev1>..<rev2>
    git-story --version
"""

import click


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
def main(
    revision_range: str,
    output: str | None,
    style: str,
    model: str | None,
    provider: str | None,
) -> None:
    """git-story — Transform git history into PR-ready prose using LLMs.

    REVISION_RANGE is a git revision range like main..feature or
    v1.0..v1.1. If omitted, defaults to HEAD.

    Examples:

        git-story main..feature

        git-story v1.0..v1.1 --style conventional-commit --output CHANGELOG.md
    """
    click.echo(
        f"git-story: processing range '{revision_range}' "
        f"(--style={style}, --output={output}, "
        f"--provider={provider}, --model={model})"
    )


if __name__ == "__main__":
    main()
