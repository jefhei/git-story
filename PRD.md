# git-story — Product Requirements Document

**Status:** Draft v1.0
**Owner:** Product Management
**Last Updated:** 2025

---

## 1. Executive Summary

**git-story** is a command-line tool that transforms raw git history into PR-ready prose. Given a commit range (e.g., `git-story main..feature`), it reads commits and diffs, uses an LLM to semantically group changes by theme, and outputs a polished markdown changelog plus structured JSON.

Unlike template-driven tools (conventional-changelog, git-cliff) that mechanically parse commit prefixes, git-story understands *what changed and why*. This matters more than ever: AI coding agents now generate massive volumes of commits that need coherent, human-readable documentation. git-story fills a clear gap — a dead-simple CLI that turns noisy git history into narrative.

The MVP ships in 1-2 weeks as a Python CLI with pluggable LLM backends (OpenAI, Anthropic, Ollama for local/private use).

---

## 2. Problem Statement

Developers spend disproportionate time translating git history into human-readable documentation:

- **Changelogs** require manually scanning dozens of commits, deduplicating, and rewriting into user-facing language.
- **PR descriptions** are frequently thin or absent because writing them from scratch is tedious.
- **Release notes** demand grouping related changes into themes — work that existing tools don't do.

**Why existing tools fall short:**

- **conventional-changelog / git-cliff** are template engines. They rely on strict commit conventions and produce mechanical, list-shaped output with zero semantic grouping. Output is stale and reads like a git log, not a story.
- They assume disciplined, human-authored commit messages. That assumption is collapsing.

**Why now:**

- **AI coding agents** (Claude Code, Codex, Gemini CLI) generate an explosion of commits — often small, numerous, and inconsistently messaged. The gap between commit volume and documentation quality is widening fast.
- **Demand is proven.** gitlogue (4,856★) demonstrates strong appetite for git history tooling.
- **LLMs make semantic understanding cheap.** Grouping commits by theme and writing prose is now a solved primitive — nobody has packaged it into a focused CLI.

**The core insight:** git history contains everything needed to write great documentation. The missing piece is semantic understanding — precisely what an LLM provides.

---

## 3. Target Audience

| Segment | Need | Usage Pattern |
|---|---|---|
| **Individual developers** | Fast PR descriptions & personal changelogs | Ad-hoc, per-branch |
| **Open-source maintainers** | Release notes from many contributors | Per-release, CI-integrated |
| **Engineering teams** | Consistent changelogs across repos | CI pipeline, pre-release |
| **AI-agent power users** | Documentation for high-volume auto-generated commits | Frequent, high-commit-count ranges |
| **Tech leads / EMs** | Human-readable summaries of sprint work | Weekly/bi-weekly |

**Primary persona (MVP):** The individual developer or maintainer who wants `git-story main..feature` to produce a clean changelog they can paste into a PR or release without manual editing.

---

## 4. Feature Requirements

### Must-Have (MVP)

| ID | Feature | Description |
|---|---|---|
| F1 | **Commit range parsing** | Accept `git-story <rev1>..<rev2>` syntax; read commits + diffs via gitpython. |
| F2 | **LLM semantic grouping** | Send commits/diffs to LLM; group by theme (features, fixes, refactors, docs, etc.). |
| F3 | **Markdown output** | Produce a clean, PR-ready markdown changelog grouped by theme. |
| F4 | **Structured JSON output** | Emit machine-readable JSON with themes, entries, and metadata. |
| F5 | **`--style` flag** | Support `conventional-commit` and `plain-markdown` output styles. |
| F6 | **`--output` flag** | Write to file or stdout. |
| F7 | **Multi-backend LLM** | OpenAI, Anthropic, and Ollama (local) via config/env vars. |
| F8 | **API key & config handling** | Read from env vars / config file; clear error if missing. |
| F9 | **Diff truncation / token budgeting** | Handle large ranges without exceeding context limits. |

### Nice-to-Have (Post-MVP)

| ID | Feature | Description |
|---|---|---|
| N1 | **Fixup commit detection** | Detect unsquashed `fixup!`/`squash!` commits and flag or collapse them. |
| N2 | **Terminal UI** | Interactive review/edit of grouped output via rich/textual. |
| N3 | **PR description mode** | `--mode pr` optimized for single-PR summaries. |
| N4 | **Custom prompt templates** | User-supplied prompts / tone control. |
| N5 | **CI integration recipes** | GitHub Actions / GitLab CI snippets. |
| N6 | **Caching** | Cache LLM responses per commit hash to avoid re-billing. |
| N7 | **Tag-to-tag mode** | `git-story v1.0..v1.1` release-note optimization. |
| N8 | **Rust port** | Performance-focused rewrite for large repos. |

---

## 5. Technical Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      git-story CLI                        │
│                    (Click / argparse)                     │
└───────────────┬──────────────────────────┬───────────────┘
                │                          │
        ┌───────▼────────┐        ┌────────▼─────────┐
        │  Git Reader    │        │  Config / Auth   │
        │  (gitpython)   │        │  (env, .toml)    │
        │  commits+diffs │        └──────────────────┘
        └───────┬────────┘
                │
        ┌───────▼─────────────┐
        │  Preprocessor       │
        │  - diff truncation  │
        │  - token budgeting  │
        │  - fixup detection  │
        └───────┬─────────────┘
                │
        ┌───────▼─────────────┐
        │  LLM Engine         │  ← pluggable backend interface
        │  OpenAI / Anthropic │
        │  / Ollama           │
        │  semantic grouping  │
        └───────┬─────────────┘
                │
        ┌───────▼─────────────┐
        │  Renderer           │
        │  - markdown styles  │
        │  - JSON schema      │
        └───────┬─────────────┘
                │
        ┌───────▼─────────────┐
        │  Output (stdout/file)│
        └─────────────────────┘
```

**Key components:**

- **Git Reader:** Uses gitpython to resolve the commit range and extract commit metadata + diffs. Handles merge commits and empty ranges gracefully.
- **Preprocessor:** Normalizes and truncates diffs, applies a token budget, and (optionally) detects fixup/squash commits. Chunks large ranges for map-reduce summarization if needed.
- **LLM Engine:** Abstract `LLMBackend` interface with concrete adapters. A single well-tested prompt returns structured groupings (JSON-mode where available).
- **Renderer:** Converts the LLM's structured output into markdown (per `--style`) and canonical JSON.
- **Config:** Layered config — CLI flags > env vars > `~/.config/git-story/config.toml` > defaults.

**Tech decisions:**

- **MVP language:** Python (fast iteration, gitpython, mature LLM SDKs).
- **CLI framework:** Click.
- **Packaging:** `pip install git-story` / `pipx`.
- **Rust:** Deferred; considered only if performance on huge repos becomes a limiter.

**JSON output schema (draft):**

```json
{
  "range": "main..feature",
  "generated_at": "ISO8601",
  "themes": [
    {
      "name": "Features",
      "summary": "string",
      "entries": [
        {"title": "string", "description": "string", "commits": ["sha"]}
      ]
    }
  ],
  "warnings": ["unsquashed fixup commit detected: <sha>"]
}
```

---

## 6. Milestones & Timeline

| Milestone | Goal | Duration |
|---|---|---|
| **M1 — Foundation** | Project scaffold, git reading, CLI skeleton | Days 1–2 |
| **M2 — LLM Core** | Backend abstraction, grouping prompt, one working backend | Days 3–5 |
| **M3 — Output & Styles** | Markdown + JSON rendering, `--style`, `--output` | Days 6–7 |
| **M4 — Polish & Ship** | Multi-backend, token budgeting, docs, packaging | Days 8–10 |
| **M5 — Enhancements** | Fixup detection, TUI, caching, CI recipes | Post-MVP |

**MVP target:** End of Week 2 (Days 8–10), with a functional prototype by Day 7.

---

## 7. Success Metrics

**Adoption**
- 500 GitHub stars within 3 months; 2,000 within 6 months.
- 1,000+ PyPI downloads/month by month 2.

**Quality / Utility**
- ≥70% of generated changelogs used with *zero manual edits* (self-reported / survey).
- Median generation time < 30s for a 50-commit range (excluding LLM latency).
- < 5% failure rate on valid commit ranges.

**Engagement**
- ≥3 external contributors within 3 months.
- Ollama (local) backend used by ≥20% of users (signals privacy demand).

**Leading indicators**
- Time-to-first-successful-run < 5 minutes from install.
- README → run conversion tracked via docs analytics.

---

## Task List

```json
[
  {
    "milestone": "M1 - Foundation",
    "tasks": [
      {"id": "M1.1", "title": "Initialize Python project (pyproject, Click CLI skeleton, packaging)", "priority": "high", "estimate": "0.5d", "dependencies": []},
      {"id": "M1.2", "title": "Implement git range parsing and commit/diff extraction via gitpython", "priority": "high", "estimate": "1d", "dependencies": ["M1.1"]},
      {"id": "M1.3", "title": "Handle edge cases: empty ranges, merge commits, invalid revs", "priority": "high", "estimate": "0.5d", "dependencies": ["M1.2"]},
      {"id": "M1.4", "title": "Config layering (CLI flags > env > toml > defaults) + auth error handling", "priority": "high", "estimate": "0.5d", "dependencies": ["M1.1"]}
    ]
  },
  {
    "milestone": "M2 - LLM Core",
    "tasks": [
      {"id": "M2.1", "title": "Define LLMBackend abstract interface", "priority": "high", "estimate": "0.5d", "dependencies": ["M1.1"]},
      {"id": "M2.2", "title": "Implement OpenAI backend adapter (JSON mode)", "priority": "high", "estimate": "0.5d", "dependencies": ["M2.1"]},
      {"id": "M2.3", "title": "Design and iterate semantic grouping prompt", "priority": "high", "estimate": "1d", "dependencies": ["M2.1", "M1.2"]},
      {"id": "M2.4", "title": "Preprocessor: diff normalization, truncation, token budgeting", "priority": "high", "estimate": "1d", "dependencies": ["M1.2"]},
      {"id": "M2.5", "title": "Map-reduce chunking for large commit ranges", "priority": "medium", "estimate": "0.5d", "dependencies": ["M2.4", "M2.3"]},
      {"id": "M2.6", "title": "Parse and validate LLM structured output against JSON schema", "priority": "high", "estimate": "0.5d", "dependencies": ["M2.3"]}
    ]
  },
  {
    "milestone": "M3 - Output & Styles",
    "tasks": [
      {"id": "M3.1", "title": "Define canonical JSON output schema", "priority": "high", "estimate": "0.5d", "dependencies": ["M2.6"]},
      {"id": "M3.2", "title": "Markdown renderer (plain-markdown style)", "priority": "high", "estimate": "0.5d", "dependencies": ["M3.1"]},
      {"id": "M3.3", "title": "Conventional-commit style renderer", "priority": "medium", "estimate": "0.5d", "dependencies": ["M3.2"]},
      {"id": "M3.4", "title": "Implement --style and --output flags", "priority": "high", "estimate": "0.5d", "dependencies": ["M3.2"]},
      {"id": "M3.5", "title": "JSON output emission to stdout/file", "priority": "high", "estimate": "0.25d", "dependencies": ["M3.1"]}
    ]
  },
  {
    "milestone": "M4 - Polish & Ship",
    "tasks": [
      {"id": "M4.1", "title": "Add Anthropic backend adapter", "priority": "high", "estimate": "0.5d", "dependencies": ["M2.1"]},
      {"id": "M4.2", "title": "Add Ollama (local) backend adapter", "priority": "high", "estimate": "0.5d", "dependencies": ["M2.1"]},
      {"id": "M4.3", "title": "End-to-end tests on sample repos + LLM response fixtures", "priority": "high", "estimate": "1d", "dependencies": ["M3.4", "M4.1"]},
      {"id": "M4.4", "title": "Error handling, retries, and rate-limit backoff", "priority": "medium", "estimate": "0.5d", "dependencies": ["M2.2"]},
      {"id": "M4.5", "title": "Write README, usage docs, and quickstart", "priority": "high", "estimate": "0.5d", "dependencies": ["M3.4"]},
      {"id": "M4.6", "title": "Publish to PyPI (pip/pipx installable)", "priority": "high", "estimate": "0.5d", "dependencies": ["M4.3", "M4.5"]}
    ]
  },
  {
    "milestone": "M5 - Enhancements",
    "tasks": [
      {"id": "M5.1", "title": "Detect unsquashed fixup!/squash! commits and emit warnings", "priority": "medium", "estimate": "0.5d", "dependencies": ["M1.2"]},
      {"id": "M5.2", "title": "LLM response caching keyed by commit hash", "priority": "medium", "estimate": "1d", "dependencies": ["M2.6"]},
      {"id": "M5.3", "title": "PR description mode (--mode pr)", "priority": "medium", "estimate": "1d", "dependencies": ["M3.2"]},
      {"id": "M5.4", "title": "Interactive TUI review/edit via textual", "priority": "low", "estimate": "3d", "dependencies": ["M3.2"]},
      {"id": "M5.5", "title": "Custom prompt templates and tone control", "priority": "low", "estimate": "1d", "dependencies": ["M2.3"]},
      {"id": "M5.6", "title": "CI integration recipes (GitHub Actions / GitLab CI)", "priority": "low", "estimate": "0.5d", "dependencies": ["M4.6"]},
      {"id": "M5.7", "title": "Evaluate Rust port for large-repo performance", "priority": "low", "estimate": "5d", "dependencies": ["M4.6"]}
    ]
  }
]
```