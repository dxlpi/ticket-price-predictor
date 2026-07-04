#!/usr/bin/env python3
"""Validate that file paths referenced in context/docs actually exist.

Guards against "hallucinated paths" — stale references in CLAUDE.md, AGENTS.md,
README files, and .claude/rules that point at files which have moved or been
deleted. A stale reference is worse than a missing one: an agent will follow it.

Scans markdown context files for path-like tokens and relative links, then
verifies each resolves against the repo root, the referencing file's directory,
or the source package root (the docs use package-relative shorthand such as
``ml/features/event_pricing.py`` for ``src/ticket_price_predictor/ml/...``).

Exit status: 0 if all references resolve, 1 if any broken reference is found.
Pure stdlib — no external dependencies.

Usage:
    python scripts/check_doc_paths.py            # scan default doc set
    python scripts/check_doc_paths.py --quiet    # only print on failure
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = REPO / "src" / "ticket_price_predictor"

# Files whose path references we validate.
CONTEXT_GLOBS = (
    "CLAUDE.md",
    "MEMORY.md",
    "README.md",
    ".claude/CLAUDE.md",
    ".claude/rules/*.md",
    "docs/**/*.md",
    "evals/*.md",
    "src/**/AGENTS.md",
    "src/**/CLAUDE.md",
)

# Skip generated / vendored trees.
IGNORE_PARTS = {
    ".venv", "node_modules", ".git", "__pycache__", ".mypy_cache",
    ".ruff_cache", ".pytest_cache", "worktrees",
}

# Extensions, longest-first so ``.json`` isn't shortened to ``.js``; a trailing
# lookahead keeps ``.js`` from matching inside ``.json``.
_EXT = (
    r"(?:tsx|ts|jsx|jsonl|json|js|py|md|sql|yaml|yml|toml|html|css|sh|cfg|ini|joblib)"
    r"(?![A-Za-z0-9])"
)
# Path-like token: optional ./ or ../ prefix, or a dotted/plain dir segment.
RE_PATH = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"((?:\.{1,2}/|\.?[A-Za-z0-9_]+/)[A-Za-z0-9_./-]+\." + _EXT + r")"
)
# Markdown relative links: [text](path) that aren't URLs or bare anchors.
RE_LINK = re.compile(r"\[[^\]]+\]\((?!https?://|#|mailto:)([^)#\s]+)(?:#[^)]*)?\)")
RE_URL = re.compile(r"https?://\S+")

# Obvious non-references — example placeholders, not real paths.
PLACEHOLDER = ("your_", "path/to", "<", ">", "example", "...")

# Runtime artifact trees that are intentionally gitignored (trained model
# binaries, collected datasets). Docs legitimately reference *where* these are
# written, but they do not exist in a fresh checkout — validating them would
# make CI depend on local, uncommitted state. Skip references into these trees.
ARTIFACT_PREFIXES = ("data/models/", "data/snapshots/")


def iter_context_files() -> list[Path]:
    seen: set[Path] = set()
    for pattern in CONTEXT_GLOBS:
        for p in REPO.glob(pattern):
            if p.is_file() and not any(part in IGNORE_PARTS for part in p.parts):
                seen.add(p)
    return sorted(seen)


def candidate_refs(text: str) -> set[str]:
    text = RE_URL.sub(" ", text)  # drop URLs so their path segments don't leak in
    refs = set(RE_PATH.findall(text))
    refs.update(RE_LINK.findall(text))
    out: set[str] = set()
    for r in refs:
        r = r.strip().rstrip(".,;:)")
        if not r or any(tok in r for tok in PLACEHOLDER):
            continue
        if r.lstrip("./").startswith(ARTIFACT_PREFIXES):
            continue  # gitignored runtime artifact — not expected in a checkout
        out.add(r)
    return out


def resolves(ref: str, source: Path) -> bool:
    # Try repo root, the referencing file's directory, and the package root.
    # (base / ref).exists() lets the OS collapse any ./ or ../ segments.
    bases = [REPO, source.parent]
    if not ref.startswith((".", "/")):
        bases.append(PACKAGE_ROOT)
    return any((base / ref).exists() for base in bases)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--quiet", action="store_true", help="print only on failure")
    args = ap.parse_args()

    broken: list[tuple[Path, str]] = []
    total = 0
    for f in iter_context_files():
        text = f.read_text(errors="ignore")
        for ref in candidate_refs(text):
            total += 1
            if not resolves(ref, f):
                broken.append((f.relative_to(REPO), ref))

    if broken:
        print(f"✗ {len(broken)} broken doc reference(s) out of {total} checked:\n")
        for src, ref in sorted(broken):
            print(f"  {src}: {ref}")
        print("\nFix the reference or the path it points to before merging.")
        return 1

    if not args.quiet:
        print(f"✓ all {total} doc path references resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main())
