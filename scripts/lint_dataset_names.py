#!/usr/bin/env python3
"""Reject untagged ``bench-v1-K{k}-v226-lineage`` dataset references.

Slice FARM-EXP.12 makes the PLM axis explicit in dataset naming: the
canonical form is now ``bench-v1-K{k}-v{val_band}-lineage-{plm_short}``
(e.g. ``bench-v1-K5-v226-lineage-prostt5``). Untagged references like
``bench-v1-K5-v226-lineage`` are PLM-blind and lose information about
which protein-language-model embedding the dataset was built against.

This linter scans the repository for the legacy untagged form and
emits one ``file:line:col`` line per offence. It is wired into both
``.pre-commit-config.yaml`` (block-on-commit) and the CI workflow
(block-on-PR).

A short allowlist permits prose that explicitly discusses the rename
itself (ADR D36, CHANGELOG entries that note the migration) and the
legacy ``-mini`` smoke-test dataset (``bench-v1-K5-v226-lineage-mini``
is the historical lineage-mini dataset, which has its own naming
convention and is out of scope for the per-PLM rename).

Exit codes:
    0 - clean (no offences)
    1 - offences found
    2 - invocation error
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Match ``bench-v1-K{k}-v{val_band}-lineage`` NOT followed by
# ``-{plm_short}`` or ``-mini``. The negative lookahead permits the
# canonical per-PLM form and the legacy mini smoke dataset.
#
# Allowed PLM short keys come from the canonical 8-PLM list (ADR D35)
# plus the ``esmc_300m`` baseline kept for chapter-6 single-PLM
# comparisons.
ALLOWED_SUFFIXES: tuple[str, ...] = (
    "esm2_150m",
    "esm2_650m",
    "esm2_3b",
    "prot_t5",
    "prostt5",
    "ankh_base",
    "ankh_large",
    "esmc_600m",
    "esmc_300m",
    "mini",
)

_suffix_alt = "|".join(re.escape(s) for s in ALLOWED_SUFFIXES)
PATTERN = re.compile(
    r"bench-v1-K\d+-v\d+-lineage(?!-(?:" + _suffix_alt + r")\b)(?![\w-])"
)

# File extensions worth scanning. Code files (.py) legitimately
# reference the canonical name as strings, so they ARE included; the
# rename slice rewrites those references too.
SCAN_EXTENSIONS: frozenset[str] = frozenset({
    ".md", ".rst", ".tex", ".txt", ".yaml", ".yml", ".py", ".csv", ".json",
})

# Directories the walker skips entirely.
SKIP_DIRS: frozenset[str] = frozenset({
    ".git", ".venv", "venv", "node_modules",
    "_build", "build", "dist", "runs",
    ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "__pycache__",
})

# Default scan roots, relative to cwd.
DEFAULT_ROOTS: tuple[str, ...] = (
    "champions.md",
    "spec_catalog.md",
    "EXPERIMENTS.md",
    "experiments",
    "scripts",
    "src",
    "tests",
    "lb2_multiseed_sweep.py",
    "v27_binary_multiseed_sweep.py",
)

# Files the linter must not flag: meta documents that discuss the
# rename itself, the linter source, and the linter test fixtures.
ALLOWLIST_PATHS: frozenset[str] = frozenset({
    "scripts/lint_dataset_names.py",
    "tests/test_lint_dataset_names.py",
    "CHANGELOG.md",
    # F-DATA-PACK.2 README generator + tests reference the legacy
    # ``bench-v1-K5-v226-lineage`` form to exercise the ghost prostt5
    # alias path (manifest stores the PLM blind name, registry stores
    # the canonical per PLM form). These are not new untagged references
    # but documentation of an existing aliasing landmine.
    "scripts/generate_dataset_readme.py",
    "tests/test_generate_dataset_readme.py",
    # F-LAFA-IA.2 palanca-1 probe trains on the real on-disk legacy study
    # dataset ``bench-v1-K5-v226-lineage`` (the v26-binary champion's source,
    # PLM-blind blind name), not a new untagged reference. Same landmine the
    # README generator documents.
    "experiments/lafa_ia_palanca1/probe_lk_bpo.py",
})


def iter_files(roots: list[Path]) -> list[Path]:
    out: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        if root.is_file():
            if root.suffix in SCAN_EXTENSIONS:
                out.append(root)
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if any(part in SKIP_DIRS for part in path.parts):
                continue
            if path.suffix not in SCAN_EXTENSIONS:
                continue
            out.append(path)
    return out


def scan_file(path: Path, repo_root: Path) -> list[tuple[Path, int, int, str]]:
    rel = path.relative_to(repo_root) if path.is_absolute() else path
    if str(rel) in ALLOWLIST_PATHS:
        return []
    offences: list[tuple[Path, int, int, str]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for match in PATTERN.finditer(line):
            offences.append((rel, lineno, match.start() + 1, match.group(0)))
    return offences


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Files or directories to scan (default: repo-wide).",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Repository root (default: cwd).",
    )
    args = parser.parse_args(argv)

    repo_root = args.root.resolve()
    if args.paths:
        roots = [p.resolve() for p in args.paths]
    else:
        roots = [repo_root / r for r in DEFAULT_ROOTS]

    files = iter_files(roots)
    all_offences: list[tuple[Path, int, int, str]] = []
    for path in files:
        all_offences.extend(scan_file(path, repo_root))

    if not all_offences:
        return 0
    for rel, lineno, col, token in all_offences:
        print(f"{rel}:{lineno}:{col}: untagged dataset reference '{token}'", file=sys.stderr)
    print(
        f"\n{len(all_offences)} offence(s). Rewrite as "
        "'bench-v1-K{k}-v{val_band}-lineage-{plm_short}' (see ADR D36 "
        "and CHANGELOG entry for FARM-EXP.12).",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
