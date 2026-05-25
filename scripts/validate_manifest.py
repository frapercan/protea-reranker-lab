#!/usr/bin/env python3
"""Validate a frozen dataset ``manifest.json`` against the v2 schema.

Slice F-DATA-PACK.1 wires a release-gate on every
``bench-v1-K{k}-v{val_band}-lineage-{plm_short}`` dataset published by
PROTEA's ``export_research_dataset`` operation. The job is to catch
schema drift and placeholder digests BEFORE a parquet ships to Zenodo
or Hugging Face.

Two failure modes the validator must catch:

1. **Schema drift**: the producer adds or renames a top-level key
   (e.g. a new feature-family list) and the lab consumer silently
   ignores it. Pydantic's ``ManifestV1`` is intentionally permissive
   (``extra="ignore"``) so the lab can read forward-compatible
   manifests; this validator is strict and rejects any unknown key.

2. **Placeholder digests**: ``schema_sha`` and ``manifest_sha`` shipped
   with a literal ``"placeholder"`` (or equivalent) for 160 catalog
   cells (memory ``[[farm-exp-2-placeholder-digests]]``). A release
   gate is the right place to refuse those.

The validator runs as a standalone script (no pytest harness needed)
and is wired into ``.github/workflows/validate-manifests.yml`` so any
PR that touches a ``manifest.json`` is gated.

Exit codes:
    0 - clean (manifest valid)
    1 - validation failed
    2 - invocation error
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

# Required top-level keys. The exact set the v2 producer emits is the
# union of ``ManifestV1`` required fields plus the producer-stamped
# provenance triple (``producer_version``, ``producer_git_sha``,
# ``format``). Optional keys (``spec_hash``, ``parent_schema_sha``,
# ``feature_families``) are permitted but not required: the producer
# fills them when relevant. ``manifest_sha`` is NOT in the on-disk
# manifest: it is computed by the consumer from the file bytes; the
# registry row (Dataset.manifest_sha) carries it.
REQUIRED_KEYS: frozenset[str] = frozenset({
    "schema_version",
    "name",
    "k",
    "embedding_config_id",
    "ontology_snapshot_id",
    "train_snapshot_pairs",
    "eval_snapshot_pair",
    "schema_sha",
    "format",
})

OPTIONAL_KEYS: frozenset[str] = frozenset({
    "annotation_source",
    "n_train_rows",
    "n_eval_rows",
    "spec_hash",
    "parent_schema_sha",
    "feature_families",
    "producer_version",
    "producer_git_sha",
})

ALLOWED_KEYS: frozenset[str] = REQUIRED_KEYS | OPTIONAL_KEYS

# Schema version pinned to v2. Older v1 manifests are out of scope for
# the F-DATA-PACK deliverable; if a v3 lands, this constant moves and
# the test corpus is regenerated.
EXPECTED_SCHEMA_VERSION: str = "v2"

# ``schema_sha`` is the 12-hex digest of the canonical feature-name
# list (see PROTEA ``parquet_export._compute_schema_sha``). The
# canonical 8-PLM v226-lineage exports all carry ``6d97a624b8a7``.
SCHEMA_SHA_RE: re.Pattern[str] = re.compile(r"^[0-9a-f]{12}$")

# Producer git sha is the standard 40-hex SHA-1.
GIT_SHA_RE: re.Pattern[str] = re.compile(r"^[0-9a-f]{40}$")

# Tokens that betray a placeholder digest. The match is
# case-insensitive and substring-based so ``"PLACEHOLDER"``,
# ``"placeholder_v2"``, ``"tbd"`` all trip.
PLACEHOLDER_TOKENS: tuple[str, ...] = (
    "placeholder",
    "tbd",
    "todo",
    "fixme",
    "xxx",
    "n/a",
    "na",
    "null",
    "none",
)

# Canonical eval-band and snapshot pair regex (loose: just confirms the
# producer used a recognisable ``vNNN-vMMM`` form, not a free-text
# placeholder).
SNAPSHOT_PAIR_RE: re.Pattern[str] = re.compile(r"^v\d+-v\d+$")


def _is_placeholder(value: str) -> bool:
    """Return True if ``value`` looks like a placeholder digest.

    Two heuristics: literal placeholder tokens and all-zero / repeated
    single-char strings. We accept any sha-shaped string that does
    not match either heuristic.
    """
    lowered = value.strip().lower()
    if not lowered:
        return True
    for token in PLACEHOLDER_TOKENS:
        if token in lowered:
            return True
    if len(set(lowered)) == 1:
        return True
    return False


def _check_required_keys(manifest: dict[str, Any]) -> list[str]:
    missing = sorted(REQUIRED_KEYS - manifest.keys())
    return [f"missing required key: {k!r}" for k in missing]


def _check_unknown_keys(manifest: dict[str, Any]) -> list[str]:
    unknown = sorted(manifest.keys() - ALLOWED_KEYS)
    return [f"unknown top-level key: {k!r}" for k in unknown]


def _check_schema_version(manifest: dict[str, Any]) -> list[str]:
    sv = manifest.get("schema_version")
    if sv is None:
        return []
    if sv != EXPECTED_SCHEMA_VERSION:
        return [f"schema_version: expected {EXPECTED_SCHEMA_VERSION!r}, got {sv!r}"]
    return []


def _check_digests(manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    schema_sha = manifest.get("schema_sha")
    if isinstance(schema_sha, str):
        if _is_placeholder(schema_sha):
            errors.append(f"schema_sha is a placeholder: {schema_sha!r}")
        elif not SCHEMA_SHA_RE.match(schema_sha):
            errors.append(
                f"schema_sha must be 12 hex chars, got {schema_sha!r}"
            )
    elif schema_sha is not None:
        errors.append(f"schema_sha must be a string, got {type(schema_sha).__name__}")
    git_sha = manifest.get("producer_git_sha")
    if isinstance(git_sha, str):
        if _is_placeholder(git_sha):
            errors.append(f"producer_git_sha is a placeholder: {git_sha!r}")
        elif not GIT_SHA_RE.match(git_sha):
            errors.append(
                f"producer_git_sha must be 40 hex chars, got {git_sha!r}"
            )
    parent = manifest.get("parent_schema_sha")
    if isinstance(parent, str) and not _is_placeholder(parent):
        if not SCHEMA_SHA_RE.match(parent):
            errors.append(
                f"parent_schema_sha must be 12 hex chars, got {parent!r}"
            )
    elif isinstance(parent, str):
        errors.append(f"parent_schema_sha is a placeholder: {parent!r}")
    return errors


def _check_types(manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(manifest.get("name"), str) or not manifest.get("name"):
        errors.append("name must be a non-empty string")
    k = manifest.get("k")
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        errors.append(f"k must be a positive int, got {k!r}")
    for field in ("embedding_config_id", "ontology_snapshot_id"):
        val = manifest.get(field)
        if not isinstance(val, str) or _is_placeholder(val):
            errors.append(f"{field} must be a non-placeholder string, got {val!r}")
    pairs = manifest.get("train_snapshot_pairs")
    if not isinstance(pairs, list) or not pairs:
        errors.append("train_snapshot_pairs must be a non-empty list")
    else:
        for i, pair in enumerate(pairs):
            if not isinstance(pair, str) or not SNAPSHOT_PAIR_RE.match(pair):
                errors.append(
                    f"train_snapshot_pairs[{i}] not a v###-v### pair: {pair!r}"
                )
    eval_pair = manifest.get("eval_snapshot_pair")
    if not isinstance(eval_pair, str) or not SNAPSHOT_PAIR_RE.match(eval_pair):
        errors.append(f"eval_snapshot_pair not a v###-v### pair: {eval_pair!r}")
    fmt = manifest.get("format")
    if fmt is not None and fmt != "parquet":
        errors.append(f"format must be 'parquet', got {fmt!r}")
    for field in ("n_train_rows", "n_eval_rows"):
        val = manifest.get(field)
        if val is not None and (not isinstance(val, int) or isinstance(val, bool) or val < 0):
            errors.append(f"{field} must be a non-negative int if present, got {val!r}")
    return errors


def validate_manifest(manifest: dict[str, Any]) -> list[str]:
    """Run every check; return the full ordered error list."""
    errors: list[str] = []
    errors.extend(_check_required_keys(manifest))
    errors.extend(_check_unknown_keys(manifest))
    errors.extend(_check_schema_version(manifest))
    errors.extend(_check_types(manifest))
    errors.extend(_check_digests(manifest))
    return errors


def validate_path(path: Path) -> list[str]:
    """Load JSON from ``path`` and return its validation errors."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return [f"cannot read {path}: {exc}"]
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        return [f"invalid JSON at {path}: {exc}"]
    if not isinstance(data, dict):
        return [f"{path}: top-level must be an object, got {type(data).__name__}"]
    return validate_manifest(data)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate one or more dataset manifest.json files."
    )
    parser.add_argument(
        "--manifest",
        action="append",
        type=Path,
        default=[],
        help="Path to a manifest.json (repeatable).",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Additional positional manifest paths.",
    )
    args = parser.parse_args(argv)

    targets: list[Path] = [*args.manifest, *args.paths]
    if not targets:
        print("error: provide at least one --manifest or positional path", file=sys.stderr)
        return 2

    total_errors = 0
    for path in targets:
        errors = validate_path(path)
        if errors:
            total_errors += len(errors)
            print(f"{path}: FAIL ({len(errors)} error(s))", file=sys.stderr)
            for err in errors:
                print(f"  - {err}", file=sys.stderr)
        else:
            print(f"{path}: OK")
    if total_errors:
        print(
            f"\n{total_errors} validation error(s) across {len(targets)} file(s).",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
