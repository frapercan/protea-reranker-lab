"""Tests for the F-DATA-PACK.1 manifest validator.

The validator gates Zenodo / Hugging Face uploads on every frozen
``bench-v1-K{k}-v{val_band}-lineage-{plm_short}`` dataset. Tests pin:

1. All 10 real fixture manifests (captured from MinIO ``protea/datasets/``
   on 2026-05-25) pass without error.
2. Unknown top-level keys are rejected (strict mode, unlike
   ``ManifestV1`` which silently drops extras).
3. Placeholder digests (literal ``"placeholder"``, ``"TBD"``,
   all-zeros, repeated single char) are rejected.
4. Required keys are required.
5. Schema version is pinned to ``"v2"``.
6. Type errors on ``k``, ``train_snapshot_pairs``, ``eval_snapshot_pair``
   are caught.
7. CLI returns exit-code 0 on a clean manifest, 1 on a dirty one,
   and 2 on invocation error.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = REPO_ROOT / "scripts" / "validate_manifest.py"
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "manifests"

# Ensure the validator module is importable.
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import validate_manifest as vm  # noqa: E402


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(VALIDATOR), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def _load_fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def test_all_real_fixtures_pass() -> None:
    """Every captured production manifest must validate clean."""
    paths = sorted(FIXTURES.glob("*.json"))
    assert paths, "no fixture manifests captured"
    for path in paths:
        manifest = json.loads(path.read_text())
        errors = vm.validate_manifest(manifest)
        assert not errors, f"{path.name}: {errors}"


def test_all_real_fixtures_pass_via_cli() -> None:
    """CLI sweep: pass every fixture in a single invocation, expect rc=0."""
    fixture_args: list[str] = []
    for p in sorted(FIXTURES.glob("*.json")):
        fixture_args.extend(["--manifest", str(p)])
    result = _run(fixture_args)
    assert result.returncode == 0, result.stderr


def test_rejects_unknown_top_level_key() -> None:
    manifest = _load_fixture("bench-v1-K5-v226-lineage-esm2_150m.json")
    manifest["surprise_field"] = "drift"
    errors = vm.validate_manifest(manifest)
    assert any("surprise_field" in e for e in errors), errors


def test_rejects_placeholder_schema_sha() -> None:
    manifest = _load_fixture("bench-v1-K5-v226-lineage-esm2_150m.json")
    manifest["schema_sha"] = "placeholder"
    errors = vm.validate_manifest(manifest)
    assert any("schema_sha" in e and "placeholder" in e for e in errors), errors


def test_rejects_tbd_in_producer_git_sha() -> None:
    manifest = _load_fixture("bench-v1-K5-v226-lineage-esm2_150m.json")
    manifest["producer_git_sha"] = "TBD"
    errors = vm.validate_manifest(manifest)
    assert any("producer_git_sha" in e for e in errors), errors


def test_rejects_all_zero_schema_sha() -> None:
    manifest = _load_fixture("bench-v1-K5-v226-lineage-esm2_150m.json")
    manifest["schema_sha"] = "000000000000"
    errors = vm.validate_manifest(manifest)
    assert any("schema_sha" in e for e in errors), errors


def test_rejects_malformed_schema_sha_length() -> None:
    manifest = _load_fixture("bench-v1-K5-v226-lineage-esm2_150m.json")
    manifest["schema_sha"] = "deadbeef"  # 8 hex, not 12
    errors = vm.validate_manifest(manifest)
    assert any("schema_sha" in e for e in errors), errors


def test_rejects_malformed_producer_git_sha() -> None:
    manifest = _load_fixture("bench-v1-K5-v226-lineage-esm2_150m.json")
    manifest["producer_git_sha"] = "not-a-sha"
    errors = vm.validate_manifest(manifest)
    assert any("producer_git_sha" in e for e in errors), errors


def test_required_keys_enforced() -> None:
    manifest = _load_fixture("bench-v1-K5-v226-lineage-esm2_150m.json")
    del manifest["embedding_config_id"]
    errors = vm.validate_manifest(manifest)
    assert any("embedding_config_id" in e and "missing" in e for e in errors), errors


def test_schema_version_pinned_to_v2() -> None:
    manifest = _load_fixture("bench-v1-K5-v226-lineage-esm2_150m.json")
    manifest["schema_version"] = "v3"
    errors = vm.validate_manifest(manifest)
    assert any("schema_version" in e for e in errors), errors


def test_rejects_non_positive_k() -> None:
    manifest = _load_fixture("bench-v1-K5-v226-lineage-esm2_150m.json")
    manifest["k"] = 0
    errors = vm.validate_manifest(manifest)
    assert any("k must be a positive int" in e for e in errors), errors


def test_rejects_empty_train_snapshot_pairs() -> None:
    manifest = _load_fixture("bench-v1-K5-v226-lineage-esm2_150m.json")
    manifest["train_snapshot_pairs"] = []
    errors = vm.validate_manifest(manifest)
    assert any("train_snapshot_pairs" in e for e in errors), errors


def test_rejects_malformed_snapshot_pair() -> None:
    manifest = _load_fixture("bench-v1-K5-v226-lineage-esm2_150m.json")
    manifest["eval_snapshot_pair"] = "v226_to_v230"  # underscore not dash
    errors = vm.validate_manifest(manifest)
    assert any("eval_snapshot_pair" in e for e in errors), errors


def test_rejects_non_parquet_format() -> None:
    manifest = _load_fixture("bench-v1-K5-v226-lineage-esm2_150m.json")
    manifest["format"] = "csv"
    errors = vm.validate_manifest(manifest)
    assert any("format" in e for e in errors), errors


def test_rejects_negative_row_count() -> None:
    manifest = _load_fixture("bench-v1-K5-v226-lineage-esm2_150m.json")
    manifest["n_train_rows"] = -1
    errors = vm.validate_manifest(manifest)
    assert any("n_train_rows" in e for e in errors), errors


def test_cli_returns_zero_on_clean_manifest(tmp_path: Path) -> None:
    src = FIXTURES / "bench-v1-K5-v226-lineage-esm2_150m.json"
    target = tmp_path / "manifest.json"
    target.write_bytes(src.read_bytes())
    result = _run(["--manifest", str(target)])
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_cli_returns_one_on_dirty_manifest(tmp_path: Path) -> None:
    manifest = _load_fixture("bench-v1-K5-v226-lineage-esm2_150m.json")
    manifest["schema_sha"] = "placeholder"
    target = tmp_path / "manifest.json"
    target.write_text(json.dumps(manifest))
    result = _run(["--manifest", str(target)])
    assert result.returncode == 1
    assert "placeholder" in result.stderr


def test_cli_returns_two_on_no_args() -> None:
    result = _run([])
    assert result.returncode == 2


def test_cli_handles_invalid_json(tmp_path: Path) -> None:
    target = tmp_path / "manifest.json"
    target.write_text("{not valid json")
    result = _run(["--manifest", str(target)])
    assert result.returncode == 1
    assert "invalid JSON" in result.stderr


def test_cli_handles_missing_file(tmp_path: Path) -> None:
    target = tmp_path / "nope.json"
    result = _run(["--manifest", str(target)])
    assert result.returncode == 1
    assert "cannot read" in result.stderr or "FAIL" in result.stderr


def test_cli_accepts_positional_paths(tmp_path: Path) -> None:
    src = FIXTURES / "bench-v1-K5-v226-lineage-esm2_150m.json"
    target = tmp_path / "manifest.json"
    target.write_bytes(src.read_bytes())
    result = _run([str(target)])
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("token", ["placeholder", "TBD", "tbd", "TODO", "n/a", "fixme"])
def test_placeholder_tokens_are_caught(token: str) -> None:
    manifest = _load_fixture("bench-v1-K5-v226-lineage-esm2_150m.json")
    manifest["schema_sha"] = token
    errors = vm.validate_manifest(manifest)
    assert any("placeholder" in e for e in errors), (token, errors)
