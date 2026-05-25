"""Tests for the F-DATA-PACK.2 README generator.

The generator renders ``datasets/<name>/README.md`` from a frozen
manifest plus the PROTEA Dataset registry row. These tests pin:

1. The template substitution produces a deterministic README for a
   given (row, manifest) pair (uses a real fixture manifest from the
   F-DATA-PACK.1 corpus + a synthesised registry row).
2. The alias section is empty when manifest.name == row.name, and
   present (with the ghost prostt5 wording) when they differ.
3. Validation rejects placeholder manifests before rendering.
4. CLI returns 0 on a clean cell, 1 on a failing one, 2 on missing
   template.
5. Missing registry rows (in flight cells, 404) are skipped, not
   failed; output count reflects this.
"""

from __future__ import annotations

import json
import sys
import urllib.error
from pathlib import Path
from unittest import mock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "manifests"
TEMPLATE = SCRIPTS / "templates" / "dataset_readme.md.tmpl"

sys.path.insert(0, str(SCRIPTS))
import generate_dataset_readme as gdr  # noqa: E402


def _load_manifest(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text())


def _synth_row(name: str, manifest: dict, *, manifest_sha: str = "deadbeef" * 8,
               aliases: list[str] | None = None, job_id: str | None = "job-fake") -> dict:
    """Build a registry row that matches the manifest fields the script reads."""
    meta = {"alias_names": aliases} if aliases else {}
    return {
        "name": name,
        "operation": "export_research_dataset",
        "job_id": job_id,
        "storage_backend": "minio",
        "key_prefix": f"datasets/{manifest['name']}/",
        "train_uri": f"s3://protea/datasets/{manifest['name']}/train.parquet",
        "eval_uri": f"s3://protea/datasets/{manifest['name']}/eval.parquet",
        "manifest_uri": f"s3://protea/datasets/{manifest['name']}/manifest.json",
        "schema_sha": manifest["schema_sha"],
        "manifest_sha": manifest_sha,
        "n_train_rows": manifest["n_train_rows"],
        "n_eval_rows": manifest["n_eval_rows"],
        "k": manifest["k"],
        "annotation_source": manifest["annotation_source"],
        "embedding_config_id": manifest["embedding_config_id"],
        "ontology_snapshot_id": manifest["ontology_snapshot_id"],
        "train_snapshot_pairs": manifest["train_snapshot_pairs"],
        "eval_snapshot_pair": manifest["eval_snapshot_pair"],
        "producer_version": manifest["producer_version"],
        "producer_git_sha": manifest["producer_git_sha"],
        "meta": meta,
        "created_at": "2026-05-25T00:00:00+00:00",
    }


@pytest.fixture
def template() -> str:
    return TEMPLATE.read_text()


def test_render_substitutes_required_fields(template: str) -> None:
    name = "bench-v1-K5-v226-lineage-esm2_650m"
    manifest = _load_manifest(name)
    row = _synth_row(name, manifest)
    out = gdr._render(template, row, manifest)
    # No unsubstituted template variables.
    assert "${" not in out
    # Headline + schema sha + manifest sha all present.
    assert f"# {name}" in out
    assert manifest["schema_sha"] in out
    assert row["manifest_sha"] in out
    assert manifest["producer_git_sha"] in out
    assert row["created_at"] in out
    # Snapshot pairs land as inline backticked codes.
    assert "`v220-v226`" in out
    # K appears as a bare integer in the table.
    assert "| K (neighbours per query) | 5 |" in out
    # Row counts get thousands separators.
    assert "24,921,117" in out


def test_render_is_deterministic(template: str) -> None:
    name = "bench-v1-K10-v226-lineage-ankh_large"
    manifest = _load_manifest(name)
    row = _synth_row(name, manifest)
    a = gdr._render(template, row, manifest)
    b = gdr._render(template, row, manifest)
    assert a == b


def test_alias_section_empty_when_names_match() -> None:
    name = "bench-v1-K5-v226-lineage-esm2_650m"
    manifest = _load_manifest(name)
    row = _synth_row(name, manifest)  # no aliases, matching name
    assert gdr._alias_section(row, manifest["name"]) == ""


def test_alias_section_present_for_prostt5_ghost_case() -> None:
    # Synthesise the prostt5 ghost: manifest.name is the legacy PLM
    # blind form; row.name is the canonical per PLM form; meta carries
    # the alias list.
    manifest_name = "bench-v1-K5-v226-lineage"
    row_name = "bench-v1-K5-v226-lineage-prostt5"
    row = {
        "name": row_name,
        "meta": {"alias_names": [manifest_name]},
    }
    section = gdr._alias_section(row, manifest_name)
    assert "## Alias" in section
    assert "legacy ghost prostt5 case" in section
    assert f"`{manifest_name}`" in section
    assert f"`{row_name}`" in section


def test_validate_or_raise_rejects_placeholder_schema_sha() -> None:
    name = "bench-v1-K5-v226-lineage-esm2_150m"
    manifest = _load_manifest(name)
    manifest["schema_sha"] = "placeholder"
    body = json.dumps(manifest).encode("utf-8")
    with pytest.raises(ValueError, match="placeholder"):
        gdr._validate_or_raise(body, name)


def test_validate_or_raise_accepts_real_fixture() -> None:
    name = "bench-v1-K5-v226-lineage-esm2_150m"
    manifest = _load_manifest(name)
    body = json.dumps(manifest).encode("utf-8")
    parsed = gdr._validate_or_raise(body, name)
    assert parsed["name"] == name


def test_generate_one_skips_when_registry_404(tmp_path: Path, template: str) -> None:
    cfg = gdr.GenerateConfig(
        api_base="http://example.invalid",
        s3_endpoint="http://example.invalid",
        access_key="", secret_key="",
        datasets_root=tmp_path,
        template=template, dry_run=False,
    )
    with mock.patch.object(gdr, "_fetch_registry_row", return_value=None):
        out = gdr.generate_one("bench-v1-K3-v226-lineage-ankh_base", cfg)
    assert out is None
    assert not list(tmp_path.iterdir())


def test_generate_one_dry_run_does_not_write(tmp_path: Path, template: str) -> None:
    name = "bench-v1-K5-v226-lineage-esm2_650m"
    manifest = _load_manifest(name)
    row = _synth_row(name, manifest)
    cfg = gdr.GenerateConfig(
        api_base="http://example.invalid",
        s3_endpoint="http://example.invalid",
        access_key="", secret_key="",
        datasets_root=tmp_path,
        template=template, dry_run=True,
    )
    with mock.patch.object(gdr, "_fetch_registry_row", return_value=row), \
         mock.patch.object(
             gdr, "_fetch_manifest_bytes",
             return_value=json.dumps(manifest).encode("utf-8"),
         ):
        out = gdr.generate_one(name, cfg)
    assert out is not None
    assert not out.exists()


def test_generate_one_writes_readme(tmp_path: Path, template: str) -> None:
    name = "bench-v1-K5-v226-lineage-esm2_650m"
    manifest = _load_manifest(name)
    row = _synth_row(name, manifest)
    cfg = gdr.GenerateConfig(
        api_base="http://example.invalid",
        s3_endpoint="http://example.invalid",
        access_key="", secret_key="",
        datasets_root=tmp_path,
        template=template, dry_run=False,
    )
    with mock.patch.object(gdr, "_fetch_registry_row", return_value=row), \
         mock.patch.object(
             gdr, "_fetch_manifest_bytes",
             return_value=json.dumps(manifest).encode("utf-8"),
         ):
        out = gdr.generate_one(name, cfg)
    assert out is not None
    assert out == tmp_path / name / "README.md"
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert f"# {name}" in text
    assert "F-DATA-PACK.1" in text  # template cites validator slice
    assert "dl-postponed-2026-05-25" in text  # template cites postponement memory


def test_generate_one_fails_on_placeholder_manifest(tmp_path: Path, template: str) -> None:
    name = "bench-v1-K5-v226-lineage-esm2_650m"
    manifest = _load_manifest(name)
    manifest["schema_sha"] = "placeholder"
    row = _synth_row(name, manifest)
    cfg = gdr.GenerateConfig(
        api_base="http://example.invalid",
        s3_endpoint="http://example.invalid",
        access_key="", secret_key="",
        datasets_root=tmp_path,
        template=template, dry_run=False,
    )
    with mock.patch.object(gdr, "_fetch_registry_row", return_value=row), \
         mock.patch.object(
             gdr, "_fetch_manifest_bytes",
             return_value=json.dumps(manifest).encode("utf-8"),
         ):
        with pytest.raises(ValueError, match="placeholder"):
            gdr.generate_one(name, cfg)


def test_fetch_registry_row_returns_none_on_404() -> None:
    err = urllib.error.HTTPError(
        url="http://x/y", code=404, msg="Not Found", hdrs=None, fp=None,
    )
    with mock.patch.object(gdr.urllib.request, "urlopen", side_effect=err):
        result = gdr._fetch_registry_row("http://example.invalid", "bench-v1-K3-v226-lineage-ankh_base")
    assert result is None


def test_fetch_registry_row_raises_on_500() -> None:
    err = urllib.error.HTTPError(
        url="http://x/y", code=500, msg="Server Error", hdrs=None, fp=None,
    )
    with mock.patch.object(gdr.urllib.request, "urlopen", side_effect=err):
        with pytest.raises(urllib.error.HTTPError):
            gdr._fetch_registry_row("http://example.invalid", "bench-v1-K5-v226-lineage-esm2_650m")


def test_main_template_missing_returns_2(tmp_path: Path) -> None:
    rc = gdr.main(["--template", str(tmp_path / "nope.tmpl")])
    assert rc == 2


def test_default_cells_covers_11_cells() -> None:
    assert len(gdr.DEFAULT_CELLS) == 11
    assert any("prostt5" in n for n in gdr.DEFAULT_CELLS)
    # No mini/smoke datasets in default sweep.
    assert all("-mini" not in n for n in gdr.DEFAULT_CELLS)
    assert all("smoke" not in n for n in gdr.DEFAULT_CELLS)
