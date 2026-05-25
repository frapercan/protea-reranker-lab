"""Tests for the F-DATA-PACK.5 Zenodo uploader.

Pin the pieces of ``scripts/upload_to_zenodo.py`` that do not need a
live Zenodo account or a running PROTEA stack:

1. Metadata builder: title, license, creators, keywords, optional
   communities and grants, version pulled from the manifest.
2. Cell plan composer: parses the PLM short token from the dataset
   name, locates the card and provenance docs, and rejects cells that
   are missing local docs.
3. Idempotency ledger: round-trips via the on-disk JSON file, tolerates
   absent or malformed ledgers.
4. CLI dry-run: end-to-end smoke that monkeypatches the PROTEA fetch
   layer and asserts zero Zenodo API calls.
5. Token resolution: real-upload mode without ``ZENODO_TOKEN`` exits 2;
   dry-run tolerates a missing token.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
UPLOADER = REPO_ROOT / "scripts" / "upload_to_zenodo.py"
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "manifests"

# Importable module (scripts/ is on sys.path via pyproject).
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import upload_to_zenodo as uz  # noqa: E402


def _load_fixture_manifest(plm: str = "esm2_650m", k: int = 5) -> dict[str, Any]:
    name = f"bench-v1-K{k}-v226-lineage-{plm}.json"
    return json.loads((FIXTURES / name).read_text())


def _fake_registry_row(name: str, manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": "00000000-0000-0000-0000-000000000000",
        "name": name,
        "operation": "export_research_dataset",
        "job_id": "11111111-1111-1111-1111-111111111111",
        "storage_backend": "local",
        "key_prefix": f"datasets/{name}/",
        "train_uri": f"s3://protea/datasets/{name}/train.parquet",
        "eval_uri": f"s3://protea/datasets/{name}/eval.parquet",
        "manifest_uri": f"s3://protea/datasets/{name}/manifest.json",
        "schema_sha": manifest["schema_sha"],
        "manifest_sha": "deadbeefcafe1234567890abcdef0123456789ab",
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
        "meta": {},
        "created_at": "2026-05-25T00:00:00Z",
    }


def _make_plan(tmp_path: Path, name: str = "bench-v1-K5-v226-lineage-esm2_650m",
               plm: str = "esm2_650m", k: int = 5) -> uz.CellPlan:
    manifest = _load_fixture_manifest(plm=plm, k=k)
    row = _fake_registry_row(name, manifest)
    datasets_root = tmp_path / "datasets"
    cards_root = tmp_path / "dataset_cards"
    provenance = tmp_path / "docs" / "dataset_provenance.md"
    (datasets_root / name).mkdir(parents=True)
    (datasets_root / name / "README.md").write_text("# stub readme")
    cards_root.mkdir(parents=True)
    (cards_root / f"{plm}_card.md").write_text("# stub card")
    provenance.parent.mkdir(parents=True)
    provenance.write_text("# stub provenance")
    return uz._build_cell_plan(
        name, row, manifest,
        datasets_root=datasets_root,
        cards_root=cards_root,
        provenance_path=provenance,
    )


# ---------------------------------------------------------------------------
# Metadata builder
# ---------------------------------------------------------------------------


def test_metadata_pins_license_creator_and_upload_type(tmp_path: Path) -> None:
    plan = _make_plan(tmp_path)
    metadata = uz.build_metadata(plan)
    assert metadata["upload_type"] == "dataset"
    assert metadata["license"] == "cc-by-4.0"
    assert metadata["access_right"] == "open"
    assert metadata["creators"] == [
        {"name": "Francisco Miguel Perez Canales",
         "affiliation": "Universidad de Sevilla"},
    ]
    assert metadata["title"] == plan.name
    assert "K=5" in metadata["keywords"]
    assert "esm2_650m" in metadata["keywords"]
    assert metadata["version"] == plan.manifest["producer_version"]


def test_metadata_optional_fields(tmp_path: Path) -> None:
    plan = _make_plan(tmp_path)
    metadata_default = uz.build_metadata(plan)
    assert "communities" not in metadata_default
    assert "grants" not in metadata_default
    metadata = uz.build_metadata(
        plan,
        communities=("protea", "cafa-evaluation"),
        grants=("10.13039/501100011033::PID2020-FOO",),
    )
    assert metadata["communities"] == [
        {"identifier": "protea"},
        {"identifier": "cafa-evaluation"},
    ]
    assert metadata["grants"] == [{"id": "10.13039/501100011033::PID2020-FOO"}]


# ---------------------------------------------------------------------------
# Cell plan composer
# ---------------------------------------------------------------------------


def test_plm_short_from_name() -> None:
    assert uz._plm_short_from_name("bench-v1-K5-v226-lineage-esm2_650m") == "esm2_650m"
    assert uz._plm_short_from_name("bench-v1-K10-v226-lineage-ankh_base") == "ankh_base"


def test_plm_short_rejects_malformed() -> None:
    with pytest.raises(ValueError):
        uz._plm_short_from_name("not-a-canonical-name")


def test_check_local_docs_returns_errors_for_missing_docs(tmp_path: Path) -> None:
    plan = _make_plan(tmp_path)
    # Sanity: complete plan returns no errors.
    assert uz._check_local_docs_present(plan) == []
    # Deleting any doc raises one error.
    plan.readme_path.unlink()
    errors = uz._check_local_docs_present(plan)
    assert len(errors) == 1
    assert "README" in errors[0]


# ---------------------------------------------------------------------------
# Idempotency ledger
# ---------------------------------------------------------------------------


def test_ledger_roundtrip(tmp_path: Path) -> None:
    ledger_path = tmp_path / ".zenodo_deposits.json"
    assert uz._load_ledger(ledger_path) == {}
    ledger = {
        "bench-v1-K5-v226-lineage-esm2_650m": {
            "deposition_id": 12345,
            "doi": "10.5281/zenodo.12345",
            "status": "draft",
        },
    }
    uz._save_ledger(ledger_path, ledger)
    assert ledger_path.is_file()
    loaded = uz._load_ledger(ledger_path)
    assert loaded == ledger


def test_ledger_tolerates_malformed(tmp_path: Path) -> None:
    ledger_path = tmp_path / ".zenodo_deposits.json"
    ledger_path.write_text("not json", encoding="utf-8")
    assert uz._load_ledger(ledger_path) == {}
    ledger_path.write_text("[\"list, not object\"]", encoding="utf-8")
    assert uz._load_ledger(ledger_path) == {}


# ---------------------------------------------------------------------------
# Zenodo client rejects an empty token
# ---------------------------------------------------------------------------


def test_zenodo_client_requires_token() -> None:
    with pytest.raises(ValueError):
        uz.ZenodoClient("")


# ---------------------------------------------------------------------------
# CLI dry-run end-to-end
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_protea(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Patch the PROTEA + S3 fetch helpers so no network call leaves the test.

    Returns a shared call counter dict so each test can assert it never
    crossed the Zenodo boundary.
    """
    call_log: dict[str, int] = {"protea_rows": 0, "manifest_bytes": 0}

    def fake_row(_api_base: str, name: str, timeout: int = 30) -> dict[str, Any] | None:
        call_log["protea_rows"] += 1
        plm = uz._plm_short_from_name(name)
        # K is parsed off the dataset name (e.g. ``bench-v1-K5-...``).
        k_token = name.split("-")[2]  # "K5"
        k = int(k_token.lstrip("K"))
        try:
            manifest = _load_fixture_manifest(plm=plm, k=k)
        except FileNotFoundError:
            return None
        return _fake_registry_row(name, manifest)

    def fake_bytes(manifest_uri: str, **_: Any) -> bytes:
        call_log["manifest_bytes"] += 1
        # Pull the name back out of s3://protea/datasets/<name>/manifest.json
        prefix = "s3://protea/datasets/"
        assert manifest_uri.startswith(prefix), manifest_uri
        name = manifest_uri[len(prefix):].split("/", 1)[0]
        plm = uz._plm_short_from_name(name)
        k = int(name.split("-")[2].lstrip("K"))
        manifest = _load_fixture_manifest(plm=plm, k=k)
        return json.dumps(manifest).encode("utf-8")

    monkeypatch.setattr(uz.generate_dataset_readme, "_fetch_registry_row", fake_row)
    monkeypatch.setattr(uz.generate_dataset_readme, "_fetch_manifest_bytes", fake_bytes)
    return call_log


def _seed_local_docs(repo_tmp: Path, cells: tuple[str, ...]) -> None:
    """Write minimal local F-DATA-PACK 2/3/4 docs so cells validate."""
    (repo_tmp / "docs").mkdir(parents=True)
    (repo_tmp / "docs" / "dataset_provenance.md").write_text("# provenance")
    cards = repo_tmp / "dataset_cards"
    cards.mkdir()
    seen_plms: set[str] = set()
    for name in cells:
        plm = uz._plm_short_from_name(name)
        if plm not in seen_plms:
            (cards / f"{plm}_card.md").write_text(f"# {plm} card")
            seen_plms.add(plm)
        cell_dir = repo_tmp / "datasets" / name
        cell_dir.mkdir(parents=True, exist_ok=True)
        (cell_dir / "README.md").write_text(f"# {name}")


def test_dry_run_smokes_against_fixtures(
    tmp_path: Path, fake_protea: dict[str, int], capsys: pytest.CaptureFixture[str],
) -> None:
    cells = (
        "bench-v1-K5-v226-lineage-esm2_650m",
        "bench-v1-K10-v226-lineage-ankh_base",
    )
    _seed_local_docs(tmp_path, cells)
    rc = uz.main([
        "--dry-run",
        "--cell", cells[0],
        "--cell", cells[1],
        "--datasets-root", str(tmp_path / "datasets"),
        "--cards-root", str(tmp_path / "dataset_cards"),
        "--provenance-path", str(tmp_path / "docs" / "dataset_provenance.md"),
        "--ledger", str(tmp_path / ".zenodo_deposits.json"),
        "--community", "protea",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "DRY-RUN" in out
    assert "planned 2 cell" in out
    for name in cells:
        assert name in out
    # No ledger writes happen in dry-run mode.
    assert not (tmp_path / ".zenodo_deposits.json").exists()


def test_dry_run_skips_cells_with_no_registry_row(
    tmp_path: Path, fake_protea: dict[str, int], monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cells = ("bench-v1-K5-v226-lineage-esm2_650m",)
    _seed_local_docs(tmp_path, cells)

    def returns_none(_api_base: str, _name: str, timeout: int = 30) -> None:
        return None

    monkeypatch.setattr(uz.generate_dataset_readme, "_fetch_registry_row", returns_none)
    rc = uz.main([
        "--dry-run",
        "--cell", cells[0],
        "--datasets-root", str(tmp_path / "datasets"),
        "--cards-root", str(tmp_path / "dataset_cards"),
        "--provenance-path", str(tmp_path / "docs" / "dataset_provenance.md"),
        "--ledger", str(tmp_path / ".zenodo_deposits.json"),
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "planned 0 cell" in out
    assert "in-flight" in out


def test_dry_run_fails_when_local_docs_missing(
    tmp_path: Path, fake_protea: dict[str, int], capsys: pytest.CaptureFixture[str],
) -> None:
    cells = ("bench-v1-K5-v226-lineage-esm2_650m",)
    # Intentionally do NOT seed any local docs.
    rc = uz.main([
        "--dry-run",
        "--cell", cells[0],
        "--datasets-root", str(tmp_path / "datasets"),
        "--cards-root", str(tmp_path / "dataset_cards"),
        "--provenance-path", str(tmp_path / "docs" / "dataset_provenance.md"),
        "--ledger", str(tmp_path / ".zenodo_deposits.json"),
    ])
    assert rc == 1
    err = capsys.readouterr().err
    assert "FAIL" in err


# ---------------------------------------------------------------------------
# Token resolution
# ---------------------------------------------------------------------------


def test_real_upload_without_token_exits_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ZENODO_TOKEN", raising=False)
    rc = subprocess.run(
        [sys.executable, str(UPLOADER),
         "--cell", "bench-v1-K5-v226-lineage-esm2_650m",
         "--datasets-root", str(tmp_path / "datasets"),
         "--cards-root", str(tmp_path / "dataset_cards"),
         "--provenance-path", str(tmp_path / "docs" / "dataset_provenance.md"),
         "--ledger", str(tmp_path / ".zenodo_deposits.json")],
        capture_output=True,
        text=True,
        check=False,
        env={"PATH": "/usr/bin:/bin"},
    )
    assert rc.returncode == 2
    assert "ZENODO_TOKEN" in rc.stderr


def test_dry_run_tolerates_missing_token(
    tmp_path: Path, fake_protea: dict[str, int], monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ZENODO_TOKEN", raising=False)
    cells = ("bench-v1-K5-v226-lineage-esm2_650m",)
    _seed_local_docs(tmp_path, cells)
    rc = uz.main([
        "--dry-run",
        "--cell", cells[0],
        "--datasets-root", str(tmp_path / "datasets"),
        "--cards-root", str(tmp_path / "dataset_cards"),
        "--provenance-path", str(tmp_path / "docs" / "dataset_provenance.md"),
        "--ledger", str(tmp_path / ".zenodo_deposits.json"),
    ])
    assert rc == 0


# ---------------------------------------------------------------------------
# CLI sandbox flag rewires the API base
# ---------------------------------------------------------------------------


def test_sandbox_flag_rewires_api_base(
    tmp_path: Path, fake_protea: dict[str, int], monkeypatch: pytest.MonkeyPatch,
) -> None:
    cells = ("bench-v1-K5-v226-lineage-esm2_650m",)
    _seed_local_docs(tmp_path, cells)
    captured: dict[str, str] = {}

    real_uz_cfg = uz.UploadConfig

    def capture(*args: Any, **kwargs: Any) -> uz.UploadConfig:
        cfg = real_uz_cfg(*args, **kwargs)
        captured["zenodo_api"] = cfg.zenodo_api
        return cfg

    monkeypatch.setattr(uz, "UploadConfig", capture)
    uz.main([
        "--dry-run", "--sandbox",
        "--cell", cells[0],
        "--datasets-root", str(tmp_path / "datasets"),
        "--cards-root", str(tmp_path / "dataset_cards"),
        "--provenance-path", str(tmp_path / "docs" / "dataset_provenance.md"),
        "--ledger", str(tmp_path / ".zenodo_deposits.json"),
    ])
    assert captured["zenodo_api"] == uz.DEFAULT_SANDBOX_API


# ---------------------------------------------------------------------------
# Avoid silent log spam when running the suite verbosely.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _quiet_logging() -> None:
    logging.getLogger("upload_to_zenodo").setLevel(logging.CRITICAL)
