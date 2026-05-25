#!/usr/bin/env python3
"""Upload the bench-v1 v226-lineage dataset family to Zenodo.

Slice F-DATA-PACK.5 of the farm-platform loop. Closes the FAIR
publishing loop opened by F-DATA-PACK.1 (manifest validator),
F-DATA-PACK.2 (per-dataset README), F-DATA-PACK.3 (per-PLM dataset
card), and F-DATA-PACK.4 (provenance + FAIR document).

For each ``bench-v1-K{3,5,10}-v226-lineage-{plm_short}`` dataset
registered in PROTEA with a SUCCEEDED status (registry row present +
non-placeholder ``manifest_sha`` + manifest passes the F-DATA-PACK.1
schema validator), the script:

1. Fetches the registry row from ``GET /v1/datasets/<name>``.
2. Fetches ``manifest.json`` from the storage backend listed in the
   registry row and validates it.
3. Composes a Zenodo deposition with:
    a. ``train.parquet``, ``eval.parquet``, ``manifest.json`` resolved
       from the storage backend.
    b. ``README.md`` produced by F-DATA-PACK.2
       (``datasets/<name>/README.md``).
    c. The matching ``dataset_cards/<plm_short>_card.md`` produced by
       F-DATA-PACK.3.
    d. ``docs/dataset_provenance.md`` from F-DATA-PACK.4 (shared by
       every cell so consumers can cite one FAIR document for the
       whole family).
4. Sets deposition metadata: CC-BY-4.0 license, creator
   ``Francisco Miguel Perez Canales``, ``communities`` and ``grants``
   configurable on the CLI.
5. Records the resulting DOI in a local manifest
   (``datasets/.zenodo_deposits.json``) so re-runs are idempotent.

The script does NOT publish the deposition (it stays as a draft until
the user reviews and clicks publish in the Zenodo UI). It also does
NOT auto-edit ``dataset_cards/<plm>_card.md`` (DOI backfill is a
separate slice that runs once depositions are actually published).

Idempotency model:

* Two layers. First, the local ``datasets/.zenodo_deposits.json``
  records ``(dataset_name, deposition_id, doi, parent_doi, status)``
  per cell. On re-run, cells already present are skipped.
* Second, when the local manifest is missing or stale, the script
  queries the Zenodo deposition list ``GET /api/deposit/depositions``
  filtered by ``q=metadata.title:"<dataset_name>"`` and reuses any
  existing draft / published deposition. Only if both layers say
  "no deposition exists" does a new one get created.

Dry-run mode (``--dry-run``) makes zero Zenodo API calls. It iterates
the PROTEA registry, validates every cell, and prints what would be
uploaded. CI ships ``--dry-run`` only; real uploads require the
user-supplied token at runtime.

Exit codes:

    0 - clean (every selected cell either uploaded, skipped as
        idempotent, or dry-run-validated)
    1 - at least one cell failed validation, fetch, or upload
    2 - invocation error (e.g. missing token in non-dry-run mode)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import generate_dataset_readme  # local script (F-DATA-PACK.2)
import validate_manifest  # local script (F-DATA-PACK.1)

log = logging.getLogger("upload_to_zenodo")


# Canonical dataset family. Matches F-DATA-PACK.2 default cells; cells
# without a SUCCEEDED registry row are skipped automatically (in flight
# on protea.training and will be picked up by a later re-run).
DEFAULT_CELLS: tuple[str, ...] = (
    "bench-v1-K3-v226-lineage-esm2_650m",
    "bench-v1-K5-v226-lineage-esm2_150m",
    "bench-v1-K5-v226-lineage-esm2_650m",
    "bench-v1-K5-v226-lineage-esm2_3b",
    "bench-v1-K5-v226-lineage-ankh_large",
    "bench-v1-K5-v226-lineage-prostt5",
    "bench-v1-K10-v226-lineage-esm2_150m",
    "bench-v1-K10-v226-lineage-esm2_650m",
    "bench-v1-K10-v226-lineage-esm2_3b",
    "bench-v1-K10-v226-lineage-ankh_base",
    "bench-v1-K10-v226-lineage-ankh_large",
)

DEFAULT_API: str = "http://localhost:8000"
DEFAULT_S3_ENDPOINT: str = "http://localhost:9000"
DEFAULT_ZENODO_API: str = "https://zenodo.org/api"
DEFAULT_SANDBOX_API: str = "https://sandbox.zenodo.org/api"

REPO_ROOT: Path = Path(__file__).resolve().parents[1]
DATASETS_ROOT: Path = REPO_ROOT / "datasets"
DATASET_CARDS_ROOT: Path = REPO_ROOT / "dataset_cards"
PROVENANCE_PATH: Path = REPO_ROOT / "docs" / "dataset_provenance.md"
DEPOSIT_LEDGER_PATH: Path = DATASETS_ROOT / ".zenodo_deposits.json"

CREATOR_NAME: str = "Francisco Miguel Perez Canales"
LICENSE_ID: str = "cc-by-4.0"
UPLOAD_TYPE: str = "dataset"


# ---------------------------------------------------------------------------
# Cell selection + manifest validation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CellPlan:
    """Bundle of every artefact and metadatum needed for one deposition.

    Built once per cell during the planning phase. The upload phase
    consumes it without touching the network for anything other than
    Zenodo itself.
    """

    name: str
    plm_short: str
    k: int
    manifest: dict[str, Any]
    registry_row: dict[str, Any]
    readme_path: Path
    card_path: Path
    provenance_path: Path
    artefact_uris: dict[str, str] = field(default_factory=dict)


def _plm_short_from_name(name: str) -> str:
    """Return the PLM short token from a canonical dataset name.

    ``bench-v1-K5-v226-lineage-esm2_650m`` -> ``esm2_650m``.
    """
    parts = name.rsplit("-lineage-", 1)
    if len(parts) != 2 or not parts[1]:
        raise ValueError(f"cannot parse PLM short from dataset name: {name!r}")
    return parts[1]


def _resolve_card_path(plm_short: str, cards_root: Path) -> Path:
    """Return the dataset card path for one PLM. ProstT5 has a special name."""
    return cards_root / f"{plm_short}_card.md"


def _load_manifest_via_protea(name: str, *, api_base: str, s3_endpoint: str,
                              access_key: str, secret_key: str
                              ) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Fetch the registry row + manifest bytes; validate; return (row, manifest).

    Reuses ``generate_dataset_readme`` helpers so behaviour is identical
    to the F-DATA-PACK.2 README generator. Returns ``None`` if the row
    is missing (in flight cell) or the manifest_uri is empty.
    """
    row = generate_dataset_readme._fetch_registry_row(api_base, name)
    if row is None:
        log.warning("%s: no registry row (in flight or missing) - skipping", name)
        return None
    manifest_uri = row.get("manifest_uri")
    if not manifest_uri:
        log.warning("%s: registry row has no manifest_uri - skipping", name)
        return None
    manifest_sha = row.get("manifest_sha")
    if not manifest_sha or validate_manifest._is_placeholder(manifest_sha):
        log.warning("%s: manifest_sha is placeholder %r - skipping", name, manifest_sha)
        return None
    manifest_bytes = generate_dataset_readme._fetch_manifest_bytes(
        manifest_uri, s3_endpoint=s3_endpoint,
        access_key=access_key, secret_key=secret_key,
    )
    manifest = generate_dataset_readme._validate_or_raise(manifest_bytes, name)
    return row, manifest


def _build_cell_plan(name: str, row: dict[str, Any], manifest: dict[str, Any],
                     *, datasets_root: Path, cards_root: Path,
                     provenance_path: Path) -> CellPlan:
    """Compose a ``CellPlan`` from validated registry + manifest + local docs."""
    plm_short = _plm_short_from_name(name)
    card_path = _resolve_card_path(plm_short, cards_root)
    readme_path = datasets_root / name / "README.md"
    artefact_uris = {
        "train.parquet": row.get("train_uri") or "",
        "eval.parquet": row.get("eval_uri") or "",
        "manifest.json": row.get("manifest_uri") or "",
    }
    return CellPlan(
        name=name,
        plm_short=plm_short,
        k=int(row.get("k") or manifest.get("k") or 0),
        manifest=manifest,
        registry_row=row,
        readme_path=readme_path,
        card_path=card_path,
        provenance_path=provenance_path,
        artefact_uris=artefact_uris,
    )


def _check_local_docs_present(plan: CellPlan) -> list[str]:
    """Verify every local doc the deposition needs is on disk."""
    errors: list[str] = []
    for path, label in (
        (plan.readme_path, "F-DATA-PACK.2 README"),
        (plan.card_path, "F-DATA-PACK.3 dataset card"),
        (plan.provenance_path, "F-DATA-PACK.4 provenance"),
    ):
        if not path.is_file():
            errors.append(f"missing {label}: {path}")
    return errors


# ---------------------------------------------------------------------------
# Idempotency ledger
# ---------------------------------------------------------------------------


def _load_ledger(path: Path) -> dict[str, dict[str, Any]]:
    """Load the local deposit ledger; return empty dict if absent."""
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("ledger unreadable (%s), starting fresh: %s", path, exc)
        return {}
    if not isinstance(data, dict):
        log.warning("ledger %s is not a JSON object, starting fresh", path)
        return {}
    return {k: v for k, v in data.items() if isinstance(v, dict)}


def _save_ledger(path: Path, ledger: dict[str, dict[str, Any]]) -> None:
    """Atomically write the ledger to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(ledger, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Zenodo metadata + API client
# ---------------------------------------------------------------------------


def _render_description(plan: CellPlan) -> str:
    """Render the HTML description block. Extracted to keep build_metadata small."""
    train_pairs = plan.manifest.get("train_snapshot_pairs") or []
    eval_pair = plan.manifest.get("eval_snapshot_pair") or ""
    schema_sha = plan.manifest.get("schema_sha") or ""
    producer_git_sha = plan.manifest.get("producer_git_sha") or ""
    n_train_rows = plan.registry_row.get("n_train_rows") or 0
    n_eval_rows = plan.registry_row.get("n_eval_rows") or 0
    first_pair = train_pairs[0] if train_pairs else ""
    last_pair = train_pairs[-1] if train_pairs else ""
    lines = [
        f"<p>Frozen LightGBM feature dataset cell of the <code>bench-v1</code> "
        f"family, K={plan.k}, PLM <code>{plan.plm_short}</code>, eval window "
        f"<code>{eval_pair}</code>.</p>",
        "<p>Produced by PROTEA's <code>export_research_dataset</code> "
        f"operation (version {plan.manifest.get('producer_version', '')}, "
        f"git sha <code>{producer_git_sha}</code>) on "
        f"{plan.registry_row.get('created_at', '')}.</p>",
        f"<p>Schema sha: <code>{schema_sha}</code>. "
        f"Manifest sha: <code>{plan.registry_row.get('manifest_sha', '')}</code>.</p>",
        f"<p>Rows: train={n_train_rows:,}, eval={n_eval_rows:,}.</p>",
        f"<p>Training window: {len(train_pairs)} lineage delta pairs from "
        f"<code>{first_pair}</code> to <code>{last_pair}</code>.</p>",
        "<p>Full provenance, schema, and FAIR compliance posture in the "
        "bundled <code>dataset_provenance.md</code> document.</p>",
    ]
    return "\n".join(lines)


def _keywords_for(plan: CellPlan) -> list[str]:
    return [
        "protein function prediction",
        "gene ontology",
        "reranker",
        "lightgbm",
        "knn",
        "protein language model",
        plan.plm_short,
        f"K={plan.k}",
        "bench-v1",
        "v226-lineage",
    ]


def build_metadata(plan: CellPlan, *, communities: tuple[str, ...] = (),
                   grants: tuple[str, ...] = (),
                   creator_name: str = CREATOR_NAME,
                   license_id: str = LICENSE_ID) -> dict[str, Any]:
    """Build the Zenodo deposition ``metadata`` block for one cell.

    Lifted out of the client so tests can pin it without any network.
    """
    metadata: dict[str, Any] = {
        "title": plan.name,
        "upload_type": UPLOAD_TYPE,
        "description": _render_description(plan),
        "creators": [
            {"name": creator_name, "affiliation": "Universidad de Sevilla"},
        ],
        "license": license_id,
        "access_right": "open",
        "keywords": _keywords_for(plan),
        "version": plan.manifest.get("producer_version") or "v1",
        "language": "eng",
    }
    if communities:
        metadata["communities"] = [{"identifier": c} for c in communities]
    if grants:
        metadata["grants"] = [{"id": g} for g in grants]
    return metadata


class ZenodoClient:
    """Tiny urllib-based client for the Zenodo REST API.

    Only the four endpoints F-DATA-PACK.5 needs are wired:
    ``GET /api/deposit/depositions`` (search), ``POST`` (create),
    ``PUT bucket/<filename>`` (upload), ``PUT /api/deposit/depositions/<id>``
    (metadata patch). ``publish`` is intentionally absent: depositions
    stay as drafts until the user reviews them in the UI.
    """

    def __init__(self, token: str, *, api_base: str = DEFAULT_ZENODO_API,
                 timeout: int = 300) -> None:
        if not token:
            raise ValueError("Zenodo token must be a non-empty string")
        self._token = token
        self._api = api_base.rstrip("/")
        self._timeout = timeout

    def _request(self, method: str, url: str, *, body: bytes | None = None,
                 content_type: str | None = None) -> dict[str, Any]:
        headers: dict[str, str] = {
            "Authorization": f"Bearer {self._token}",
            "Accept": "application/json",
        }
        if content_type is not None:
            headers["Content-Type"] = content_type
        req = urllib.request.Request(url, data=body, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code} {method} {url}: {detail}") from exc
        if not raw:
            return {}
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"non-JSON response from {url}: {exc}") from exc

    def find_deposition_by_title(self, title: str) -> dict[str, Any] | None:
        """Return the newest draft / published deposition whose title matches."""
        query = urllib.parse.urlencode({
            "q": f'metadata.title:"{title}"',
            "size": "10",
            "all_versions": "false",
        })
        url = f"{self._api}/deposit/depositions?{query}"
        data = self._request("GET", url)
        # Zenodo returns a JSON list for this endpoint when results exist.
        if isinstance(data, dict):
            # Empty result -> {} from our wrapper. Anything else with a
            # ``hits`` key is the search-API shape; the deposit list
            # endpoint normally returns a bare list.
            items = data.get("hits", {}).get("hits", []) if data else []
        elif isinstance(data, list):
            items = data
        else:
            items = []
        for item in items:
            md = (item or {}).get("metadata") or {}
            if md.get("title") == title:
                return item
        return None

    def create_deposition(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """POST /api/deposit/depositions with the given metadata."""
        url = f"{self._api}/deposit/depositions"
        body = json.dumps({"metadata": metadata}).encode("utf-8")
        return self._request("POST", url, body=body, content_type="application/json")

    def update_metadata(self, deposition_id: int, metadata: dict[str, Any]) -> dict[str, Any]:
        """PUT /api/deposit/depositions/<id> to replace metadata."""
        url = f"{self._api}/deposit/depositions/{deposition_id}"
        body = json.dumps({"metadata": metadata}).encode("utf-8")
        return self._request("PUT", url, body=body, content_type="application/json")

    def upload_file(self, bucket_url: str, filename: str, payload: bytes,
                    *, content_type: str = "application/octet-stream"
                    ) -> dict[str, Any]:
        """PUT raw bytes to the deposition's bucket under ``filename``."""
        encoded = urllib.parse.quote(filename, safe="")
        url = f"{bucket_url.rstrip('/')}/{encoded}"
        return self._request("PUT", url, body=payload, content_type=content_type)


# ---------------------------------------------------------------------------
# Upload orchestration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UploadConfig:
    """Static config for one orchestrated upload pass."""

    api_base: str
    s3_endpoint: str
    access_key: str
    secret_key: str
    zenodo_api: str
    zenodo_token: str
    datasets_root: Path
    cards_root: Path
    provenance_path: Path
    ledger_path: Path
    communities: tuple[str, ...]
    grants: tuple[str, ...]
    creator_name: str
    license_id: str
    dry_run: bool


def _fetch_remote_artefact_bytes(uri: str, cfg: UploadConfig) -> bytes:
    """Fetch a remote artefact (manifest, parquet) via the S3 or HTTP backend."""
    return generate_dataset_readme._fetch_manifest_bytes(
        uri, s3_endpoint=cfg.s3_endpoint,
        access_key=cfg.access_key, secret_key=cfg.secret_key,
    )


def _plan_one(name: str, cfg: UploadConfig) -> CellPlan | None:
    """Build a ``CellPlan`` for one cell or return ``None`` if skippable."""
    result = _load_manifest_via_protea(
        name, api_base=cfg.api_base, s3_endpoint=cfg.s3_endpoint,
        access_key=cfg.access_key, secret_key=cfg.secret_key,
    )
    if result is None:
        return None
    row, manifest = result
    plan = _build_cell_plan(
        name, row, manifest,
        datasets_root=cfg.datasets_root,
        cards_root=cfg.cards_root,
        provenance_path=cfg.provenance_path,
    )
    doc_errors = _check_local_docs_present(plan)
    if doc_errors:
        raise ValueError("; ".join(doc_errors))
    return plan


def _upload_one(plan: CellPlan, client: ZenodoClient, cfg: UploadConfig,
                ledger: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Push one cell's deposition to Zenodo. Returns a ledger entry."""
    metadata = build_metadata(
        plan,
        communities=cfg.communities,
        grants=cfg.grants,
        creator_name=cfg.creator_name,
        license_id=cfg.license_id,
    )
    existing = client.find_deposition_by_title(plan.name)
    if existing is not None:
        deposition = client.update_metadata(int(existing["id"]), metadata)
        action = "reused"
    else:
        deposition = client.create_deposition(metadata)
        action = "created"
    deposition_id = int(deposition["id"])
    bucket_url = deposition.get("links", {}).get("bucket", "")
    if not bucket_url:
        raise RuntimeError(
            f"{plan.name}: Zenodo deposition {deposition_id} has no bucket URL"
        )
    file_specs: list[tuple[str, bytes, str]] = []
    for filename, uri in plan.artefact_uris.items():
        if not uri:
            continue
        content = _fetch_remote_artefact_bytes(uri, cfg)
        mime = "application/json" if filename.endswith(".json") else "application/octet-stream"
        file_specs.append((filename, content, mime))
    file_specs.append((
        "README.md", plan.readme_path.read_bytes(), "text/markdown",
    ))
    file_specs.append((
        f"{plan.plm_short}_card.md", plan.card_path.read_bytes(), "text/markdown",
    ))
    file_specs.append((
        "dataset_provenance.md", plan.provenance_path.read_bytes(), "text/markdown",
    ))
    for filename, payload, mime in file_specs:
        client.upload_file(bucket_url, filename, payload, content_type=mime)
        log.info("%s: uploaded %s (%d bytes)", plan.name, filename, len(payload))
    doi = (deposition.get("metadata") or {}).get("doi") or deposition.get("doi") or ""
    prereserve = (deposition.get("metadata") or {}).get("prereserve_doi") or {}
    if not doi and isinstance(prereserve, dict):
        doi = prereserve.get("doi") or ""
    entry = {
        "deposition_id": deposition_id,
        "doi": doi,
        "concept_doi": deposition.get("conceptdoi") or "",
        "status": deposition.get("state", "unsubmitted"),
        "action": action,
        "links": {k: v for k, v in (deposition.get("links") or {}).items()
                  if isinstance(v, str)},
    }
    ledger[plan.name] = entry
    return entry


def _dry_run_plan(plan: CellPlan, cfg: UploadConfig) -> dict[str, Any]:
    """Return a summary dict describing what would be uploaded, no API call."""
    metadata = build_metadata(
        plan,
        communities=cfg.communities,
        grants=cfg.grants,
        creator_name=cfg.creator_name,
        license_id=cfg.license_id,
    )
    files = [
        f"train.parquet   -> {plan.artefact_uris['train.parquet']}",
        f"eval.parquet    -> {plan.artefact_uris['eval.parquet']}",
        f"manifest.json   -> {plan.artefact_uris['manifest.json']}",
        f"README.md       (local: {plan.readme_path})",
        f"{plan.plm_short}_card.md  (local: {plan.card_path})",
        f"dataset_provenance.md  (local: {plan.provenance_path})",
    ]
    return {
        "name": plan.name,
        "k": plan.k,
        "plm_short": plan.plm_short,
        "metadata": metadata,
        "files": files,
    }


# ---------------------------------------------------------------------------
# Orchestrator + CLI
# ---------------------------------------------------------------------------


def _print_dry_run_report(planned: list[dict[str, Any]],
                          skipped_in_flight: list[str],
                          failures: list[tuple[str, str]],
                          total: int) -> None:
    print(
        f"upload_to_zenodo DRY-RUN: planned {len(planned)} cell(s), "
        f"skipped {len(skipped_in_flight)} in-flight, "
        f"{len(failures)} failed (of {total})"
    )
    for entry in planned:
        print(f"\n=== {entry['name']} (K={entry['k']}, PLM={entry['plm_short']}) ===")
        print(f"  title:       {entry['metadata']['title']}")
        print(f"  license:     {entry['metadata']['license']}")
        print(f"  creator:     {entry['metadata']['creators'][0]['name']}")
        print(f"  version:     {entry['metadata']['version']}")
        print(f"  keywords:    {', '.join(entry['metadata']['keywords'])}")
        communities = entry["metadata"].get("communities") or []
        if communities:
            print(f"  communities: {[c['identifier'] for c in communities]}")
        grants = entry["metadata"].get("grants") or []
        if grants:
            print(f"  grants:      {[g['id'] for g in grants]}")
        print("  files:")
        for line in entry["files"]:
            print(f"    - {line}")


def _print_real_run_report(uploaded: list[dict[str, Any]],
                           skipped_idempotent: list[str],
                           skipped_in_flight: list[str],
                           failures: list[tuple[str, str]],
                           ledger: dict[str, dict[str, Any]],
                           total: int) -> None:
    print(
        f"upload_to_zenodo: uploaded {len(uploaded)}, "
        f"reused {len(skipped_idempotent)} idempotent, "
        f"skipped {len(skipped_in_flight)} in-flight, "
        f"{len(failures)} failed (of {total})"
    )
    for entry in uploaded:
        print(f"  uploaded: {entry['name']} -> doi={entry.get('doi') or '(unset)'}")
    for name in skipped_idempotent:
        entry = ledger.get(name) or {}
        print(f"  reused:   {name} -> doi={entry.get('doi') or '(unset)'}")
    if uploaded or skipped_idempotent:
        print(
            "\nNext step: review the draft depositions in the Zenodo UI, "
            "click publish, and record the DOIs in agent-farm memory. "
            "The dataset_cards/<plm>_card.md DOI backfill runs in a "
            "separate slice once publishing is confirmed."
        )


def run(cells: tuple[str, ...], cfg: UploadConfig) -> int:
    """Plan + (optionally) upload every cell. Return a CLI-style exit code."""
    ledger = _load_ledger(cfg.ledger_path)
    client: ZenodoClient | None = None
    if not cfg.dry_run:
        client = ZenodoClient(cfg.zenodo_token, api_base=cfg.zenodo_api)
    failures: list[tuple[str, str]] = []
    skipped_in_flight: list[str] = []
    skipped_idempotent: list[str] = []
    planned: list[dict[str, Any]] = []
    uploaded: list[dict[str, Any]] = []
    for name in cells:
        try:
            plan = _plan_one(name, cfg)
        except (ValueError, urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
            log.error("%s: planning FAIL - %s", name, exc)
            failures.append((name, f"plan: {exc}"))
            continue
        if plan is None:
            skipped_in_flight.append(name)
            continue
        if cfg.dry_run:
            planned.append(_dry_run_plan(plan, cfg))
            continue
        existing_entry = ledger.get(name)
        if existing_entry and existing_entry.get("doi") and existing_entry.get("deposition_id"):
            log.info("%s: ledger hit, skipping (doi=%s)", name, existing_entry.get("doi"))
            skipped_idempotent.append(name)
            continue
        try:
            assert client is not None
            entry = _upload_one(plan, client, cfg, ledger)
        except (RuntimeError, ValueError, urllib.error.URLError, OSError) as exc:
            log.error("%s: upload FAIL - %s", name, exc)
            failures.append((name, f"upload: {exc}"))
            continue
        uploaded.append({"name": name, **entry})
        _save_ledger(cfg.ledger_path, ledger)
    if cfg.dry_run:
        _print_dry_run_report(planned, skipped_in_flight, failures, len(cells))
    else:
        _print_real_run_report(
            uploaded, skipped_idempotent, skipped_in_flight, failures, ledger, len(cells),
        )
    for name in skipped_in_flight:
        print(f"  in-flight: {name}")
    for name, reason in failures:
        print(f"  FAIL: {name}: {reason}", file=sys.stderr)
    return 1 if failures else 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--cell", action="append", default=[],
        help="dataset name to upload (repeatable; default = the 11 SUCCEEDED cells)",
    )
    p.add_argument("--api-base", default=DEFAULT_API,
                   help="PROTEA API base URL (default: http://localhost:8000)")
    p.add_argument("--s3-endpoint", default=DEFAULT_S3_ENDPOINT,
                   help="MinIO/S3 endpoint (default: http://localhost:9000)")
    p.add_argument("--access-key", default="minioadmin",
                   help="S3 access key (default: minioadmin)")
    p.add_argument("--secret-key", default="minioadmin",
                   help="S3 secret key (default: minioadmin)")
    p.add_argument("--zenodo-api", default=DEFAULT_ZENODO_API,
                   help=(
                       "Zenodo API base (default: production zenodo.org; pass "
                       "https://sandbox.zenodo.org/api for the sandbox)"
                   ))
    p.add_argument("--sandbox", action="store_true",
                   help="shortcut for --zenodo-api https://sandbox.zenodo.org/api")
    p.add_argument("--datasets-root", type=Path, default=DATASETS_ROOT,
                   help="local datasets root (default: <repo>/datasets)")
    p.add_argument("--cards-root", type=Path, default=DATASET_CARDS_ROOT,
                   help="local dataset_cards root (default: <repo>/dataset_cards)")
    p.add_argument("--provenance-path", type=Path, default=PROVENANCE_PATH,
                   help="local provenance doc (default: <repo>/docs/dataset_provenance.md)")
    p.add_argument("--ledger", type=Path, default=DEPOSIT_LEDGER_PATH,
                   help="idempotency ledger path (default: <datasets>/.zenodo_deposits.json)")
    p.add_argument("--community", action="append", default=[], dest="communities",
                   help="Zenodo community identifier (repeatable)")
    p.add_argument("--grant", action="append", default=[], dest="grants",
                   help="Zenodo grant id (repeatable; e.g. 10.13039/501100011033::PID2020-...)")
    p.add_argument("--creator-name", default=CREATOR_NAME,
                   help=f"deposition creator name (default: {CREATOR_NAME!r})")
    p.add_argument("--license-id", default=LICENSE_ID,
                   help=f"Zenodo license id (default: {LICENSE_ID!r})")
    p.add_argument("--dry-run", action="store_true",
                   help=(
                       "do not call Zenodo; plan + validate + print what would be "
                       "uploaded. This is the mode CI ships green on."
                   ))
    p.add_argument("--verbose", action="store_true", help="enable info-level logging")
    return p


def _resolve_token(args: argparse.Namespace) -> str:
    """Return the Zenodo token from env, or empty in dry-run mode."""
    token = os.environ.get("ZENODO_TOKEN", "").strip()
    if args.dry_run:
        return token  # may be empty; dry-run never calls the API
    if not token:
        print(
            "error: ZENODO_TOKEN env var is required for real uploads. "
            "Store it in ~/.secrets/ and `set -a; source ~/.secrets/zenodo.env; set +a` "
            "before invoking. Use --dry-run to validate without uploading.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return token


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    if args.sandbox:
        args.zenodo_api = DEFAULT_SANDBOX_API
    token = _resolve_token(args)
    cfg = UploadConfig(
        api_base=args.api_base,
        s3_endpoint=args.s3_endpoint,
        access_key=args.access_key,
        secret_key=args.secret_key,
        zenodo_api=args.zenodo_api,
        zenodo_token=token,
        datasets_root=args.datasets_root,
        cards_root=args.cards_root,
        provenance_path=args.provenance_path,
        ledger_path=args.ledger,
        communities=tuple(args.communities),
        grants=tuple(args.grants),
        creator_name=args.creator_name,
        license_id=args.license_id,
        dry_run=args.dry_run,
    )
    cells = tuple(args.cell) if args.cell else DEFAULT_CELLS
    return run(cells, cfg)


if __name__ == "__main__":
    sys.exit(main())
