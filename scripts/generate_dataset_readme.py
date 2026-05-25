#!/usr/bin/env python3
"""Generate per dataset README.md files from manifest + registry row.

Slice F-DATA-PACK.2 of the farm-platform loop. The dataset family
``bench-v1-K{3,5,10}-v226-lineage-{plm_short}`` is the primary
research deliverable per the 2026-05-25 user decision (memory
``[[dl-postponed-2026-05-25]]``). To be citable and reproducible from
a downloaded parquet alone, each cell needs a README alongside the
manifest with provenance, schema, splits, and known caveats.

Inputs per cell:

1. ``manifest.json`` fetched from the storage backend (MinIO/S3) at
   the registry row's ``manifest_uri`` (or, for the legacy prostt5
   alias, the underlying ``key_prefix``). Validated against the
   F-DATA-PACK.1 schema validator before render.
2. The PROTEA Dataset registry row served by ``GET /v1/datasets/<name>``
   (includes ``manifest_sha``, ``job_id``, ``created_at``, storage URIs,
   and the alias list in ``meta``).

Output:

    datasets/<name>/README.md

The template is ``scripts/templates/dataset_readme.md.tmpl`` and uses
``string.Template`` (no jinja dependency).

Cells in flight (no SUCCEEDED Dataset row yet) are skipped with a
warning; they will be filled in by a post-EXP.13 re-run of the same
script.

Exit codes:

    0 - clean (every requested cell generated and validated)
    1 - at least one cell failed validation or fetch
    2 - invocation error
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import logging
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any

import validate_manifest  # local script

log = logging.getLogger("generate_dataset_readme")


# Canonical dataset family. The 11th cell on 2026-05-25 is
# ``bench-v1-K5-v226-lineage-prostt5`` whose manifest carries the
# legacy PLM blind name (see alias_names in meta and ADR D36). The
# other 13 (3 PLMs x K3, plus ankh_base K3+K5 etc.) are in flight on
# protea.training; this script skips any cell whose registry GET 404s.
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
TEMPLATE_PATH: Path = Path(__file__).resolve().parent / "templates" / "dataset_readme.md.tmpl"
DATASETS_ROOT: Path = Path(__file__).resolve().parents[1] / "datasets"

_S3_URI_RE: re.Pattern[str] = re.compile(r"^s3://([^/]+)/(.+)$")


def _fetch_registry_row(api_base: str, name: str, timeout: int = 30) -> dict[str, Any] | None:
    """GET /v1/datasets/<name>. Returns None on 404, raises otherwise."""
    url = f"{api_base.rstrip('/')}/v1/datasets/{urllib.parse.quote(name, safe='')}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise


def _sign_v4(req: urllib.request.Request, *, access_key: str, secret_key: str,
             region: str = "us-east-1") -> None:
    """Attach AWS Signature v4 Authorization header to *req* in place.

    Mirrors ``scripts/pull_dataset.py`` to keep the lab self contained.
    Unsigned payload, GET only.
    """
    method = req.get_method()
    parsed = urllib.parse.urlparse(req.full_url)
    host = parsed.netloc
    path = parsed.path or "/"
    query = parsed.query
    now = time.gmtime()
    datestamp = time.strftime("%Y%m%d", now)
    amzdate = time.strftime("%Y%m%dT%H%M%SZ", now)
    canonical_headers = (
        f"host:{host}\nx-amz-content-sha256:UNSIGNED-PAYLOAD\nx-amz-date:{amzdate}\n"
    )
    signed_headers = "host;x-amz-content-sha256;x-amz-date"
    canonical_request = "\n".join([
        method, path, query,
        canonical_headers, signed_headers, "UNSIGNED-PAYLOAD",
    ])
    scope = f"{datestamp}/{region}/s3/aws4_request"
    string_to_sign = "\n".join([
        "AWS4-HMAC-SHA256", amzdate, scope,
        hashlib.sha256(canonical_request.encode("utf-8")).hexdigest(),
    ])

    def _hmac(key: bytes, msg: str) -> bytes:
        return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()

    signing_key = _hmac(
        _hmac(_hmac(_hmac(f"AWS4{secret_key}".encode("utf-8"), datestamp), region), "s3"),
        "aws4_request",
    )
    sig = hmac.new(
        signing_key, string_to_sign.encode("utf-8"), hashlib.sha256,
    ).hexdigest()
    req.add_header(
        "Authorization",
        f"AWS4-HMAC-SHA256 Credential={access_key}/{scope}, "
        f"SignedHeaders={signed_headers}, Signature={sig}",
    )
    req.add_header("x-amz-date", amzdate)
    req.add_header("x-amz-content-sha256", "UNSIGNED-PAYLOAD")


def _fetch_manifest_bytes(manifest_uri: str, *, s3_endpoint: str,
                          access_key: str, secret_key: str,
                          timeout: int = 30) -> bytes:
    """Resolve s3://bucket/key to *s3_endpoint*/bucket/key and GET."""
    m = _S3_URI_RE.match(manifest_uri)
    if not m:
        raise ValueError(f"manifest_uri is not an s3 URI: {manifest_uri!r}")
    bucket, key = m.group(1), m.group(2)
    url = f"{s3_endpoint.rstrip('/')}/{bucket}/{key}"
    req = urllib.request.Request(url)
    if access_key and secret_key:
        _sign_v4(req, access_key=access_key, secret_key=secret_key)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _alias_section(row: dict[str, Any], manifest_name: str) -> str:
    """Render the alias note for the prostt5 legacy ghost case."""
    aliases = (row.get("meta") or {}).get("alias_names") or []
    registry_name = row.get("name", "")
    if not aliases and manifest_name == registry_name:
        return ""
    lines = ["## Alias", ""]
    if manifest_name and manifest_name != registry_name:
        lines.append(
            f"The manifest's internal `name` field is `{manifest_name}`, "
            f"which differs from the registry name `{registry_name}`. "
            "This is the legacy ghost prostt5 case: the dataset was "
            "originally exported under a PLM blind name and later "
            "aliased to the canonical per PLM form. Both names resolve "
            "to the same parquet bytes."
        )
        lines.append("")
    if aliases:
        alias_list = ", ".join(f"`{a}`" for a in aliases)
        lines.append(f"Registered aliases: {alias_list}")
        lines.append("")
    return "\n".join(lines) + "\n"


def _render(template: str, row: dict[str, Any], manifest: dict[str, Any]) -> str:
    """Format the template with merged manifest + registry context."""
    train_pairs = manifest.get("train_snapshot_pairs") or []
    ctx: dict[str, Any] = {
        "name": row.get("name", ""),
        "manifest_name": manifest.get("name", ""),
        "schema_version": manifest.get("schema_version", ""),
        "schema_sha": row.get("schema_sha") or manifest.get("schema_sha") or "",
        "manifest_sha": row.get("manifest_sha") or "",
        "producer_version": manifest.get("producer_version", ""),
        "producer_git_sha": manifest.get("producer_git_sha", ""),
        "job_id": row.get("job_id") or "(legacy, no job id on record)",
        "created_at": row.get("created_at", ""),
        "embedding_config_id": row.get("embedding_config_id", ""),
        "ontology_snapshot_id": row.get("ontology_snapshot_id", ""),
        "k": row.get("k", ""),
        "annotation_source": row.get("annotation_source", ""),
        "train_uri": row.get("train_uri", ""),
        "eval_uri": row.get("eval_uri", ""),
        "manifest_uri": row.get("manifest_uri", ""),
        "storage_backend": row.get("storage_backend", ""),
        "key_prefix": row.get("key_prefix", ""),
        "alias_section": _alias_section(row, manifest.get("name", "")),
        "n_train_pairs": len(train_pairs),
        "train_snapshot_pairs": ", ".join(f"`{p}`" for p in train_pairs),
        "eval_snapshot_pair": manifest.get("eval_snapshot_pair", ""),
        "n_train_rows": f"{row.get('n_train_rows', 0):,}",
        "n_eval_rows": f"{row.get('n_eval_rows', 0):,}",
    }
    return Template(template).substitute(ctx)


def _validate_or_raise(manifest_bytes: bytes, name: str) -> dict[str, Any]:
    """Parse + validate. Raises ValueError on validation failure."""
    data = json.loads(manifest_bytes.decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{name}: manifest top level must be an object")
    errors = validate_manifest.validate_manifest(data)
    if errors:
        joined = "; ".join(errors)
        raise ValueError(f"{name}: manifest failed validation: {joined}")
    return data


@dataclass(frozen=True)
class GenerateConfig:
    """Static config shared across all per cell generate_one() calls."""
    api_base: str
    s3_endpoint: str
    access_key: str
    secret_key: str
    datasets_root: Path
    template: str
    dry_run: bool = False


def generate_one(name: str, cfg: GenerateConfig) -> Path | None:
    """Generate the README for one cell. Returns the output path or None if skipped."""
    row = _fetch_registry_row(cfg.api_base, name)
    if row is None:
        log.warning("%s: no registry row (in flight or missing) - skipping", name)
        return None
    manifest_uri = row.get("manifest_uri")
    if not manifest_uri:
        log.warning("%s: registry row has no manifest_uri - skipping", name)
        return None
    manifest_bytes = _fetch_manifest_bytes(
        manifest_uri, s3_endpoint=cfg.s3_endpoint,
        access_key=cfg.access_key, secret_key=cfg.secret_key,
    )
    manifest = _validate_or_raise(manifest_bytes, name)
    rendered = _render(cfg.template, row, manifest)
    out_path = cfg.datasets_root / name / "README.md"
    if cfg.dry_run:
        log.info("%s: would write %d bytes to %s", name, len(rendered), out_path)
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(rendered, encoding="utf-8")
    log.info("%s: wrote %s (%d bytes)", name, out_path, len(rendered))
    return out_path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--cell", action="append", default=[],
        help="dataset name to generate (repeatable; default = all 11 SUCCEEDED cells)",
    )
    p.add_argument("--api-base", default=DEFAULT_API,
                   help="PROTEA API base URL (default: http://localhost:8000)")
    p.add_argument("--s3-endpoint", default=DEFAULT_S3_ENDPOINT,
                   help="MinIO/S3 endpoint (default: http://localhost:9000)")
    p.add_argument("--access-key", default="minioadmin",
                   help="S3 access key (default: minioadmin)")
    p.add_argument("--secret-key", default="minioadmin",
                   help="S3 secret key (default: minioadmin)")
    p.add_argument("--datasets-root", type=Path, default=DATASETS_ROOT,
                   help="output root (default: <repo>/datasets)")
    p.add_argument("--template", type=Path, default=TEMPLATE_PATH,
                   help="template path (default: <repo>/scripts/templates/dataset_readme.md.tmpl)")
    p.add_argument("--dry-run", action="store_true",
                   help="validate + render but do not write the README")
    p.add_argument("--verbose", action="store_true", help="enable info-level logging")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        template = args.template.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"error: cannot read template {args.template}: {exc}", file=sys.stderr)
        return 2
    cfg = GenerateConfig(
        api_base=args.api_base,
        s3_endpoint=args.s3_endpoint,
        access_key=args.access_key,
        secret_key=args.secret_key,
        datasets_root=args.datasets_root,
        template=template,
        dry_run=args.dry_run,
    )
    cells = tuple(args.cell) if args.cell else DEFAULT_CELLS
    failures: list[tuple[str, str]] = []
    written: list[Path] = []
    skipped: list[str] = []
    for name in cells:
        try:
            result = generate_one(name, cfg)
        except (ValueError, urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
            log.error("%s: FAIL - %s", name, exc)
            failures.append((name, str(exc)))
            continue
        if result is None:
            skipped.append(name)
        else:
            written.append(result)
    print(
        f"generate_dataset_readme: {len(written)} written, "
        f"{len(skipped)} skipped, {len(failures)} failed (of {len(cells)} cells)"
    )
    for name in skipped:
        print(f"  skipped: {name}")
    for name, reason in failures:
        print(f"  FAIL: {name}: {reason}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
