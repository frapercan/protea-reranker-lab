#!/usr/bin/env python
"""Upload a lab-trained booster to PROTEA's RerankerModel registry.

Usage:
    # Upload a single run dir (must contain model.txt, spec.yaml, run.json)
    python scripts/upload_model.py runs/study_v9/replication/nk-bpo_seed42

    # Upload all 9 v9 winners
    python scripts/upload_model.py --winners

The upload hits ``POST /reranker-models/import`` (multipart). PROTEA stores
the booster via its artifact store (MinIO in prod, local FS in dev) and
registers a ``RerankerModel`` row keyed by ``run_id``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import urllib.request
import urllib.error

REPO = Path(__file__).resolve().parents[1]
DEFAULT_API = "http://localhost:8000"
RESULTS_CSV = REPO / "runs" / "study_v9" / "replication" / "results.csv"


def _winner_dirs() -> list[Path]:
    rows = list(csv.DictReader(RESULTS_CSV.open()))
    best: dict[str, dict] = {}
    for r in rows:
        if r.get("status") != "ok":
            continue
        cell = r["spec"].rsplit("_seed", 1)[0]
        try:
            f = float(r["fmax"])
        except (TypeError, ValueError):
            continue
        if cell not in best or f > best[cell]["fmax"]:
            best[cell] = {**r, "fmax": f}
    return [REPO / row["output_dir"] for row in best.values()]


def _post_multipart(url: str, files: dict[str, tuple[str, bytes, str]],
                    forms: dict[str, str] | None = None) -> dict:
    """Tiny multipart POST without external deps (``requests`` not installed)."""
    import uuid
    boundary = f"----LabBoundary{uuid.uuid4().hex}"
    parts: list[bytes] = []
    for field_name, (filename, content, content_type) in files.items():
        parts.append(f"--{boundary}".encode())
        parts.append(
            f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"'.encode()
        )
        parts.append(f"Content-Type: {content_type}".encode())
        parts.append(b"")
        parts.append(content)
    for field_name, value in (forms or {}).items():
        parts.append(f"--{boundary}".encode())
        parts.append(f'Content-Disposition: form-data; name="{field_name}"'.encode())
        parts.append(b"")
        parts.append(value.encode())
    parts.append(f"--{boundary}--".encode())
    parts.append(b"")
    body = b"\r\n".join(parts)

    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc


def upload_run(run_dir: Path, *, api_url: str, name_override: str | None = None,
               external_source: str | None = None, force: bool = False) -> dict:
    model_file = run_dir / "model.txt"
    spec_file = run_dir / "spec.yaml"
    run_file = run_dir / "run.json"
    for p in (model_file, spec_file, run_file):
        if not p.exists():
            raise FileNotFoundError(f"missing {p}")

    files = {
        "model_file": ("model.txt", model_file.read_bytes(), "application/octet-stream"),
        "spec_yaml": ("spec.yaml", spec_file.read_bytes(), "text/yaml"),
        "run_json": ("run.json", run_file.read_bytes(), "application/json"),
    }
    forms: dict[str, str] = {}
    if name_override:
        forms["name"] = name_override
    if external_source:
        forms["external_source"] = external_source
    if force:
        forms["force"] = "true"

    url = f"{api_url.rstrip('/')}/reranker-models/import"
    return _post_multipart(url, files, forms)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("run_dirs", nargs="*", type=Path,
                   help="run directories (each containing model.txt, spec.yaml, run.json)")
    p.add_argument("--winners", action="store_true",
                   help="upload the 9 study_v9 cell winners (overrides positional args)")
    p.add_argument("--api", default=DEFAULT_API, help="PROTEA API base URL")
    p.add_argument("--name", default=None, help="override RerankerModel.name")
    p.add_argument("--external-source", "--external_source",
                   dest="external_source", default=None,
                   help='e.g. "protea-reranker-lab@<git_sha>"')
    p.add_argument("--force", action="store_true",
                   help="allow overwriting existing RerankerModel with same run_id")
    args = p.parse_args(argv)

    targets: list[Path]
    if args.winners:
        targets = _winner_dirs()
        if not targets:
            print(f"[err] no winners found in {RESULTS_CSV}")
            return 2
    elif args.run_dirs:
        targets = args.run_dirs
    else:
        p.error("pass run dirs or --winners")
        return 2

    # Resolve git sha for default external_source
    if not args.external_source:
        try:
            import subprocess
            sha = subprocess.check_output(
                ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                text=True,
            ).strip()
            args.external_source = f"protea-reranker-lab@{sha}"
        except subprocess.CalledProcessError:
            pass

    failures: list[str] = []
    for run_dir in targets:
        print(f"[upload] {run_dir.name}  →  {args.api}", flush=True)
        try:
            result = upload_run(
                run_dir, api_url=args.api,
                name_override=args.name if len(targets) == 1 else None,
                external_source=args.external_source, force=args.force,
            )
            print(f"  ok  id={result['id']}  artifact_uri={result['artifact_uri']}")
        except Exception as exc:
            print(f"  FAILED: {exc}")
            failures.append(run_dir.name)

    if failures:
        print(f"[done] {len(targets) - len(failures)}/{len(targets)} uploaded; "
              f"failed: {failures}")
        return 1
    print(f"[done] {len(targets)} model(s) uploaded successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
