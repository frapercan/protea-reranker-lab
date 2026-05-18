#!/usr/bin/env python
"""Download a frozen feature dataset from a MinIO / S3-compatible store.

The store is accessed via its HTTP endpoint (not AWS SDK) so no new
dependencies are required beyond the Python stdlib.  Configuration is
passed via environment variables or CLI flags.

Usage:
    python scripts/pull_dataset.py bench-v1-K5

    # Override endpoint / bucket at runtime
    python scripts/pull_dataset.py bench-v1-K5 \\
        --endpoint http://localhost:9000 \\
        --bucket protea-datasets \\
        --dest datasets/

Environment variables (all optional if flags are given):
    PROTEA_S3_ENDPOINT   e.g. http://minio:9000
    PROTEA_S3_BUCKET     e.g. protea-datasets
    PROTEA_S3_PREFIX     key prefix inside the bucket (default "datasets")
    PROTEA_S3_ACCESS_KEY
    PROTEA_S3_SECRET_KEY

Resilience:
    - Transient errors (5xx, connection-reset, timeout) are retried with
      exponential backoff: delays 1, 2, 4, 8, 16 s (max 5 attempts).
    - Partial downloads are resumed via ``Range: bytes=<offset>-`` so a
      killed download continues from where it stopped.
    - Progress is logged every 30 s showing bytes-fetched / total.
    - ETag is verified on completion for non-multipart objects (ETag == MD5).

Dataset layout expected inside the bucket under <prefix>/<dataset_name>/:
    train.parquet
    eval.parquet
    manifest.json
    parent_map.json   (optional)
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import logging
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [pull_dataset] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Retry policy
# ---------------------------------------------------------------------------
_MAX_ATTEMPTS = 5
_BASE_DELAY_S = 1.0
_PROGRESS_INTERVAL_S = 30.0
_CHUNK_SIZE = 256 * 1024  # 256 KiB per read chunk

# HTTP status codes treated as transient (worth retrying)
_TRANSIENT_STATUS = frozenset({500, 502, 503, 504, 429})


def _is_transient(exc: Exception) -> bool:
    """Return True for errors that justify a retry."""
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in _TRANSIENT_STATUS
    if isinstance(exc, (urllib.error.URLError, ConnectionError,
                        TimeoutError, OSError)):
        return True
    return False


def _backoff_delay(attempt: int) -> float:
    """Return seconds to sleep before attempt (0-indexed)."""
    return _BASE_DELAY_S * (2 ** attempt)


# ---------------------------------------------------------------------------
# Minimal HMAC-SHA256 AWS Signature v4 signer (presigned-URL alternative:
# we skip signing when no credentials are present and rely on bucket-level
# public read, which is the typical lab setup for MinIO dev environments).
# When credentials ARE present we sign via Authorization header.
# ---------------------------------------------------------------------------

def _sign_request(req: urllib.request.Request, *,
                  access_key: str, secret_key: str,
                  region: str = "us-east-1") -> None:
    """Attach AWS Signature v4 Authorization header to *req* in-place.

    This implementation handles unsigned-payload GET/HEAD requests only,
    which covers all read operations needed by pull_dataset.
    """
    method = req.get_method()
    parsed = urllib.parse.urlparse(req.full_url)
    host = parsed.netloc
    path = parsed.path or "/"
    query = parsed.query

    now = time.gmtime()
    datestamp = time.strftime("%Y%m%d", now)
    amzdate = time.strftime("%Y%m%dT%H%M%SZ", now)

    # Canonical headers (must be sorted)
    canonical_headers = f"host:{host}\nx-amz-content-sha256:UNSIGNED-PAYLOAD\nx-amz-date:{amzdate}\n"
    signed_headers = "host;x-amz-content-sha256;x-amz-date"
    payload_hash = "UNSIGNED-PAYLOAD"

    canonical_qs = query  # already sorted by caller if needed
    canonical_request = "\n".join([
        method, path, canonical_qs,
        canonical_headers, signed_headers, payload_hash,
    ])

    credential_scope = f"{datestamp}/{region}/s3/aws4_request"
    string_to_sign = "\n".join([
        "AWS4-HMAC-SHA256", amzdate, credential_scope,
        hashlib.sha256(canonical_request.encode("utf-8")).hexdigest(),
    ])

    def _hmac(key: bytes, msg: str) -> bytes:
        return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()

    signing_key = _hmac(
        _hmac(
            _hmac(
                _hmac(f"AWS4{secret_key}".encode("utf-8"), datestamp),
                region,
            ),
            "s3",
        ),
        "aws4_request",
    )
    signature = hmac.new(signing_key, string_to_sign.encode("utf-8"),
                         hashlib.sha256).hexdigest()

    auth = (
        f"AWS4-HMAC-SHA256 Credential={access_key}/{credential_scope}, "
        f"SignedHeaders={signed_headers}, Signature={signature}"
    )
    req.add_header("Authorization", auth)
    req.add_header("x-amz-date", amzdate)
    req.add_header("x-amz-content-sha256", "UNSIGNED-PAYLOAD")


# ---------------------------------------------------------------------------
# Core download primitive with retry + resume
# ---------------------------------------------------------------------------

def _head_object(url: str, *, access_key: str, secret_key: str,
                 timeout: int = 30) -> tuple[int, str | None]:
    """Return (content_length, etag) for *url*, or (-1, None) on failure."""
    req = urllib.request.Request(url, method="HEAD")
    if access_key and secret_key:
        _sign_request(req, access_key=access_key, secret_key=secret_key)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            cl = int(resp.headers.get("Content-Length", -1))
            etag = resp.headers.get("ETag", "").strip('"')
            return cl, etag or None
    except Exception:
        return -1, None


def download_object(
    url: str,
    dest: Path,
    *,
    access_key: str = "",
    secret_key: str = "",
    timeout: int = 120,
) -> None:
    """Download *url* to *dest* with exponential backoff and resume.

    On each transient error the download is retried from the byte after the
    last byte already written to *dest* (partial-resume semantics).  The
    caller is responsible for removing *dest* if a clean retry is desired.

    Raises RuntimeError after _MAX_ATTEMPTS consecutive failures.
    Raises urllib.error.HTTPError for non-transient HTTP errors.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    # HEAD to learn total size and ETag
    total_size, expected_etag = _head_object(
        url, access_key=access_key, secret_key=secret_key, timeout=timeout
    )

    last_exc: Exception | None = None
    for attempt in range(_MAX_ATTEMPTS):
        resume_offset = dest.stat().st_size if dest.exists() else 0

        if total_size > 0 and resume_offset >= total_size:
            log.info("  %s already complete (%d bytes), skipping", dest.name, resume_offset)
            return

        if resume_offset > 0:
            log.info(
                "  %s: resuming from byte %d / %d (%.1f%%)",
                dest.name, resume_offset, total_size,
                100.0 * resume_offset / max(total_size, 1),
            )

        req = urllib.request.Request(url)
        if resume_offset > 0:
            req.add_header("Range", f"bytes={resume_offset}-")
        if access_key and secret_key:
            _sign_request(req, access_key=access_key, secret_key=secret_key)

        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                mode = "ab" if resume_offset > 0 else "wb"
                md5 = hashlib.md5()
                if resume_offset > 0:
                    # Re-hash already-downloaded portion so we can verify the
                    # full file at the end.  For large files this is a small
                    # sequential read — acceptable because it happens at most
                    # once per resume.
                    with open(dest, "rb") as fh:
                        while True:
                            chunk = fh.read(_CHUNK_SIZE)
                            if not chunk:
                                break
                            md5.update(chunk)

                with open(dest, mode) as fh:
                    fetched = resume_offset
                    last_log_t = time.monotonic()
                    while True:
                        chunk = resp.read(_CHUNK_SIZE)
                        if not chunk:
                            break
                        fh.write(chunk)
                        md5.update(chunk)
                        fetched += len(chunk)
                        now = time.monotonic()
                        if now - last_log_t >= _PROGRESS_INTERVAL_S:
                            if total_size > 0:
                                log.info(
                                    "  %s: %d / %d bytes (%.1f%%)",
                                    dest.name, fetched, total_size,
                                    100.0 * fetched / total_size,
                                )
                            else:
                                log.info("  %s: %d bytes fetched", dest.name, fetched)
                            last_log_t = now

            # Verify checksum — ETag is MD5 for non-multipart objects (no "-N" suffix)
            if expected_etag and "-" not in expected_etag:
                actual = md5.hexdigest()
                if actual != expected_etag:
                    dest.unlink(missing_ok=True)
                    raise RuntimeError(
                        f"checksum mismatch for {dest.name}: "
                        f"expected {expected_etag}, got {actual}"
                    )
                log.info("  %s: checksum ok (md5=%s)", dest.name, actual)
            else:
                log.info("  %s: downloaded %d bytes", dest.name, fetched)
            return  # success

        except RuntimeError:
            raise  # checksum failure is not retried
        except Exception as exc:
            last_exc = exc
            if not _is_transient(exc):
                raise
            delay = _backoff_delay(attempt)
            log.warning(
                "  %s: transient error (attempt %d/%d): %s — retrying in %.0fs",
                dest.name, attempt + 1, _MAX_ATTEMPTS, exc, delay,
            )
            time.sleep(delay)

    raise RuntimeError(
        f"Failed to download {dest.name} after {_MAX_ATTEMPTS} attempts"
    ) from last_exc


# ---------------------------------------------------------------------------
# Dataset pull
# ---------------------------------------------------------------------------

_REQUIRED_FILES = ("manifest.json", "train.parquet", "eval.parquet")
_OPTIONAL_FILES = ("parent_map.json",)


def pull_dataset(
    dataset_name: str,
    *,
    endpoint: str,
    bucket: str,
    prefix: str = "datasets",
    dest_root: Path,
    access_key: str = "",
    secret_key: str = "",
    force: bool = False,
) -> Path:
    """Pull all files for *dataset_name* from S3/MinIO into *dest_root*.

    Returns the local dataset directory path.
    """
    dest_dir = dest_root / dataset_name
    dest_dir.mkdir(parents=True, exist_ok=True)

    base_url = endpoint.rstrip("/")
    key_prefix = f"{prefix}/{dataset_name}" if prefix else dataset_name

    log.info("Pulling dataset '%s' from %s/%s/%s", dataset_name, base_url, bucket, key_prefix)

    files_to_fetch = list(_REQUIRED_FILES) + list(_OPTIONAL_FILES)
    for filename in files_to_fetch:
        dest_file = dest_dir / filename
        required = filename in _REQUIRED_FILES

        if dest_file.exists() and not force:
            # Quick size sanity: skip only if file is non-empty (avoids
            # silently keeping truncated artefacts from a prior crash).
            if dest_file.stat().st_size > 0:
                log.info("  %s: already present (%d bytes), skipping",
                         filename, dest_file.stat().st_size)
                continue

        object_key = f"{key_prefix}/{filename}"
        url = f"{base_url}/{bucket}/{object_key}"

        log.info("  Fetching %s …", filename)
        try:
            download_object(
                url, dest_file,
                access_key=access_key,
                secret_key=secret_key,
            )
        except urllib.error.HTTPError as exc:
            if exc.code == 404 and not required:
                log.info("  %s: not found on remote (optional — skipping)", filename)
                continue
            raise

    log.info("Dataset '%s' ready at %s", dataset_name, dest_dir)
    return dest_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("dataset", help="dataset name (e.g. bench-v1-K5)")
    p.add_argument(
        "--endpoint",
        default=os.environ.get("PROTEA_S3_ENDPOINT", "http://localhost:9000"),
        help="S3/MinIO HTTP endpoint (default: $PROTEA_S3_ENDPOINT or http://localhost:9000)",
    )
    p.add_argument(
        "--bucket",
        default=os.environ.get("PROTEA_S3_BUCKET", "protea-datasets"),
        help="bucket name (default: $PROTEA_S3_BUCKET or protea-datasets)",
    )
    p.add_argument(
        "--prefix",
        default=os.environ.get("PROTEA_S3_PREFIX", "datasets"),
        help="key prefix inside the bucket (default: $PROTEA_S3_PREFIX or 'datasets')",
    )
    p.add_argument(
        "--dest",
        default="datasets",
        type=Path,
        help="local root directory for datasets (default: datasets/)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="re-download even if files are already present",
    )
    p.add_argument(
        "--access-key",
        default=os.environ.get("PROTEA_S3_ACCESS_KEY", ""),
        help="S3 access key ($PROTEA_S3_ACCESS_KEY)",
    )
    p.add_argument(
        "--secret-key",
        default=os.environ.get("PROTEA_S3_SECRET_KEY", ""),
        help="S3 secret key ($PROTEA_S3_SECRET_KEY)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    dest_dir = pull_dataset(
        args.dataset,
        endpoint=args.endpoint,
        bucket=args.bucket,
        prefix=args.prefix,
        dest_root=args.dest,
        access_key=args.access_key,
        secret_key=args.secret_key,
        force=args.force,
    )
    print(f"[done] {dest_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
