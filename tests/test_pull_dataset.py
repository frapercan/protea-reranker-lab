"""Tests for scripts/pull_dataset.py.

All S3 I/O is mocked so no live MinIO instance is required.
"""

from __future__ import annotations

import hashlib
import sys
import urllib.error
import urllib.request
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Make the scripts dir importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import pull_dataset as pd  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_response(body: bytes, status: int = 200, headers: dict | None = None) -> MagicMock:
    """Build a context-manager mock that mimics urllib response."""
    headers = headers or {}
    resp = MagicMock()
    resp.status = status
    resp.headers = MagicMock()
    resp.headers.get = lambda key, default="": headers.get(key, default)
    buf = BytesIO(body)
    resp.read = buf.read
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


def _make_http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        url="http://fake",
        code=code,
        msg=f"HTTP {code}",
        hdrs=None,  # type: ignore[arg-type]
        fp=None,
    )


# ---------------------------------------------------------------------------
# _is_transient
# ---------------------------------------------------------------------------

class TestIsTransient:
    def test_http_503_is_transient(self):
        assert pd._is_transient(_make_http_error(503))

    def test_http_429_is_transient(self):
        assert pd._is_transient(_make_http_error(429))

    def test_http_404_is_not_transient(self):
        assert not pd._is_transient(_make_http_error(404))

    def test_connection_error_is_transient(self):
        assert pd._is_transient(ConnectionError("reset"))

    def test_timeout_is_transient(self):
        assert pd._is_transient(TimeoutError("timed out"))

    def test_url_error_is_transient(self):
        assert pd._is_transient(urllib.error.URLError("name resolution"))


# ---------------------------------------------------------------------------
# download_object — retry on transient errors
# ---------------------------------------------------------------------------

class TestDownloadObjectRetry:
    """503 returned N times, then 200 with content — assert backoff + success."""

    def test_retries_503_then_succeeds(self, tmp_path):
        content = b"hello world dataset bytes"
        etag = hashlib.md5(content).hexdigest()

        call_count = 0

        def fake_urlopen(req, timeout=None):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise _make_http_error(503)
            headers = {"Content-Length": str(len(content)), "ETag": f'"{etag}"'}
            return _fake_response(content, headers=headers)

        dest = tmp_path / "file.parquet"

        with patch.object(pd.urllib.request, "urlopen", side_effect=fake_urlopen), \
             patch.object(pd.time, "sleep") as mock_sleep, \
             patch.object(pd, "_head_object", return_value=(len(content), etag)):
            pd.download_object("http://fake/obj", dest)

        assert dest.exists()
        assert dest.read_bytes() == content
        # Two failures → two sleeps; backoff delays were 1s and 2s
        assert mock_sleep.call_count == 2
        delays = [c.args[0] for c in mock_sleep.call_args_list]
        assert delays[1] >= delays[0], "backoff delay should increase"

    def test_raises_after_max_attempts(self, tmp_path):
        dest = tmp_path / "file.parquet"

        with patch.object(pd.urllib.request, "urlopen",
                          side_effect=_make_http_error(503)), \
             patch.object(pd.time, "sleep"), \
             patch.object(pd, "_head_object", return_value=(-1, None)):
            with pytest.raises(RuntimeError, match="Failed to download"):
                pd.download_object("http://fake/obj", dest)

    def test_non_transient_error_propagates_immediately(self, tmp_path):
        dest = tmp_path / "file.parquet"
        call_count = 0

        def fake_urlopen(req, timeout=None):
            nonlocal call_count
            call_count += 1
            raise _make_http_error(403)

        with patch.object(pd.urllib.request, "urlopen", side_effect=fake_urlopen), \
             patch.object(pd.time, "sleep") as mock_sleep, \
             patch.object(pd, "_head_object", return_value=(-1, None)):
            with pytest.raises(urllib.error.HTTPError) as exc_info:
                pd.download_object("http://fake/obj", dest)

        assert exc_info.value.code == 403
        assert call_count == 1, "should not retry non-transient error"
        mock_sleep.assert_not_called()


# ---------------------------------------------------------------------------
# download_object — resume via Range header
# ---------------------------------------------------------------------------

class TestDownloadObjectResume:
    """Simulate a partial file on disk, then a ConnectionError, then success
    with the Range header set correctly."""

    def test_resumes_partial_download(self, tmp_path):
        prefix = b"partial data already written"
        suffix = b" and the rest of the file"
        full_content = prefix + suffix
        etag = hashlib.md5(full_content).hexdigest()

        dest = tmp_path / "data.parquet"
        dest.write_bytes(prefix)  # simulate partially downloaded file

        seen_ranges: list[str] = []
        call_count = 0

        def fake_urlopen(req, timeout=None):
            nonlocal call_count
            call_count += 1
            range_header = req.get_header("Range") or ""
            seen_ranges.append(range_header)
            if call_count == 1:
                raise ConnectionError("reset by peer")
            # Second attempt: serve suffix
            headers = {
                "Content-Length": str(len(suffix)),
                "ETag": f'"{etag}"',
            }
            return _fake_response(suffix, headers=headers)

        with patch.object(pd.urllib.request, "urlopen", side_effect=fake_urlopen), \
             patch.object(pd.time, "sleep"), \
             patch.object(pd, "_head_object", return_value=(len(full_content), etag)):
            pd.download_object("http://fake/obj", dest)

        assert dest.read_bytes() == full_content
        # Both attempts should carry the Range header (file existed before first attempt)
        assert all(r.startswith("bytes=") for r in seen_ranges), (
            f"expected Range header on every attempt, got: {seen_ranges}"
        )
        expected_offset = len(prefix)
        assert f"bytes={expected_offset}-" in seen_ranges[0]

    def test_checksum_mismatch_raises_and_deletes_file(self, tmp_path):
        content = b"some content"
        bad_etag = "deadbeef" * 4  # wrong MD5
        dest = tmp_path / "bad.parquet"

        def fake_urlopen(req, timeout=None):
            return _fake_response(content, headers={
                "Content-Length": str(len(content)),
                "ETag": f'"{bad_etag}"',
            })

        with patch.object(pd.urllib.request, "urlopen", side_effect=fake_urlopen), \
             patch.object(pd, "_head_object", return_value=(len(content), bad_etag)):
            with pytest.raises(RuntimeError, match="checksum mismatch"):
                pd.download_object("http://fake/obj", dest)

        assert not dest.exists(), "corrupt file should be removed on checksum failure"


# ---------------------------------------------------------------------------
# pull_dataset — integration (all files present)
# ---------------------------------------------------------------------------

class TestPullDataset:
    def _make_downloader(self, files: dict[str, bytes]):
        """Return a side_effect for download_object that writes fake file bodies."""
        def _fake_download(url, dest, *, access_key="", secret_key="", timeout=120):
            name = Path(url).name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(files.get(name, b""))
        return _fake_download

    def test_downloads_required_files(self, tmp_path):
        fake_files = {
            "manifest.json": b'{"name":"bench-v1-K5"}',
            "train.parquet": b"PAR1fakecontent",
            "eval.parquet": b"PAR1fakecontent",
        }
        with patch.object(pd, "download_object",
                          side_effect=self._make_downloader(fake_files)):
            dest = pd.pull_dataset(
                "bench-v1-K5",
                endpoint="http://minio:9000",
                bucket="protea-datasets",
                dest_root=tmp_path,
            )

        assert (dest / "manifest.json").exists()
        assert (dest / "train.parquet").exists()
        assert (dest / "eval.parquet").exists()

    def test_optional_file_404_is_skipped(self, tmp_path):
        def fake_download(url, dest, **kwargs):
            name = Path(url).name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if name == "parent_map.json":
                raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)  # type: ignore
            dest.write_bytes(b"content")

        with patch.object(pd, "download_object", side_effect=fake_download):
            dest = pd.pull_dataset(
                "bench-v1-K5",
                endpoint="http://minio:9000",
                bucket="protea-datasets",
                dest_root=tmp_path,
            )

        assert not (dest / "parent_map.json").exists()

    def test_skip_non_empty_existing_files(self, tmp_path):
        dest_dir = tmp_path / "bench-v1-K5"
        dest_dir.mkdir(parents=True)
        (dest_dir / "manifest.json").write_bytes(b'{"name":"bench-v1-K5"}')
        (dest_dir / "train.parquet").write_bytes(b"PAR1content")
        (dest_dir / "eval.parquet").write_bytes(b"PAR1content")

        download_calls: list[str] = []

        def fake_download(url, dest, **kwargs):
            download_calls.append(Path(url).name)
            dest.write_bytes(b"new")

        with patch.object(pd, "download_object", side_effect=fake_download):
            pd.pull_dataset(
                "bench-v1-K5",
                endpoint="http://minio:9000",
                bucket="protea-datasets",
                dest_root=tmp_path,
            )

        # Only the optional parent_map.json (missing) should trigger a download attempt;
        # the three required files already exist and are non-empty.
        assert "manifest.json" not in download_calls
        assert "train.parquet" not in download_calls
        assert "eval.parquet" not in download_calls
