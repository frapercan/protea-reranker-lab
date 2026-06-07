"""Tests for the multi-manifest pooled loader (F-RERANK-UNIVERSAL.2).

Smoke pool: 2 PLM x K3,K5 over the v226-lineage manifests available in the
dev workspace. Tests verify:

1. ManifestSource.from_path infers plm_id and k_context from the manifest.
2. MultiManifestSpec has a bit-stable schema_sha.
3. multi_source_iter_batches yields batches with plm_id and k_context injected.
4. The total pooled row count equals the sum of per-manifest filtered counts.
5. plm_id and k_context vary across rows (confirming multi-source tagging works).
6. DatasetRef accepts multi_manifests and rejects all-None or double-set.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pytest

from protea_reranker_lab.experiment import DatasetRef
from protea_reranker_lab.multi_source import (
    ManifestSource,
    MultiManifestSpec,
    build_smoke_pool,
    count_multi_source_rows,
    multi_source_iter_batches,
    per_source_row_counts,
)


# ---------------------------------------------------------------------------
# Dataset root (dev workspace; tests skip if parquets are absent)
# ---------------------------------------------------------------------------

_DEV_DATASETS = Path(
    "/home/frapercan/Thesis2/repositories/protea-reranker-lab/datasets"
)

_SMOKE_PLM_IDS = ["prot_t5", "esm2_650m"]
_SMOKE_K_VALUES = [3, 5]


def _smoke_manifests() -> list[Path]:
    """Return paths to the 4 smoke manifests (2 PLM x K3,K5), or [] if absent."""
    manifests = []
    for plm in _SMOKE_PLM_IDS:
        for k in _SMOKE_K_VALUES:
            p = _DEV_DATASETS / f"bench-v1-K{k}-v226-lineage-{plm}" / "manifest.json"
            if p.exists():
                manifests.append(p)
    return manifests


_SMOKE_MANIFESTS = _smoke_manifests()
_HAVE_SMOKE = len(_SMOKE_MANIFESTS) == 4
_SKIP_MSG = (
    f"smoke manifests not present under {_DEV_DATASETS} "
    f"(need bench-v1-K{{3,5}}-v226-lineage-{{prot_t5,esm2_650m}}/manifest.json)"
)


# ---------------------------------------------------------------------------
# Unit tests (no parquet I/O)
# ---------------------------------------------------------------------------


class TestManifestSource:
    @pytest.mark.skipif(not _HAVE_SMOKE, reason=_SKIP_MSG)
    def test_from_path_infers_plm_and_k(self) -> None:
        p = _DEV_DATASETS / "bench-v1-K5-v226-lineage-prot_t5" / "manifest.json"
        src = ManifestSource.from_path(p)
        assert src.plm_id == "prot_t5"
        assert src.k_context == 5

    @pytest.mark.skipif(not _HAVE_SMOKE, reason=_SKIP_MSG)
    def test_from_path_override_plm(self) -> None:
        p = _DEV_DATASETS / "bench-v1-K5-v226-lineage-prot_t5" / "manifest.json"
        src = ManifestSource.from_path(p, plm_id="custom_plm")
        assert src.plm_id == "custom_plm"

    @pytest.mark.skipif(not _HAVE_SMOKE, reason=_SKIP_MSG)
    def test_parquet_dir_is_manifest_parent(self) -> None:
        p = _DEV_DATASETS / "bench-v1-K3-v226-lineage-prot_t5" / "manifest.json"
        src = ManifestSource.from_path(p)
        assert src.parquet_dir == p.parent


class TestMultiManifestSpec:
    @pytest.mark.skipif(not _HAVE_SMOKE, reason=_SKIP_MSG)
    def test_schema_sha_is_12_hex(self) -> None:
        spec = MultiManifestSpec.from_manifest_paths(_SMOKE_MANIFESTS)
        assert len(spec.schema_sha) == 12
        int(spec.schema_sha, 16)

    @pytest.mark.skipif(not _HAVE_SMOKE, reason=_SKIP_MSG)
    def test_schema_sha_bit_stable(self) -> None:
        """Two specs from the same manifest list must have identical sha."""
        spec_a = MultiManifestSpec.from_manifest_paths(_SMOKE_MANIFESTS)
        spec_b = MultiManifestSpec.from_manifest_paths(_SMOKE_MANIFESTS)
        assert spec_a.schema_sha == spec_b.schema_sha

    @pytest.mark.skipif(not _HAVE_SMOKE, reason=_SKIP_MSG)
    def test_schema_sha_changes_when_source_added(self) -> None:
        """Adding a manifest must change the sha."""
        spec_a = MultiManifestSpec.from_manifest_paths(_SMOKE_MANIFESTS[:2])
        spec_b = MultiManifestSpec.from_manifest_paths(_SMOKE_MANIFESTS[:3])
        assert spec_a.schema_sha != spec_b.schema_sha

    @pytest.mark.skipif(not _HAVE_SMOKE, reason=_SKIP_MSG)
    def test_plm_ids_property(self) -> None:
        spec = MultiManifestSpec.from_manifest_paths(_SMOKE_MANIFESTS)
        assert set(spec.plm_ids) == set(_SMOKE_PLM_IDS)

    @pytest.mark.skipif(not _HAVE_SMOKE, reason=_SKIP_MSG)
    def test_k_values_property(self) -> None:
        spec = MultiManifestSpec.from_manifest_paths(_SMOKE_MANIFESTS)
        assert set(spec.k_values) == set(_SMOKE_K_VALUES)

    @pytest.mark.skipif(not _HAVE_SMOKE, reason=_SKIP_MSG)
    def test_build_smoke_pool(self) -> None:
        spec = build_smoke_pool(
            _DEV_DATASETS,
            plm_ids=_SMOKE_PLM_IDS,
            k_values=_SMOKE_K_VALUES,
        )
        assert len(spec.sources) == 4


# ---------------------------------------------------------------------------
# DatasetRef relaxation tests (no parquet I/O)
# ---------------------------------------------------------------------------


class TestDatasetRefRelaxation:
    def test_multi_manifests_accepted(self) -> None:
        ref = DatasetRef(multi_manifests=[Path("/fake/a.json"), Path("/fake/b.json")])
        assert ref.is_multi()
        assert ref.manifest is None
        assert ref.spec is None

    def test_all_none_raises(self) -> None:
        with pytest.raises(Exception):
            DatasetRef()  # type: ignore[call-arg]

    def test_manifest_and_multi_raises(self) -> None:
        with pytest.raises(Exception):
            DatasetRef(
                manifest=Path("/fake/m.json"),
                multi_manifests=[Path("/fake/a.json")],
            )

    def test_single_manifest_still_works(self) -> None:
        ref = DatasetRef(manifest=Path("/fake/m.json"))
        assert not ref.is_multi()
        assert ref.manifest == Path("/fake/m.json")


# ---------------------------------------------------------------------------
# Smoke pool tests (require parquet files in dev workspace)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAVE_SMOKE, reason=_SKIP_MSG)
class TestSmokePoolIterator:
    """Integration tests: stream 2 PLM x K3,K5 and verify row accounting."""

    def _build_spec(self) -> MultiManifestSpec:
        return MultiManifestSpec.from_manifest_paths(_SMOKE_MANIFESTS)

    def test_batches_have_plm_id_column(self) -> None:
        spec = self._build_spec()
        for batch, plm_id, k_ctx in multi_source_iter_batches(spec, split="eval"):
            assert "plm_id" in batch.schema.names
            assert batch.column("plm_id").type == pa.string()
            break  # just the first batch is enough

    def test_batches_have_k_context_column(self) -> None:
        spec = self._build_spec()
        for batch, plm_id, k_ctx in multi_source_iter_batches(spec, split="eval"):
            assert "k_context" in batch.schema.names
            assert batch.column("k_context").type == pa.int32()
            break

    def test_plm_id_constant_within_batch(self) -> None:
        """All rows in a batch must have the same plm_id (constant injection)."""
        spec = self._build_spec()
        for batch, plm_id, k_ctx in multi_source_iter_batches(spec, split="eval"):
            col = batch.column("plm_id").to_pylist()
            assert all(v == plm_id for v in col), (
                f"plm_id not constant within batch: expected {plm_id!r}, "
                f"got unique values {set(col)!r}"
            )
            break

    def test_k_context_constant_within_batch(self) -> None:
        """All rows in a batch must have the same k_context."""
        spec = self._build_spec()
        for batch, plm_id, k_ctx in multi_source_iter_batches(spec, split="eval"):
            col = batch.column("k_context").to_pylist()
            assert all(v == k_ctx for v in col), (
                f"k_context not constant within batch: expected {k_ctx}, "
                f"got unique values {set(col)!r}"
            )
            break

    def test_plm_id_varies_across_sources(self) -> None:
        """Over all batches the set of plm_id values must equal the source PLMs."""
        spec = self._build_spec()
        seen_plms: set[str] = set()
        for _batch, plm_id, _k_ctx in multi_source_iter_batches(spec, split="eval"):
            seen_plms.add(plm_id)
        assert seen_plms == set(_SMOKE_PLM_IDS), (
            f"Expected plm_ids {set(_SMOKE_PLM_IDS)!r}, saw {seen_plms!r}"
        )

    def test_k_context_varies_across_sources(self) -> None:
        """Over all batches the set of k_context values must equal the source Ks."""
        spec = self._build_spec()
        seen_ks: set[int] = set()
        for _batch, _plm_id, k_ctx in multi_source_iter_batches(spec, split="eval"):
            seen_ks.add(k_ctx)
        assert seen_ks == set(_SMOKE_K_VALUES), (
            f"Expected k_values {set(_SMOKE_K_VALUES)!r}, saw {seen_ks!r}"
        )

    def test_pooled_row_count_equals_sum_of_per_source(self) -> None:
        """Total rows from the iterator must equal the sum of per-source counts."""
        spec = self._build_spec()
        # Streaming count via PyArrow pushdown
        per_source = per_source_row_counts(spec, split="eval")
        expected_total = sum(n for _, _, n in per_source)

        # Streaming count via iterator
        actual_total = sum(
            batch.num_rows
            for batch, _, _ in multi_source_iter_batches(spec, split="eval")
        )
        assert actual_total == expected_total, (
            f"Pooled row count mismatch: iterator yielded {actual_total} rows, "
            f"but per-source counts sum to {expected_total}. "
            f"Per-source breakdown: {per_source!r}"
        )

    def test_count_multi_source_rows(self) -> None:
        """count_multi_source_rows must match the iterator's row count."""
        spec = self._build_spec()
        streaming_count = count_multi_source_rows(spec, split="eval")
        iterator_count = sum(
            batch.num_rows
            for batch, _, _ in multi_source_iter_batches(spec, split="eval")
        )
        assert streaming_count == iterator_count, (
            f"count_multi_source_rows={streaming_count} != "
            f"iterator sum={iterator_count}"
        )

    def test_schema_sha_is_stable(self) -> None:
        """schema_sha must not change between two spec constructions."""
        spec_a = MultiManifestSpec.from_manifest_paths(_SMOKE_MANIFESTS)
        spec_b = MultiManifestSpec.from_manifest_paths(_SMOKE_MANIFESTS)
        assert spec_a.schema_sha == spec_b.schema_sha

    def test_protein_accession_present_in_batches(self) -> None:
        """Batches must include protein_accession for downstream routing."""
        spec = self._build_spec()
        for batch, _, _ in multi_source_iter_batches(
            spec, split="eval", columns=["protein_accession", "label"]
        ):
            assert "protein_accession" in batch.schema.names
            break
