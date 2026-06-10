"""Multi-manifest pooled loader for the universal multi-PLM reranker.

Builds a RUNTIME VIEW (no physical all-PLM parquet) over the 24 v226-lineage
manifests (8 PLM x K{3,5,10}).  Each manifest source contributes its parquet
rows tagged with two stage-time constants:

- ``plm_id``   (str, categorical): the PLM that produced the embeddings.
- ``k_context`` (int, numeric):    the KNN neighbourhood size K.

Both columns are absent from the raw parquet dumps; they are injected by this
module at iteration time, before any staging pipeline processes the batch.

Design constraints (from the task spec and hard constraints):

- NEVER torch GPU KNN / pgvector.  This module is pure CPU/numpy.
- No physical all-PLM parquet: rows are streamed and concatenated lazily.
- schema_sha is derived from ``"|".join(sorted(str(p) for p in manifest_uris))``,
  so adding or removing a source changes the sha and invalidates cached boosters.

Usage::

    from pathlib import Path
    from protea_reranker_lab.multi_source import MultiManifestSpec, multi_source_iter_batches

    spec = MultiManifestSpec.from_manifest_paths([
        Path("datasets/bench-v1-K3-v226-lineage-prot_t5/manifest.json"),
        Path("datasets/bench-v1-K5-v226-lineage-prot_t5/manifest.json"),
        Path("datasets/bench-v1-K3-v226-lineage-esm2_650m/manifest.json"),
        Path("datasets/bench-v1-K5-v226-lineage-esm2_650m/manifest.json"),
    ])

    for batch, plm, k in multi_source_iter_batches(spec, category="nk", aspect="mfo"):
        # batch is a pa.RecordBatch with plm_id and k_context already injected.
        ...

The injected columns use the same names registered in
``protea_contracts.feature_schema.FEATURE_FAMILIES``: ``plm_id`` (categorical,
family ``plm_context``) and ``k_context`` (numeric, family ``k_neighborhood``).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import pyarrow as pa

from .data import iter_batches


# ---------------------------------------------------------------------------
# Per-source record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ManifestSource:
    """One entry in a multi-manifest pool.

    Attributes:
        manifest_path: absolute or relative path to a ``manifest.json`` file.
        plm_id:        identifier of the PLM (e.g. ``"prot_t5"``).  If
                       ``None`` the loader infers it from the manifest's
                       ``embedding_config_id`` field (see
                       :func:`_infer_plm_id`).
        k_context:     KNN neighbourhood size K.  If ``None`` it is inferred
                       from ``manifest.json``'s ``k`` field.
        parquet_dir:   directory containing ``train.parquet`` and
                       ``eval.parquet``.  Defaults to the manifest's parent.
    """

    manifest_path: Path
    plm_id: str
    k_context: int
    parquet_dir: Path

    @classmethod
    def from_path(
        cls,
        manifest_path: Path | str,
        *,
        plm_id: str | None = None,
        k_context: int | None = None,
    ) -> "ManifestSource":
        manifest_path = Path(manifest_path)
        data = json.loads(manifest_path.read_text())
        resolved_plm = plm_id or _plm_id_from_name(data.get("name", ""))
        resolved_k = k_context if k_context is not None else int(data["k"])
        return cls(
            manifest_path=manifest_path,
            plm_id=resolved_plm,
            k_context=resolved_k,
            parquet_dir=manifest_path.parent,
        )


# ---------------------------------------------------------------------------
# Multi-manifest spec
# ---------------------------------------------------------------------------


@dataclass
class MultiManifestSpec:
    """Pool specification for the universal multi-PLM loader.

    Attributes:
        sources:    ordered list of :class:`ManifestSource` entries.
        schema_sha: 12-hex digest of ``"|".join(sorted(str(p) for p in
                    manifest_paths))``.  Bit-stable: adding or removing a
                    source changes it.
    """

    sources: list[ManifestSource] = field(default_factory=list)
    schema_sha: str = ""

    def __post_init__(self) -> None:
        if not self.schema_sha and self.sources:
            self.schema_sha = _compute_pool_sha(self.sources)

    @classmethod
    def from_manifest_paths(
        cls,
        paths: Sequence[Path | str],
        *,
        plm_overrides: dict[str, str] | None = None,
        k_overrides: dict[str, int] | None = None,
    ) -> "MultiManifestSpec":
        """Build a spec from a list of manifest paths.

        Args:
            paths:         list of ``manifest.json`` paths.
            plm_overrides: mapping from path-str to explicit plm_id.
            k_overrides:   mapping from path-str to explicit k_context.
        """
        plm_ov = plm_overrides or {}
        k_ov = k_overrides or {}
        sources = [
            ManifestSource.from_path(
                p,
                plm_id=plm_ov.get(str(p)),
                k_context=k_ov.get(str(p)),
            )
            for p in paths
        ]
        return cls(sources=sources)

    @property
    def plm_ids(self) -> list[str]:
        """All distinct PLM identifiers in this pool (insertion order)."""
        seen: dict[str, None] = {}
        for s in self.sources:
            seen[s.plm_id] = None
        return list(seen)

    @property
    def k_values(self) -> list[int]:
        """All distinct K values in this pool (insertion order)."""
        seen: dict[int, None] = {}
        for s in self.sources:
            seen[s.k_context] = None
        return list(seen)


# ---------------------------------------------------------------------------
# Filter options dataclass
# ---------------------------------------------------------------------------


@dataclass
class PoolFilter:
    """Row-level filter options for the multi-source iterator.

    Attributes:
        columns:        column subset to read (``None`` = all columns).
        category:       PyArrow pushdown filter on ``category``.
        aspect:         PyArrow pushdown filter on ``aspect``.
        snapshot_pair:  single snapshot pair filter.
        snapshot_pairs: multi-pair filter (mutually exclusive with
                        ``snapshot_pair``).
        batch_size:     rows per batch (before protein filtering).
        protein_filter: if set, only rows whose ``protein_accession`` is in
                        this set are yielded (post-scan; not pushed down to
                        PyArrow because the set may be large).
    """

    columns: list[str] | None = None
    category: str | None = None
    aspect: str | None = None
    snapshot_pair: str | None = None
    snapshot_pairs: list[str] | None = None
    batch_size: int = 200_000
    protein_filter: frozenset[str] | None = None


# ---------------------------------------------------------------------------
# Core streaming iterator
# ---------------------------------------------------------------------------


def multi_source_iter_batches(
    spec: MultiManifestSpec,
    *,
    split: str = "train",
    filt: PoolFilter | None = None,
    **kwargs: object,
) -> Iterator[tuple[pa.RecordBatch, str, int]]:
    """Yield ``(batch, plm_id, k_context)`` tuples across all manifest sources.

    Each batch is a :class:`pyarrow.RecordBatch` with ``plm_id`` and
    ``k_context`` columns appended (constant per source).

    Pass a :class:`PoolFilter` via ``filt`` for row-level filtering, or pass
    keyword arguments that map to :class:`PoolFilter` fields for convenience::

        for batch, plm, k in multi_source_iter_batches(spec, category="nk"):
            ...

    Rows are yielded in source order; protein contiguity is preserved within
    each source batch. The downstream ``stage_for_training`` call re-buckets
    by protein so cross-source ordering does not matter.
    """
    if filt is None:
        filt = PoolFilter(**{k: v for k, v in kwargs.items() if v is not None})  # type: ignore[arg-type]
    for src in spec.sources:
        yield from _iter_source_batches(src, split=split, filt=filt)


def _iter_source_batches(
    src: ManifestSource,
    *,
    split: str,
    filt: PoolFilter,
) -> Iterator[tuple[pa.RecordBatch, str, int]]:
    """Yield tagged batches from one :class:`ManifestSource`."""
    parquet_path = src.parquet_dir / f"{split}.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Expected {split}.parquet for source {src.manifest_path}: "
            f"not found at {parquet_path}"
        )
    for raw_batch in iter_batches(
        parquet_path,
        columns=filt.columns,
        category=filt.category,
        aspect=filt.aspect,
        snapshot_pair=filt.snapshot_pair,
        snapshot_pairs=filt.snapshot_pairs,
        batch_size=filt.batch_size,
    ):
        if raw_batch.num_rows == 0:
            continue
        batch = _inject_source_columns(raw_batch, src.plm_id, src.k_context)
        if filt.protein_filter is not None:
            batch = _apply_protein_filter(batch, filt.protein_filter)
            if batch is None:
                continue
        yield batch, src.plm_id, src.k_context


def _apply_protein_filter(
    batch: pa.RecordBatch, protein_filter: frozenset[str]
) -> pa.RecordBatch | None:
    """Return the batch filtered to ``protein_filter``, or ``None`` if empty."""
    accessions = batch.column("protein_accession").to_numpy(zero_copy_only=False)
    mask = np.fromiter(
        (a in protein_filter for a in accessions),
        dtype=bool,
        count=len(accessions),
    )
    if not mask.any():
        return None
    return batch.take(pa.array(np.flatnonzero(mask)))


def count_multi_source_rows(
    spec: MultiManifestSpec,
    *,
    split: str = "train",
    category: str | None = None,
    aspect: str | None = None,
    snapshot_pair: str | None = None,
    snapshot_pairs: list[str] | None = None,
) -> int:
    """Streaming count of rows matching the given filter across all sources.

    Uses PyArrow's pushdown; does not materialise any batch in memory.
    """
    from .data import count_filtered_rows

    total = 0
    for src in spec.sources:
        parquet_path = src.parquet_dir / f"{split}.parquet"
        if not parquet_path.exists():
            continue
        total += count_filtered_rows(
            parquet_path,
            category=category,
            aspect=aspect,
            snapshot_pair=snapshot_pair,
            snapshot_pairs=snapshot_pairs,
        )
    return total


def per_source_row_counts(
    spec: MultiManifestSpec,
    *,
    split: str = "train",
    category: str | None = None,
    aspect: str | None = None,
    snapshot_pair: str | None = None,
    snapshot_pairs: list[str] | None = None,
) -> list[tuple[str, int, int]]:
    """Return ``[(plm_id, k_context, n_rows)]`` per source for diagnostics."""
    from .data import count_filtered_rows

    result: list[tuple[str, int, int]] = []
    for src in spec.sources:
        parquet_path = src.parquet_dir / f"{split}.parquet"
        if not parquet_path.exists():
            result.append((src.plm_id, src.k_context, 0))
            continue
        n = count_filtered_rows(
            parquet_path,
            category=category,
            aspect=aspect,
            snapshot_pair=snapshot_pair,
            snapshot_pairs=snapshot_pairs,
        )
        result.append((src.plm_id, src.k_context, n))
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _inject_source_columns(
    batch: pa.RecordBatch, plm_id: str, k_context: int
) -> pa.RecordBatch:
    """Return a new batch with ``plm_id`` and ``k_context`` columns appended."""
    n = batch.num_rows
    plm_col = pa.array([plm_id] * n, type=pa.string())
    k_col = pa.array(np.full(n, k_context, dtype=np.int32), type=pa.int32())

    existing_names = batch.schema.names
    if "plm_id" in existing_names or "k_context" in existing_names:
        # Already injected (e.g. second pass); return as-is.
        return batch

    new_schema = batch.schema.append(pa.field("plm_id", pa.string()))
    new_schema = new_schema.append(pa.field("k_context", pa.int32()))
    arrays = list(batch.columns) + [plm_col, k_col]
    return pa.RecordBatch.from_arrays(arrays, schema=new_schema)


def _plm_id_from_name(dataset_name: str) -> str:
    """Extract PLM identifier from a dataset name like ``bench-v1-K5-v226-lineage-prot_t5``.

    Falls back to the full name if no known PLM token is found.
    The parser strips the ``bench-v1-K{N}-v{M}-lineage-`` prefix and returns
    the remainder as the PLM id.
    """
    # Pattern: bench-v1-K<k>-v<ver>-lineage[-<plm>]
    m = re.match(r"bench-v\d+-K\d+-v\d+-lineage-(.+)$", dataset_name)
    if m:
        return m.group(1)
    # Fallback: return the full name (callers can pass explicit plm_id)
    return dataset_name


def _compute_pool_sha(sources: list[ManifestSource]) -> str:
    """12-hex sha256 digest of the sorted manifest URI list."""
    uris = sorted(str(s.manifest_path) for s in sources)
    blob = "|".join(uris).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Smoke-pool helper (used by tests and the CLI)
# ---------------------------------------------------------------------------


def build_smoke_pool(
    datasets_root: Path | str,
    *,
    plm_ids: list[str] | None = None,
    k_values: list[int] | None = None,
) -> MultiManifestSpec:
    """Build a :class:`MultiManifestSpec` from the standard v226-lineage layout.

    Scans ``datasets_root`` for directories matching
    ``bench-v1-K{k}-v226-lineage-{plm}`` and builds the pool.

    Args:
        datasets_root: directory containing the dataset subdirectories.
        plm_ids:       PLM subset to include.  ``None`` = include all found.
        k_values:      K subset to include.  ``None`` = include all found.
    """
    root = Path(datasets_root)
    manifests: list[Path] = []
    pattern = re.compile(r"bench-v1-K(\d+)-v226-lineage-(.+)$")
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        m = pattern.match(d.name)
        if m is None:
            continue
        k_val = int(m.group(1))
        plm_val = m.group(2)
        if k_values is not None and k_val not in k_values:
            continue
        if plm_ids is not None and plm_val not in plm_ids:
            continue
        manifest = d / "manifest.json"
        if manifest.exists():
            manifests.append(manifest)
    if not manifests:
        raise FileNotFoundError(
            f"No v226-lineage manifests found under {root} "
            f"(plm_ids={plm_ids}, k_values={k_values})"
        )
    # Cast to list[Path | str] to satisfy mypy's list invariance.
    return MultiManifestSpec.from_manifest_paths(list(manifests))
