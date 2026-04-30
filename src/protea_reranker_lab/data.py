"""Streaming primitives for parquet feature dumps.

A dataset directory is expected to contain:

    train.parquet   — union of all multisnap delta pairs
    eval.parquet    — held-out evaluation set (one row per (protein, go) candidate)
    manifest.json   — metadata (K, embedding, deltas, schema sha)

This module provides only **streaming** primitives — every operation is
batch-oriented via PyArrow, no full-file pandas materialisation. The heavy
lifting (filter + sort + cat-encode + train/val split) lives in
:mod:`protea_reranker_lab.staging`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from .schemas import RESERVED_COLUMNS, ManifestV1


RESERVED_COLS = list(RESERVED_COLUMNS)
DatasetManifest = ManifestV1


def parquet_columns(path: str | Path) -> list[str]:
    return pq.read_schema(str(path)).names


def parquet_row_count(path: str | Path) -> int:
    return pq.ParquetFile(str(path)).metadata.num_rows


def _expr_for(category: str | None, aspect: str | None,
              snapshot_pair: str | None,
              snapshot_pairs: list[str] | None = None):
    expr = None
    if category is not None:
        e = ds.field("category") == category.lower()
        expr = e if expr is None else expr & e
    if aspect is not None:
        e = ds.field("aspect") == aspect.lower()
        expr = e if expr is None else expr & e
    if snapshot_pair is not None:
        e = ds.field("snapshot_pair") == snapshot_pair
        expr = e if expr is None else expr & e
    if snapshot_pairs:
        e = ds.field("snapshot_pair").isin(list(snapshot_pairs))
        expr = e if expr is None else expr & e
    return expr


def iter_batches(
    path: str | Path,
    *,
    columns: list[str] | None = None,
    category: str | None = None,
    aspect: str | None = None,
    snapshot_pair: str | None = None,
    snapshot_pairs: list[str] | None = None,
    batch_size: int = 200_000,
) -> Iterator[pa.RecordBatch]:
    """Yield filtered, column-pruned record batches from a parquet file.

    All filtering happens via PyArrow's pushdown — no full-file load.
    """
    dataset = ds.dataset(str(path), format="parquet")
    expr = _expr_for(category, aspect, snapshot_pair, snapshot_pairs)
    use_cols = columns
    if use_cols is not None:
        present = set(dataset.schema.names)
        use_cols = [c for c in use_cols if c in present]
    scanner = dataset.scanner(
        columns=use_cols,
        filter=expr,
        batch_size=batch_size,
        use_threads=False,
    )
    yield from scanner.to_batches()


def count_filtered_rows(
    path: str | Path,
    *,
    category: str | None = None,
    aspect: str | None = None,
    snapshot_pair: str | None = None,
    snapshot_pairs: list[str] | None = None,
) -> int:
    """Streaming count of rows matching the given filter."""
    dataset = ds.dataset(str(path), format="parquet")
    expr = _expr_for(category, aspect, snapshot_pair, snapshot_pairs)
    return dataset.count_rows(filter=expr)


def expected_columns() -> list[str]:
    from .reranker import ALL_FEATURES
    return RESERVED_COLS + ALL_FEATURES
