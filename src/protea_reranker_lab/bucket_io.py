"""Low-level parquet/bucket primitives shared by the staging passes.

These helpers are pure (no dependency on the rest of :mod:`staging`): crc32
bucket routing, categorical encoding, schema construction, and the per-bucket
parquet writer set. Kept in their own module so the staging orchestration file
stays focused on the multi-pass flow.
"""

from __future__ import annotations

import zlib
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

CAT_MISSING_CODE = -1


def _bucket_of(accession: str, n: int) -> int:
    return zlib.crc32(accession.encode("ascii")) % n


def _bucket_array(accessions: pa.Array, n: int) -> np.ndarray:
    arr = accessions.to_numpy(zero_copy_only=False)
    out = np.empty(len(arr), dtype=np.int32)
    for i, a in enumerate(arr):
        out[i] = zlib.crc32(a.encode("ascii")) % n
    return out


def _encode_cat_batch(batch_col: pa.Array, code_map: dict[str, int]) -> np.ndarray:
    values = batch_col.to_numpy(zero_copy_only=False)
    out = np.empty(len(values), dtype=np.int32)
    for i, v in enumerate(values):
        if v is None or (isinstance(v, float) and v != v):
            out[i] = CAT_MISSING_CODE
        else:
            out[i] = code_map.get(v, CAT_MISSING_CODE)
    return out


def _present_columns(parquet_path: Path, want: Iterable[str]) -> list[str]:
    schema = pq.read_schema(str(parquet_path))
    have = set(schema.names)
    return [c for c in want if c in have]


def _open_bucket_writers(
    out_dir: Path, schema: pa.Schema, n_buckets: int, prefix: str
) -> list[pq.ParquetWriter]:
    out_dir.mkdir(parents=True, exist_ok=True)
    return [
        pq.ParquetWriter(str(out_dir / f"{prefix}_bucket_{i:02d}.parquet"), schema,
                         compression="zstd")
        for i in range(n_buckets)
    ]


def _close_writers(writers: list[pq.ParquetWriter]) -> None:
    for w in writers:
        w.close()


def _build_pass1_schema(
    feature_cols: list[str],
    categorical_cols: list[str],
    *,
    carry_go_terms: bool = False,
    carry_aspect: bool = False,
    carry_snapshot_pair: bool = False,
) -> pa.Schema:
    """Build the PyArrow schema for one pass-1 (or eval) bucket parquet.

    Parameters
    ----------
    feature_cols:           ordered list of feature columns (numeric + categorical).
    categorical_cols:       subset of ``feature_cols`` that are encoded as int32.
    carry_go_terms:         include ``go_term_id`` (string) for IA sample weighting.
    carry_aspect:           include ``aspect`` (string) for aspect-conditioned
                            LambdaRank grouping (F-RERANK-UNIVERSAL.3).
    carry_snapshot_pair:    include ``snapshot_pair`` (string) for the extended
                            group key (snapshot_pair, protein, aspect, plm_id)
                            required by F-RERANK-UNIVERSAL.5d.
    """
    fields = [pa.field("protein_accession", pa.string()),
              pa.field("label", pa.int8())]
    if carry_go_terms:
        fields.append(pa.field("go_term_id", pa.string()))
    if carry_aspect:
        fields.append(pa.field("aspect", pa.string()))
    if carry_snapshot_pair:
        fields.append(pa.field("snapshot_pair", pa.string()))
    cat_set = set(categorical_cols)
    for c in feature_cols:
        if c in cat_set:
            fields.append(pa.field(c, pa.int32()))
        else:
            fields.append(pa.field(c, pa.float32()))
    return pa.schema(fields)


def _encode_feature_arrays(
    batch: pa.RecordBatch,
    feature_cols: list[str],
    cat_set: set[str],
    code_maps: dict[str, dict[str, int]],
) -> dict[str, np.ndarray]:
    """Cat-encode / float-cast every feature column of a batch (missing→NaN)."""
    m = batch.num_rows
    feat_arrays: dict[str, np.ndarray] = {}
    for c in feature_cols:
        if c not in batch.schema.names:
            feat_arrays[c] = (
                np.full(m, CAT_MISSING_CODE, dtype=np.int32)
                if c in cat_set
                else np.full(m, np.nan, dtype=np.float32)
            )
            continue
        col = batch.column(c)
        if c in cat_set:
            feat_arrays[c] = _encode_cat_batch(col, code_maps[c])
        else:
            arr = col.to_numpy(zero_copy_only=False)
            if arr.dtype != np.float32:
                arr = arr.astype(np.float32, copy=False)
            feat_arrays[c] = arr
    return feat_arrays
