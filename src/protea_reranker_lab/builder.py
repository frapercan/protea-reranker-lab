"""Streaming derived-dataset builder: PROTEA dump → experiment-ready parquet.

The primary dump (raw 52-feature export) is produced by PROTEA's
``export_research_dataset`` operation. This module reshapes that dump per a
:class:`DatasetSpec` — feature-family subset + snapshot filter + sort by
``protein_accession`` — and writes a fresh ``train/eval.parquet`` plus a
:class:`ManifestV1` into ``out_dir``.

All passes are streamed via PyArrow record batches, with bucket-sort to keep
peak RAM bounded by the largest bucket (≈ source / bucket_count).
"""

from __future__ import annotations

import tempfile
import zlib
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .data import iter_batches
from .schemas import (
    DatasetSpec,
    ManifestV1,
    compute_schema_sha,
    required_columns,
)


_BUCKET_COUNT = 32
_BATCH_SIZE = 200_000


def build_dataset(spec: DatasetSpec, out_dir: str | Path) -> ManifestV1:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    src_manifest_path = Path(spec.source_manifest)
    source = ManifestV1.load(src_manifest_path)
    src_dir = src_manifest_path.parent

    cols = required_columns(spec.enabled_feature_families, spec.drop_features)
    train_pairs = spec.train_snapshot_pairs or source.train_snapshot_pairs
    eval_pair = spec.eval_snapshot_pair or source.eval_snapshot_pair

    n_train = _stream_reshape(
        src_dir / "train.parquet",
        out_dir / "train.parquet",
        wanted_cols=cols,
        snapshot_pairs=train_pairs,
    )
    n_eval = _stream_reshape(
        src_dir / "eval.parquet",
        out_dir / "eval.parquet",
        wanted_cols=cols,
        snapshot_pairs=[eval_pair],
    )

    manifest = ManifestV1(
        name=spec.name,
        k=source.k,
        embedding_config_id=source.embedding_config_id,
        ontology_snapshot_id=source.ontology_snapshot_id,
        annotation_source=source.annotation_source,
        train_snapshot_pairs=train_pairs,
        eval_snapshot_pair=eval_pair,
        schema_sha=compute_schema_sha(cols),
        n_train_rows=n_train,
        n_eval_rows=n_eval,
        format=spec.format,
        spec_hash=spec.hash(),
        parent_schema_sha=source.schema_sha,
        feature_families=spec.enabled_feature_families,
    )
    manifest.dump(out_dir / "manifest.json")
    return manifest


def _stream_reshape(
    src_path: Path,
    dst_path: Path,
    *,
    wanted_cols: list[str],
    snapshot_pairs: list[str] | None,
) -> int:
    src_schema = pq.read_schema(str(src_path))
    have = set(src_schema.names)
    keep = [c for c in wanted_cols if c in have]
    out_schema = pa.schema([src_schema.field(c) for c in keep])

    with tempfile.TemporaryDirectory(prefix="builder_buckets_", dir=dst_path.parent) as tmp:
        tmp_dir = Path(tmp)
        writers = [
            pq.ParquetWriter(str(tmp_dir / f"bucket_{i:02d}.parquet"), out_schema,
                             compression="zstd")
            for i in range(_BUCKET_COUNT)
        ]
        try:
            for batch in iter_batches(
                src_path,
                columns=keep,
                snapshot_pairs=snapshot_pairs,
                batch_size=_BATCH_SIZE,
            ):
                if batch.num_rows == 0:
                    continue
                accs = batch.column("protein_accession").to_numpy(zero_copy_only=False)
                buckets = np.fromiter(
                    (zlib.crc32(a.encode("ascii")) % _BUCKET_COUNT for a in accs),
                    count=len(accs), dtype=np.int32,
                )
                for b in np.unique(buckets):
                    idx = np.flatnonzero(buckets == b)
                    if idx.size == 0:
                        continue
                    sub = batch.take(pa.array(idx))
                    writers[int(b)].write_table(pa.Table.from_batches([sub], schema=out_schema))
        finally:
            for w in writers:
                w.close()

        with pq.ParquetWriter(str(dst_path), out_schema, compression="zstd") as final:
            total = 0
            for b in range(_BUCKET_COUNT):
                bp = tmp_dir / f"bucket_{b:02d}.parquet"
                if not bp.exists():
                    continue
                t = pq.read_table(str(bp))
                if t.num_rows == 0:
                    continue
                indices = pc.sort_indices(
                    t, sort_keys=[("protein_accession", "ascending")]
                )
                final.write_table(t.take(indices))
                total += t.num_rows
    return total
