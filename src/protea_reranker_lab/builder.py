"""Derived-dataset builder: PROTEA dump → experiment-ready parquet shards.

The primary dump (raw 52-feature export) is produced by PROTEA's
``train_reranker_auto --dump-only``. This module reshapes that dump per a
:class:`DatasetSpec` — feature-family subset, snapshot filter, reproducible
hash — and writes a fresh ``train/eval.parquet`` + :class:`ManifestV1` into
``out_dir``. The builder never touches the PROTEA DB.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .schemas import (
    DatasetSpec,
    ManifestV1,
    compute_schema_sha,
    required_columns,
)


def build_dataset(spec: DatasetSpec, out_dir: str | Path) -> ManifestV1:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    src_manifest_path = Path(spec.source_manifest)
    source = ManifestV1.load(src_manifest_path)
    src_dir = src_manifest_path.parent

    cols = required_columns(spec.enabled_feature_families, spec.drop_features)

    train_pairs = spec.train_snapshot_pairs or source.train_snapshot_pairs
    eval_pair = spec.eval_snapshot_pair or source.eval_snapshot_pair

    df_train = _read_parquet(src_dir / "train.parquet", cols, snapshot_pairs=train_pairs)
    df_eval = _read_parquet(src_dir / "eval.parquet", cols, snapshot_pairs=[eval_pair])

    _sort_by_group(df_train)
    _sort_by_group(df_eval)

    df_train.to_parquet(out_dir / "train.parquet", index=False)
    df_eval.to_parquet(out_dir / "eval.parquet", index=False)

    manifest = ManifestV1(
        name=spec.name,
        k=source.k,
        embedding_config_id=source.embedding_config_id,
        ontology_snapshot_id=source.ontology_snapshot_id,
        annotation_source=source.annotation_source,
        train_snapshot_pairs=train_pairs,
        eval_snapshot_pair=eval_pair,
        schema_sha=compute_schema_sha(cols),
        n_train_rows=len(df_train),
        n_eval_rows=len(df_eval),
        format=spec.format,
        spec_hash=spec.hash(),
        parent_schema_sha=source.schema_sha,
        feature_families=spec.enabled_feature_families,
    )
    manifest.dump(out_dir / "manifest.json")
    return manifest


def _read_parquet(path: Path, columns: list[str],
                  *, snapshot_pairs: list[str] | None) -> pd.DataFrame:
    filters = None
    if snapshot_pairs:
        filters = [("snapshot_pair", "in", snapshot_pairs)]
    present = _available_columns(path)
    keep = [c for c in columns if c in present]
    return pd.read_parquet(path, columns=keep, filters=filters)


def _available_columns(path: Path) -> set[str]:
    import pyarrow.parquet as pq
    return set(pq.read_schema(path).names)


def _sort_by_group(df: pd.DataFrame) -> None:
    if "protein_accession" in df.columns:
        df.sort_values("protein_accession", kind="stable", inplace=True)
        df.reset_index(drop=True, inplace=True)
