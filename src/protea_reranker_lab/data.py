"""Load frozen feature parquet datasets.

A dataset directory is expected to contain:

    train.parquet   — union of all multisnap delta pairs
    eval.parquet    — held-out evaluation set (one row per (protein, go) candidate)
    manifest.json   — metadata (K, embedding, deltas, schema sha)

Parquet columns (reserved + 52 features):

    protein_accession, go_term_id, label, category, aspect, snapshot_pair,
    plus all of NUMERIC_FEATURES + CATEGORICAL_FEATURES from reranker.py.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .reranker import ALL_FEATURES
from .schemas import RESERVED_COLUMNS, ManifestV1


RESERVED_COLS = list(RESERVED_COLUMNS)

DatasetManifest = ManifestV1


def load_partition(
    parquet_path: str | Path,
    *,
    category: str | None = None,
    aspect: str | None = None,
    snapshot_pair: str | None = None,
    columns: list[str] | None = None,
    sort_by_group: bool = True,
) -> pd.DataFrame:
    """Load a filtered slice of a parquet dump.

    ``category`` in {nk, lk, pk}; ``aspect`` in {bpo, mfo, cco}.  ``snapshot_pair``
    filters the training dump to one delta (e.g. ``"v160-v165"``).  Sorting by
    ``protein_accession`` is required for LightGBM LambdaRank group computation.
    """
    filters = []
    if category is not None:
        filters.append(("category", "=", category.lower()))
    if aspect is not None:
        filters.append(("aspect", "=", aspect.lower()))
    if snapshot_pair is not None:
        filters.append(("snapshot_pair", "=", snapshot_pair))
    df = pd.read_parquet(
        parquet_path,
        columns=columns,
        filters=filters if filters else None,
    )
    if sort_by_group and "protein_accession" in df.columns:
        df = df.sort_values("protein_accession", kind="stable").reset_index(drop=True)
    return df


def split_train_val(
    df: pd.DataFrame, val_fraction: float = 0.2, *, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Group-aware random split: whole proteins go to one side.

    Avoids the "same protein in train and val" leak that saturates val NDCG.
    Re-sorts each side by protein_accession to keep LightGBM group alignment.
    """
    if val_fraction <= 0:
        return df, df.iloc[0:0]
    proteins = df["protein_accession"].unique()
    import numpy as np
    rng = np.random.default_rng(seed)
    rng.shuffle(proteins)
    n_val = int(len(proteins) * val_fraction)
    val_prots = set(proteins[:n_val])
    mask_val = df["protein_accession"].isin(val_prots)
    df_val = df[mask_val].sort_values("protein_accession", kind="stable").reset_index(drop=True)
    df_tr = df[~mask_val].sort_values("protein_accession", kind="stable").reset_index(drop=True)
    return df_tr, df_val


def split_train_val_temporal(
    df: pd.DataFrame, holdout_snapshot_pair: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use one snapshot pair as val, rest as train. Mirrors the temporal holdout
    structure of the eval set — way harder than random, better for generalisation."""
    mask = df["snapshot_pair"] == holdout_snapshot_pair
    df_val = df[mask].sort_values("protein_accession", kind="stable").reset_index(drop=True)
    df_tr = df[~mask].sort_values("protein_accession", kind="stable").reset_index(drop=True)
    return df_tr, df_val


def expected_columns() -> list[str]:
    return RESERVED_COLS + ALL_FEATURES
