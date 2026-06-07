"""Staged-split dataclasses and the bucket-sort / materialisation layer.

A :class:`StagedSplit` is the on-disk handle for one split (train/val/eval):
sorted bucket parquets plus row-aligned ``labels``/``groups``/``proteins``
(and optionally ``go_terms`` / ``aspects``) npy arrays.
:func:`_materialise_split` turns a set of unsorted bucket writers into that
handle.

Aspect-conditioned staging (F-RERANK-UNIVERSAL.3)
--------------------------------------------------
When the source parquet contains an ``aspect`` column and the staging pipeline
is called WITHOUT an aspect-level filter (i.e. ``aspect_conditioned=True``),
every ``(protein_accession, aspect)`` pair forms its own LambdaRank group.
The bucket router still uses ``crc32(protein_accession)`` so all rows for one
protein, across ALL aspects, land in the same bucket, preventing any group
from crossing bucket boundaries.  Within a bucket the sort key is
``(protein_accession, aspect)`` so groups are contiguous.

The ``aspects_per_group.npy`` array aligns one-to-one with ``groups.npy``
and ``proteins.npy`` and records the aspect label for each group.  Consumers
that need per-group aspect metadata (e.g. candidate recall, IA weighting,
selective-deploy CI) load this file.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

# Columns that are never treated as LightGBM features.
# ``aspect`` and ``snapshot_pair`` drive group-key computation and are
# persisted in bucket parquets so _sort_bucket can use them, then
# stripped from the write-back.
_RESERVED_BUCKET_COLS = frozenset(
    ("protein_accession", "label", "go_term_id", "aspect", "snapshot_pair")
)


@dataclass
class StagedSplit:
    bucket_paths: list[Path]
    labels_path: Path
    groups_path: Path
    proteins_path: Path
    n_rows: int
    n_groups: int
    # Row-aligned GO term id per row (same order as ``labels_path``), emitted
    # only when ``carry_go_terms=True`` so IA sample weighting can map each
    # row to IA(go). ``None`` for the historical uniform-weight path.
    go_terms_path: Path | None = None
    # One aspect string per group (same length as ``groups.npy``), emitted
    # when aspect-conditioned staging is active.  ``None`` in the legacy
    # per-cell (single-aspect) path.
    aspects_path: Path | None = None


@dataclass
class StageResult:
    train: StagedSplit
    val: StagedSplit | None
    eval: StagedSplit
    cat_codes: dict[str, list[str]]
    feature_cols: list[str]
    numeric_cols: list[str]
    categorical_cols: list[str]
    n_proteins_train: int
    n_proteins_val: int
    n_proteins_eval: int
    out_dir: Path

    def to_json(self, path: Path) -> None:
        path.write_text(json.dumps({
            "train": _split_to_dict(self.train, self.out_dir),
            "val": _split_to_dict(self.val, self.out_dir) if self.val else None,
            "eval": _split_to_dict(self.eval, self.out_dir),
            "cat_codes": self.cat_codes,
            "feature_cols": self.feature_cols,
            "numeric_cols": self.numeric_cols,
            "categorical_cols": self.categorical_cols,
            "n_proteins_train": self.n_proteins_train,
            "n_proteins_val": self.n_proteins_val,
            "n_proteins_eval": self.n_proteins_eval,
        }, indent=2, default=str))


def _split_to_dict(split: StagedSplit, root: Path) -> dict:
    return {
        "bucket_paths": [str(Path(p).relative_to(root)) for p in split.bucket_paths],
        "labels_path": str(Path(split.labels_path).relative_to(root)),
        "groups_path": str(Path(split.groups_path).relative_to(root)),
        "proteins_path": str(Path(split.proteins_path).relative_to(root)),
        "n_rows": split.n_rows,
        "n_groups": split.n_groups,
    }


def _compute_group_edges(
    proteins: np.ndarray,
    aspects: np.ndarray | None,
    snapshot_pairs: np.ndarray | None = None,
    plm_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Compute group edges, sizes, and per-group aspect labels.

    Group key selection (F-RERANK-UNIVERSAL.5d):
    - When ``snapshot_pairs`` and ``plm_ids`` are both provided AND ``aspects``
      is provided: key is ``(protein, snapshot_pair, aspect, plm_id)``.
    - When only ``aspects`` is provided (legacy 3-tuple): key is
      ``(protein, aspect)``.
    - When none are provided: key is ``protein`` alone (legacy).

    Returns ``(edges, group_sizes, aspects_per_group_or_None)``.
    """
    if aspects is not None and snapshot_pairs is not None and plm_ids is not None:
        # 4-tuple key: (protein, snapshot_pair, aspect, plm_id)
        prot_change = np.concatenate(([True], proteins[1:] != proteins[:-1]))
        snap_change = np.concatenate(([True], snapshot_pairs[1:] != snapshot_pairs[:-1]))
        asp_change = np.concatenate(([True], aspects[1:] != aspects[:-1]))
        plm_change = np.concatenate(([True], plm_ids[1:] != plm_ids[:-1]))
        edges = np.flatnonzero(prot_change | snap_change | asp_change | plm_change)
        aspects_per_group: np.ndarray | None = aspects[edges]
    elif aspects is not None:
        prot_change = np.concatenate(([True], proteins[1:] != proteins[:-1]))
        asp_change = np.concatenate(([True], aspects[1:] != aspects[:-1]))
        edges = np.flatnonzero(prot_change | asp_change)
        aspects_per_group = aspects[edges]
    else:
        edges = np.flatnonzero(
            np.concatenate(([True], proteins[1:] != proteins[:-1]))
        )
        aspects_per_group = None
    group_sizes = np.diff(
        np.concatenate((edges, [len(proteins)]))
    ).astype(np.int32)
    return edges, group_sizes, aspects_per_group


def _k_collapse_table(
    sorted_table: "pa.Table",
) -> "pa.Table":
    """Within each group, keep ONE row per go_term_id (K-collapse).

    Groups are defined by contiguous identical values of the sort key
    already applied.  ``go_term_id`` must be present; if it is not, the
    table is returned unchanged (no-op).

    Only called when ``snapshot_pair`` is present (F-RERANK-UNIVERSAL.5d
    pooled path), because that is the path where K3/K5/K10 duplicate the
    same candidate term across three physical sources.
    """
    if "go_term_id" not in sorted_table.schema.names:
        return sorted_table
    # Identify group key columns that are present.
    key_cols = ["protein_accession"]
    if "snapshot_pair" in sorted_table.schema.names:
        key_cols.append("snapshot_pair")
    if "aspect" in sorted_table.schema.names:
        key_cols.append("aspect")
    # plm_id is a feature int32 column -- include it if present.
    if "plm_id" in sorted_table.schema.names:
        key_cols.append("plm_id")

    n = sorted_table.num_rows
    if n == 0:
        return sorted_table

    # Build group boundaries from key columns.
    arrays = {c: sorted_table.column(c).to_numpy(zero_copy_only=False) for c in key_cols}
    go_arr = sorted_table.column("go_term_id").to_numpy(zero_copy_only=False)

    # First pass: for each group, find duplicate go_term_ids and keep first.
    # Group boundaries: row i starts a new group if any key col changes.
    group_start = np.zeros(n, dtype=bool)
    group_start[0] = True
    for col_arr in arrays.values():
        group_start[1:] |= (col_arr[1:] != col_arr[:-1])
    group_ids = np.cumsum(group_start) - 1  # group index per row

    # Within each group dedup on go_term_id: keep the first occurrence.
    seen: set = set()
    keep = np.ones(n, dtype=bool)
    prev_gid = -1
    for i in range(n):
        gid = int(group_ids[i])
        if gid != prev_gid:
            seen = set()
            prev_gid = gid
        key = (gid, str(go_arr[i]))
        if key in seen:
            keep[i] = False
        else:
            seen.add(key)

    if keep.all():
        return sorted_table
    keep_idx = np.flatnonzero(keep)
    return sorted_table.take(pa.array(keep_idx))


def _build_sort_keys(
    has_snapshot: bool, has_aspect: bool, has_plm: bool,
) -> list[tuple[str, str]]:
    """Return the sort-key list for a bucket based on which columns are present."""
    if has_snapshot and has_aspect:
        keys = [
            ("protein_accession", "ascending"),
            ("snapshot_pair", "ascending"),
            ("aspect", "ascending"),
        ]
        if has_plm:
            keys.append(("plm_id", "ascending"))
        return keys
    if has_aspect:
        return [("protein_accession", "ascending"), ("aspect", "ascending")]
    return [("protein_accession", "ascending")]


def _extract_sorted_arrays(
    t: "pa.Table", has_snapshot: bool, has_aspect: bool, has_plm: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Extract numpy arrays from a sorted bucket table for group computation."""
    proteins = t.column("protein_accession").to_numpy(zero_copy_only=False)
    labels = t.column("label").to_numpy(zero_copy_only=False).astype(np.int8, copy=False)
    go_terms: np.ndarray | None = (
        t.column("go_term_id").to_numpy(zero_copy_only=False)
        if "go_term_id" in t.column_names else None
    )
    aspects_arr: np.ndarray | None = (
        t.column("aspect").to_numpy(zero_copy_only=False) if has_aspect else None
    )
    snapshot_pairs_arr: np.ndarray | None = (
        t.column("snapshot_pair").to_numpy(zero_copy_only=False) if has_snapshot else None
    )
    plm_ids_arr: np.ndarray | None = (
        t.column("plm_id").to_numpy(zero_copy_only=False)
        if has_plm and has_snapshot else None
    )
    return proteins, labels, go_terms, aspects_arr, snapshot_pairs_arr, plm_ids_arr


def _sort_bucket(
    path: Path,
) -> tuple[
    int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None
]:
    """Read one bucket parquet, sort by group key, apply K-collapse, write back.

    Group key (F-RERANK-UNIVERSAL.5d): ``(protein, snapshot_pair, aspect, plm_id)``
    when all four columns are present; legacy ``(protein, aspect)`` otherwise.
    Returns ``(n_rows, n_groups, group_sizes, labels, proteins_per_group,
    go_terms_or_None, aspects_per_group_or_None)``.
    """
    table = pq.read_table(str(path))
    if table.num_rows == 0:
        path.unlink(missing_ok=True)
        return (0, 0, np.empty(0, np.int32), np.empty(0, np.int8),
                np.empty(0, dtype=object), None, None)
    names = table.schema.names
    has_snapshot = "snapshot_pair" in names
    has_aspect = "aspect" in names
    has_plm = "plm_id" in names
    sort_keys = _build_sort_keys(has_snapshot, has_aspect, has_plm)
    sorted_table = table.take(pc.sort_indices(table, sort_keys=sort_keys))
    if has_snapshot and has_aspect:
        sorted_table = _k_collapse_table(sorted_table)
    feat_cols = [c for c in sorted_table.column_names if c not in _RESERVED_BUCKET_COLS]
    pq.write_table(sorted_table.select(feat_cols), str(path), compression="zstd")
    proteins, labels, go_terms, aspects_arr, snapshot_pairs_arr, plm_ids_arr = (
        _extract_sorted_arrays(sorted_table, has_snapshot, has_aspect, has_plm)
    )
    edges, group_sizes, aspects_per_group = _compute_group_edges(
        proteins, aspects_arr, snapshot_pairs_arr, plm_ids_arr,
    )
    return (len(proteins), len(group_sizes), group_sizes, labels,
            proteins[edges], go_terms, aspects_per_group)


def _materialise_split(
    *,
    bucket_writers: list[pq.ParquetWriter],
    bucket_dir: Path,
    bucket_count: int,
    out_dir: Path,
    name: str,
) -> "StagedSplit | None":
    from .bucket_io import _close_writers

    _close_writers(bucket_writers)
    out_dir.mkdir(parents=True, exist_ok=True)

    bucket_paths_final: list[Path] = []
    all_groups: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_proteins: list[np.ndarray] = []
    all_go_terms: list[np.ndarray] = []
    all_aspects: list[np.ndarray] = []
    have_go_terms = True
    have_aspects = True

    for b in range(bucket_count):
        src = bucket_dir / f"{name}_bucket_{b:02d}.parquet"
        if not src.exists():
            continue
        n, ng, sizes, labels, proteins, go_terms, aspects = _sort_bucket(src)
        if n == 0:
            continue
        dest = out_dir / f"bucket_{b:02d}.parquet"
        shutil.move(str(src), str(dest))
        bucket_paths_final.append(dest)
        all_groups.append(sizes)
        all_labels.append(labels)
        all_proteins.append(proteins)
        if go_terms is None:
            have_go_terms = False
        else:
            all_go_terms.append(go_terms)
        if aspects is None:
            have_aspects = False
        else:
            all_aspects.append(aspects)

    if not bucket_paths_final:
        return None

    merged_go = (
        np.concatenate(all_go_terms) if have_go_terms and all_go_terms else None
    )
    merged_aspects = (
        np.concatenate(all_aspects) if have_aspects and all_aspects else None
    )
    return _save_split_arrays(
        out_dir=out_dir,
        bucket_paths=bucket_paths_final,
        groups=np.concatenate(all_groups),
        labels=np.concatenate(all_labels),
        proteins=np.concatenate(all_proteins),
        go_terms=merged_go,
        aspects=merged_aspects,
    )


def _save_split_arrays(
    *,
    out_dir: Path,
    bucket_paths: list[Path],
    groups: np.ndarray,
    labels: np.ndarray,
    proteins: np.ndarray,
    go_terms: np.ndarray | None,
    aspects: np.ndarray | None = None,
) -> StagedSplit:
    """Persist the row-aligned npy arrays and return the StagedSplit."""
    labels_path = out_dir / "labels.npy"
    groups_path = out_dir / "groups.npy"
    proteins_path = out_dir / "proteins.npy"
    np.save(labels_path, labels)
    np.save(groups_path, groups)
    np.save(proteins_path, proteins)

    go_terms_path: Path | None = None
    if go_terms is not None:
        go_terms_path = out_dir / "go_terms.npy"
        np.save(go_terms_path, go_terms)

    aspects_path: Path | None = None
    if aspects is not None:
        aspects_path = out_dir / "aspects.npy"
        np.save(aspects_path, aspects)

    return StagedSplit(
        bucket_paths=bucket_paths,
        labels_path=labels_path,
        groups_path=groups_path,
        proteins_path=proteins_path,
        n_rows=int(labels.size),
        n_groups=int(groups.size),
        go_terms_path=go_terms_path,
        aspects_path=aspects_path,
    )
