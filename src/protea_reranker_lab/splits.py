"""Staged-split dataclasses and the bucket-sort / materialisation layer.

A :class:`StagedSplit` is the on-disk handle for one split (train/val/eval):
sorted bucket parquets plus row-aligned ``labels``/``groups``/``proteins``
(and optionally ``go_terms``) npy arrays. :func:`_materialise_split` turns a
set of unsorted bucket writers into that handle.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

_RESERVED_BUCKET_COLS = ("protein_accession", "label", "go_term_id")


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


def _sort_bucket(
    path: Path,
) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Read one bucket parquet, sort by protein_accession, write back.

    Returns ``(n_rows, n_groups, group_sizes, labels, proteins_per_group,
    go_terms_or_None)``. ``go_terms`` is row-aligned with ``labels`` and is
    ``None`` unless the bucket carried the ``go_term_id`` column.
    """
    table = pq.read_table(str(path))
    if table.num_rows == 0:
        path.unlink(missing_ok=True)
        return (
            0, 0, np.empty(0, np.int32), np.empty(0, np.int8),
            np.empty(0, dtype=object), None,
        )
    indices = pc.sort_indices(table, sort_keys=[("protein_accession", "ascending")])
    sorted_table = table.take(indices)
    feature_cols = [c for c in sorted_table.column_names if c not in _RESERVED_BUCKET_COLS]
    feat_table = sorted_table.select(feature_cols)
    pq.write_table(feat_table, str(path), compression="zstd")

    proteins = sorted_table.column("protein_accession").to_numpy(zero_copy_only=False)
    labels = sorted_table.column("label").to_numpy(zero_copy_only=False).astype(np.int8, copy=False)
    go_terms = None
    if "go_term_id" in sorted_table.column_names:
        go_terms = sorted_table.column("go_term_id").to_numpy(zero_copy_only=False)
    edges = np.flatnonzero(np.concatenate(([True], proteins[1:] != proteins[:-1])))
    group_sizes = np.diff(np.concatenate((edges, [len(proteins)]))).astype(np.int32)
    proteins_per_group = proteins[edges]
    return len(proteins), len(group_sizes), group_sizes, labels, proteins_per_group, go_terms


def _materialise_split(
    *,
    bucket_writers: list[pq.ParquetWriter],
    bucket_dir: Path,
    bucket_count: int,
    out_dir: Path,
    name: str,
) -> StagedSplit | None:
    from .bucket_io import _close_writers

    _close_writers(bucket_writers)
    out_dir.mkdir(parents=True, exist_ok=True)

    bucket_paths_final: list[Path] = []
    all_groups: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_proteins: list[np.ndarray] = []
    all_go_terms: list[np.ndarray] = []
    have_go_terms = True

    for b in range(bucket_count):
        src = bucket_dir / f"{name}_bucket_{b:02d}.parquet"
        if not src.exists():
            continue
        n, ng, sizes, labels, proteins, go_terms = _sort_bucket(src)
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

    if not bucket_paths_final:
        return None

    go_terms = (
        np.concatenate(all_go_terms) if have_go_terms and all_go_terms else None
    )
    return _save_split_arrays(
        out_dir=out_dir,
        bucket_paths=bucket_paths_final,
        groups=np.concatenate(all_groups),
        labels=np.concatenate(all_labels),
        proteins=np.concatenate(all_proteins),
        go_terms=go_terms,
    )


def _save_split_arrays(
    *,
    out_dir: Path,
    bucket_paths: list[Path],
    groups: np.ndarray,
    labels: np.ndarray,
    proteins: np.ndarray,
    go_terms: np.ndarray | None,
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

    return StagedSplit(
        bucket_paths=bucket_paths,
        labels_path=labels_path,
        groups_path=groups_path,
        proteins_path=proteins_path,
        n_rows=int(labels.size),
        n_groups=int(groups.size),
        go_terms_path=go_terms_path,
    )
