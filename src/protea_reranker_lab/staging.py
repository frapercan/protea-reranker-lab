"""Stream a source parquet dump into LightGBM-ready, sorted, cell-filtered shards.

The pipeline is:

1. **Pass 0** scans only ``protein_accession``, ``label``, ``snapshot_pair``
   and the categorical columns to build stable code maps per categorical
   column and a per-protein label/snapshot footprint for train/val routing.

2. **Pass 1** — re-scan with all required feature columns, encode
   categoricals, route each row to ``train`` or ``val`` (``eval`` is its own
   single-pass pipeline), and append it to the parquet writer of one of
   ``bucket_count`` files chosen by ``crc32(protein_accession) % bucket_count``.
   Same protein → same bucket, so groups never cross bucket boundaries.

3. **Pass 2** — for each bucket, read its (small) parquet, sort by
   ``protein_accession`` (preserves original row order within a protein via
   stable sort), write the sorted version. After this pass the concatenation
   of buckets in any fixed order is a valid LightGBM ranking dataset:
   protein groups are contiguous within each bucket, and no protein appears
   in more than one bucket.

Memory bound at any time: ~one batch (≈ 200k rows × n_cols × 8 B ≈ 100 MB)
plus the largest bucket during the sort pass (~250 MB for 32 buckets on the
nk-bpo cell of bench-v1-K5). No structure ever holds the whole partition.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .data import iter_batches


CAT_MISSING_CODE = -1


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


def _scan_pass0(
    parquet_path: Path,
    *,
    category: str | None,
    aspect: str | None,
    snapshot_pairs: list[str] | None,
    cat_cols: list[str],
    batch_size: int,
    collect_go_terms: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, list[str]], np.ndarray | None]:
    """Collect protein, label, snapshot_pair, cat-value vocabulary, and
    optionally go_term_id (only needed for True-Path-Rule propagation).

    Returns ``(proteins, labels, snapshot_pair, cat_codes, go_terms_or_None)``.
    """
    cols = ["protein_accession", "label", "snapshot_pair", *cat_cols]
    if collect_go_terms:
        cols.append("go_term_id")
    cols = _present_columns(parquet_path, cols)
    cat_seen: dict[str, set] = {c: set() for c in cat_cols}

    prot_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    pair_chunks: list[np.ndarray] = []
    go_chunks: list[np.ndarray] = []

    for batch in iter_batches(
        parquet_path,
        columns=cols,
        category=category,
        aspect=aspect,
        snapshot_pairs=snapshot_pairs,
        batch_size=batch_size,
    ):
        prot_chunks.append(batch.column("protein_accession").to_numpy(zero_copy_only=False))
        label_chunks.append(batch.column("label").to_numpy(zero_copy_only=False).astype(np.int8, copy=False))
        if "snapshot_pair" in batch.schema.names:
            pair_chunks.append(batch.column("snapshot_pair").to_numpy(zero_copy_only=False))
        if collect_go_terms and "go_term_id" in batch.schema.names:
            go_chunks.append(batch.column("go_term_id").to_numpy(zero_copy_only=False))
        for c in cat_cols:
            if c not in batch.schema.names:
                continue
            uniq = pc.unique(batch.column(c))
            for v in uniq.to_pylist():
                if v is not None:
                    cat_seen[c].add(v)

    proteins = np.concatenate(prot_chunks) if prot_chunks else np.empty(0, dtype=object)
    labels = np.concatenate(label_chunks) if label_chunks else np.empty(0, dtype=np.int8)
    pairs = (np.concatenate(pair_chunks) if pair_chunks
             else np.empty(len(proteins), dtype=object))
    go_terms = np.concatenate(go_chunks) if go_chunks else None

    cat_codes = {c: sorted(cat_seen[c]) for c in cat_cols}
    return proteins, labels, pairs, cat_codes, go_terms


def load_parent_map(path: Path | str) -> dict[str, frozenset[str]]:
    """Read a parent_map.json (as produced by ``scripts/export_parent_map.py``)
    and return ``{child_go: frozenset(direct parents)}``."""
    with open(path) as f:
        payload = json.load(f)
    if "parents" in payload:
        raw = payload["parents"]
    else:
        raw = payload
    return {child: frozenset(parents) for child, parents in raw.items()}


def propagate_labels_to_ancestors(
    proteins: np.ndarray,
    go_terms: np.ndarray,
    labels: np.ndarray,
    parent_map: dict[str, frozenset[str]],
) -> tuple[np.ndarray, int]:
    """True-Path-Rule label propagation.

    For each protein, expand its set of leaf positives with every is_a /
    part_of ancestor; any row whose ``(protein, go_term)`` is in the closure
    is then re-labelled as positive.

    Returns ``(new_labels, n_promoted)`` where ``n_promoted`` counts rows
    flipped from 0 to 1.
    """
    if labels.size == 0:
        return labels.copy(), 0

    ancestor_cache: dict[str, frozenset[str]] = {}

    def _ancestors(go_id: str) -> frozenset[str]:
        cached = ancestor_cache.get(go_id)
        if cached is not None:
            return cached
        seen: set[str] = set()
        stack = [go_id]
        while stack:
            cur = stack.pop()
            for p in parent_map.get(cur, ()):
                if p not in seen:
                    seen.add(p)
                    stack.append(p)
        result = frozenset(seen)
        ancestor_cache[go_id] = result
        return result

    closure: dict[str, set[str]] = {}
    pos_idx = np.flatnonzero(labels > 0)
    for i in pos_idx:
        prot = proteins[i]
        gid = go_terms[i]
        bucket = closure.setdefault(prot, set())
        bucket.add(gid)
        bucket.update(_ancestors(gid))

    new_labels = labels.copy()
    n_promoted = 0
    for i in range(len(labels)):
        if new_labels[i]:
            continue
        cl = closure.get(proteins[i])
        if cl is not None and go_terms[i] in cl:
            new_labels[i] = 1
            n_promoted += 1
    return new_labels, n_promoted


def _decide_split(
    *,
    proteins: np.ndarray,
    labels: np.ndarray,
    pairs: np.ndarray,
    val_strategy: str,
    val_fraction: float,
    val_holdout_snapshot: str | None,
    neg_pos_ratio: float | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, set]:
    """Return ``(keep_mask, val_mask, val_protein_set)`` over the row index.

    ``keep_mask`` excludes rows dropped by negative downsampling; ``val_mask``
    is True for rows that should land in the validation split. Both arrays
    are aligned with the order of rows produced by Pass 0 (which is the same
    deterministic order Pass 1 will produce given the same filter and
    ``batch_size``).
    """
    n = len(proteins)
    rng = np.random.default_rng(seed)

    if neg_pos_ratio is not None and neg_pos_ratio > 0:
        pos_mask = labels > 0
        neg_idx = np.flatnonzero(~pos_mask)
        target_neg = int(pos_mask.sum() * neg_pos_ratio)
        if target_neg < len(neg_idx):
            keep = np.zeros(n, dtype=bool)
            keep[pos_mask] = True
            chosen = rng.choice(neg_idx, size=target_neg, replace=False)
            keep[chosen] = True
        else:
            keep = np.ones(n, dtype=bool)
    else:
        keep = np.ones(n, dtype=bool)

    val_mask = np.zeros(n, dtype=bool)
    val_proteins: set = set()
    if val_strategy == "protein_group":
        unique_prots = np.unique(proteins)
        rng.shuffle(unique_prots)
        n_val = int(len(unique_prots) * val_fraction)
        val_proteins = set(unique_prots[:n_val].tolist())
        if val_proteins:
            val_mask = np.fromiter(
                (p in val_proteins for p in proteins), count=n, dtype=bool
            )
    elif val_strategy == "temporal":
        if not val_holdout_snapshot:
            raise ValueError("val_strategy=temporal requires val_holdout_snapshot")
        val_mask = pairs == val_holdout_snapshot
    elif val_strategy != "none":
        raise ValueError(f"unknown val_strategy: {val_strategy}")

    return keep, val_mask, val_proteins


# LightGBM hardcodes ``kMaxPosition = 10000`` per query in LambdaRank. Cap
# oversized groups in staging so the booster's ranking objective never trips
# the limit. Highly annotated proteins (e.g. TGF-β1 / P01137) blow past this
# when 12 snapshot pairs × 5 KNN neighbours × dozens of GO candidates pile up.
LGBM_LAMBDARANK_MAX_GROUP = 9999


def _cap_oversized_groups(
    *,
    proteins: np.ndarray,
    labels: np.ndarray,
    keep_mask: np.ndarray,
    val_mask: np.ndarray,
    max_group_size: int,
    seed: int,
) -> tuple[np.ndarray, int]:
    """Drop random non-positive rows from any (protein, side) bucket that
    exceeds ``max_group_size``. Returns ``(keep_mask, n_dropped)``."""
    if max_group_size <= 0 or len(proteins) == 0:
        return keep_mask, 0
    rng = np.random.default_rng(seed + 1)
    side = val_mask.astype(np.int8)
    order = np.lexsort((side, proteins))
    sorted_proteins = proteins[order]
    sorted_side = side[order]
    boundary = (sorted_proteins[1:] != sorted_proteins[:-1]) | (sorted_side[1:] != sorted_side[:-1])
    edges = np.concatenate(([0], np.flatnonzero(boundary) + 1, [len(order)]))

    n_dropped = 0
    for i in range(len(edges) - 1):
        rows = order[edges[i]:edges[i + 1]]
        kept_subset = rows[keep_mask[rows]]
        if len(kept_subset) <= max_group_size:
            continue
        is_pos = labels[kept_subset] > 0
        pos = kept_subset[is_pos]
        neg = kept_subset[~is_pos]
        if len(pos) >= max_group_size:
            keep = rng.choice(pos, size=max_group_size, replace=False)
        else:
            need = max_group_size - len(pos)
            sampled_neg = rng.choice(neg, size=need, replace=False) if need < len(neg) else neg
            keep = np.concatenate([pos, sampled_neg])
        drop = np.setdiff1d(kept_subset, keep, assume_unique=False)
        keep_mask[drop] = False
        n_dropped += len(drop)
    return keep_mask, n_dropped


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
) -> pa.Schema:
    fields = [pa.field("protein_accession", pa.string()),
              pa.field("label", pa.int8())]
    if carry_go_terms:
        fields.append(pa.field("go_term_id", pa.string()))
    cat_set = set(categorical_cols)
    for c in feature_cols:
        if c in cat_set:
            fields.append(pa.field(c, pa.int32()))
        else:
            fields.append(pa.field(c, pa.float32()))
    return pa.schema(fields)


def _pass1_route_and_write(
    *,
    parquet_path: Path,
    category: str | None,
    aspect: str | None,
    snapshot_pairs: list[str] | None,
    feature_cols: list[str],
    categorical_cols: list[str],
    cat_codes: dict[str, list[str]],
    keep_mask: np.ndarray,
    val_mask: np.ndarray,
    bucket_count: int,
    train_writers: list[pq.ParquetWriter],
    val_writers: list[pq.ParquetWriter] | None,
    schema: pa.Schema,
    batch_size: int,
    override_labels: np.ndarray | None = None,
    carry_go_terms: bool = False,
) -> tuple[int, int]:
    """Stream filter + cat-encode + bucket-route. Return ``(n_train, n_val)``.

    If ``override_labels`` is provided (length matching the cell-filtered row
    count produced by Pass 0), it replaces the source's ``label`` column —
    used for True-Path-Rule label propagation. ``carry_go_terms`` adds the
    ``go_term_id`` column so IA sample weighting can map each row to IA(go).
    """
    cols = ["protein_accession", "label", *feature_cols]
    if carry_go_terms:
        cols.append("go_term_id")
    cols = _present_columns(parquet_path, cols)
    code_maps = {c: {v: i for i, v in enumerate(cat_codes[c])} for c in categorical_cols}
    cat_set = set(categorical_cols)
    n_train = 0
    n_val = 0
    cursor = 0
    for batch in iter_batches(
        parquet_path,
        columns=cols,
        category=category,
        aspect=aspect,
        snapshot_pairs=snapshot_pairs,
        batch_size=batch_size,
    ):
        m = batch.num_rows
        local_keep = keep_mask[cursor:cursor + m]
        local_val = val_mask[cursor:cursor + m]
        if override_labels is not None:
            labels = override_labels[cursor:cursor + m].astype(np.int8, copy=False)
        else:
            labels = batch.column("label").to_numpy(zero_copy_only=False).astype(np.int8, copy=False)
        cursor += m

        if not local_keep.any():
            continue

        accessions = batch.column("protein_accession")
        go_terms = (
            batch.column("go_term_id")
            if carry_go_terms and "go_term_id" in batch.schema.names
            else None
        )

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

        bucket_idx = _bucket_array(accessions, bucket_count)

        for is_val, writers in (
            (False, train_writers),
            (True, val_writers if val_writers is not None else None),
        ):
            if writers is None:
                continue
            sel = local_keep & (local_val if is_val else ~local_val)
            if not sel.any():
                continue
            for b in np.unique(bucket_idx[sel]):
                row_mask = sel & (bucket_idx == b)
                if not row_mask.any():
                    continue
                idx = np.flatnonzero(row_mask)
                idx_arr = pa.array(idx)
                cols_data: list[pa.Array] = [
                    accessions.take(idx_arr),
                    pa.array(labels[idx], type=pa.int8()),
                ]
                if carry_go_terms:
                    cols_data.append(
                        go_terms.take(idx_arr) if go_terms is not None
                        else pa.array([None] * len(idx), type=pa.string())
                    )
                for c in feature_cols:
                    a = feat_arrays[c][idx]
                    if c in cat_set:
                        cols_data.append(pa.array(a, type=pa.int32()))
                    else:
                        cols_data.append(pa.array(a, type=pa.float32()))
                table = pa.Table.from_arrays(cols_data, schema=schema)
                writers[int(b)].write_table(table)
                if is_val:
                    n_val += len(idx)
                else:
                    n_train += len(idx)
    return n_train, n_val


_RESERVED_BUCKET_COLS = ("protein_accession", "label", "go_term_id")


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


def stage_eval_only(
    *,
    source_eval_parquet: Path,
    cell: tuple[str, str],
    feature_cols: list[str],
    categorical_cols: list[str],
    out_dir: Path,
    eval_snapshot_pair: str | None = None,
    bucket_count: int = 32,
    batch_size: int = 200_000,
) -> StagedSplit:
    """Stage only the eval split for a cell — no train/val machinery.

    Intended for downstream consumers (bootstrap CI, cafaeval re-validation)
    that need eval features in the same row order the trainer's
    ``predictions.parquet`` was written in. Categorical code maps are derived
    from the eval rows themselves (no train cross-reference); for column-set
    consistency only NUMERIC_FEATURES need exact alignment, which they get.
    """
    cat, asp = cell[0].lower(), cell[1].lower()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    schema = _build_pass1_schema(feature_cols, categorical_cols)

    cat_seen: dict[str, set] = {c: set() for c in categorical_cols}
    if categorical_cols:
        for batch in iter_batches(
            source_eval_parquet,
            columns=["protein_accession", *categorical_cols],
            category=cat, aspect=asp,
            snapshot_pair=eval_snapshot_pair,
            batch_size=batch_size,
        ):
            for c in categorical_cols:
                if c not in batch.schema.names:
                    continue
                for v in pc.unique(batch.column(c)).to_pylist():
                    if v is not None:
                        cat_seen[c].add(v)
    cat_codes = {c: sorted(cat_seen[c]) for c in categorical_cols}

    with tempfile.TemporaryDirectory(prefix="stage_eval_", dir=out_dir) as tmp:
        tmp_dir = Path(tmp)
        writers = _open_bucket_writers(tmp_dir, schema, bucket_count, "eval")
        _stream_eval(
            source_eval_parquet=source_eval_parquet,
            category=cat, aspect=asp,
            snapshot_pair=eval_snapshot_pair,
            feature_cols=feature_cols,
            categorical_cols=categorical_cols,
            cat_codes=cat_codes,
            bucket_count=bucket_count,
            writers=writers,
            schema=schema,
            batch_size=batch_size,
        )
        split = _materialise_split(
            bucket_writers=writers,
            bucket_dir=tmp_dir,
            bucket_count=bucket_count,
            out_dir=out_dir,
            name="eval",
        )
    if split is None:
        raise RuntimeError(f"eval-only staging produced no rows for cell {cat}-{asp}")
    return split


def stage_for_training(
    *,
    source_train_parquet: Path,
    source_eval_parquet: Path,
    cell: tuple[str, str],
    feature_cols: list[str],
    categorical_cols: list[str],
    out_dir: Path,
    val_strategy: str,
    val_fraction: float,
    val_holdout_snapshot: str | None,
    neg_pos_ratio: float | None,
    seed: int,
    train_snapshot_pairs: list[str] | None = None,
    eval_snapshot_pair: str | None = None,
    bucket_count: int = 32,
    batch_size: int = 200_000,
    parent_map_path: Path | str | None = None,
    carry_go_terms: bool = False,
) -> StageResult:
    cat, asp = cell[0].lower(), cell[1].lower()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # The eval split never carries go_term_id (IA weights apply to training
    # only), so train/val and eval use distinct schemas.
    eval_schema = _build_pass1_schema(feature_cols, categorical_cols)
    train_schema = _build_pass1_schema(
        feature_cols, categorical_cols, carry_go_terms=carry_go_terms
    )
    numeric_cols = [c for c in feature_cols if c not in set(categorical_cols)]

    propagate = parent_map_path is not None
    proteins0, labels0, pairs0, cat_codes, go_terms0 = _scan_pass0(
        source_train_parquet,
        category=cat, aspect=asp,
        snapshot_pairs=train_snapshot_pairs,
        cat_cols=categorical_cols,
        batch_size=batch_size,
        collect_go_terms=propagate,
    )

    propagation_stats: dict[str, int] = {}
    if propagate:
        if go_terms0 is None:
            raise RuntimeError("propagation requested but go_term_id column is missing")
        parent_map = load_parent_map(parent_map_path)
        n_pos_before = int((labels0 > 0).sum())
        labels0, n_promoted = propagate_labels_to_ancestors(
            proteins0, go_terms0, labels0, parent_map,
        )
        propagation_stats = {
            "positives_before": n_pos_before,
            "positives_after": int((labels0 > 0).sum()),
            "rows_promoted": n_promoted,
        }

    keep_mask, val_mask, val_proteins = _decide_split(
        proteins=proteins0, labels=labels0, pairs=pairs0,
        val_strategy=val_strategy,
        val_fraction=val_fraction,
        val_holdout_snapshot=val_holdout_snapshot,
        neg_pos_ratio=neg_pos_ratio,
        seed=seed,
    )

    keep_mask, _ = _cap_oversized_groups(
        proteins=proteins0, labels=labels0,
        keep_mask=keep_mask, val_mask=val_mask,
        max_group_size=LGBM_LAMBDARANK_MAX_GROUP,
        seed=seed,
    )

    with tempfile.TemporaryDirectory(prefix="staging_buckets_", dir=out_dir) as tmp:
        tmp_dir = Path(tmp)
        train_writers = _open_bucket_writers(tmp_dir, train_schema, bucket_count, "train")
        val_writers = (
            _open_bucket_writers(tmp_dir, train_schema, bucket_count, "val")
            if val_strategy != "none" else None
        )

        _pass1_route_and_write(
            parquet_path=source_train_parquet,
            category=cat, aspect=asp,
            snapshot_pairs=train_snapshot_pairs,
            feature_cols=feature_cols,
            categorical_cols=categorical_cols,
            cat_codes=cat_codes,
            keep_mask=keep_mask, val_mask=val_mask,
            bucket_count=bucket_count,
            train_writers=train_writers,
            val_writers=val_writers,
            schema=train_schema,
            batch_size=batch_size,
            override_labels=labels0 if propagate else None,
            carry_go_terms=carry_go_terms,
        )

        train_split = _materialise_split(
            bucket_writers=train_writers,
            bucket_dir=tmp_dir,
            bucket_count=bucket_count,
            out_dir=out_dir / "train",
            name="train",
        )
        val_split = None
        if val_writers is not None:
            val_split = _materialise_split(
                bucket_writers=val_writers,
                bucket_dir=tmp_dir,
                bucket_count=bucket_count,
                out_dir=out_dir / "val",
                name="val",
            )

        eval_writers = _open_bucket_writers(tmp_dir, eval_schema, bucket_count, "eval")
        eval_override_labels = None
        if propagate:
            parent_map_local = load_parent_map(parent_map_path)
            ev_proteins, ev_labels, _, _, ev_go_terms = _scan_pass0(
                source_eval_parquet,
                category=cat, aspect=asp,
                snapshot_pairs=[eval_snapshot_pair] if eval_snapshot_pair else None,
                cat_cols=[],
                batch_size=batch_size,
                collect_go_terms=True,
            )
            ev_n_pos_before = int((ev_labels > 0).sum())
            ev_labels, ev_promoted = propagate_labels_to_ancestors(
                ev_proteins, ev_go_terms, ev_labels, parent_map_local,
            )
            propagation_stats.update({
                "eval_positives_before": ev_n_pos_before,
                "eval_positives_after": int((ev_labels > 0).sum()),
                "eval_rows_promoted": ev_promoted,
            })
            eval_override_labels = ev_labels
        _stream_eval(
            source_eval_parquet=source_eval_parquet,
            category=cat, aspect=asp,
            snapshot_pair=eval_snapshot_pair,
            feature_cols=feature_cols,
            categorical_cols=categorical_cols,
            cat_codes=cat_codes,
            bucket_count=bucket_count,
            writers=eval_writers,
            schema=eval_schema,
            batch_size=batch_size,
            override_labels=eval_override_labels,
        )
        eval_split = _materialise_split(
            bucket_writers=eval_writers,
            bucket_dir=tmp_dir,
            bucket_count=bucket_count,
            out_dir=out_dir / "eval",
            name="eval",
        )

    if eval_split is None:
        raise RuntimeError(
            f"eval split is empty for cell {cat}-{asp} "
            f"(snapshot_pair={eval_snapshot_pair})"
        )
    if train_split is None:
        raise RuntimeError(f"train split is empty for cell {cat}-{asp}")

    result = StageResult(
        train=train_split,
        val=val_split,
        eval=eval_split,
        cat_codes=cat_codes,
        feature_cols=feature_cols,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        n_proteins_train=int(train_split.n_groups),
        n_proteins_val=int(val_split.n_groups) if val_split else 0,
        n_proteins_eval=int(eval_split.n_groups),
        out_dir=out_dir,
    )
    if propagation_stats:
        (out_dir / "propagation.json").write_text(
            json.dumps({"parent_map": str(parent_map_path), **propagation_stats}, indent=2)
        )
    result.to_json(out_dir / "staging.json")
    return result


def _stream_eval(
    *,
    source_eval_parquet: Path,
    category: str,
    aspect: str,
    snapshot_pair: str | None,
    feature_cols: list[str],
    categorical_cols: list[str],
    cat_codes: dict[str, list[str]],
    bucket_count: int,
    writers: list[pq.ParquetWriter],
    schema: pa.Schema,
    batch_size: int,
    override_labels: np.ndarray | None = None,
) -> None:
    cols = ["protein_accession", "label", *feature_cols]
    cols = _present_columns(source_eval_parquet, cols)
    code_maps = {c: {v: i for i, v in enumerate(cat_codes[c])} for c in categorical_cols}
    cat_set = set(categorical_cols)
    cursor = 0
    for batch in iter_batches(
        source_eval_parquet,
        columns=cols,
        category=category, aspect=aspect,
        snapshot_pair=snapshot_pair,
        batch_size=batch_size,
    ):
        m = batch.num_rows
        if m == 0:
            continue
        accessions = batch.column("protein_accession")
        if override_labels is not None:
            labels = override_labels[cursor:cursor + m].astype(np.int8, copy=False)
            cursor += m
        else:
            labels = batch.column("label").to_numpy(zero_copy_only=False).astype(np.int8, copy=False)
        feat: dict[str, np.ndarray] = {}
        for c in feature_cols:
            if c not in batch.schema.names:
                feat[c] = (
                    np.full(m, CAT_MISSING_CODE, dtype=np.int32)
                    if c in cat_set
                    else np.full(m, np.nan, dtype=np.float32)
                )
                continue
            col = batch.column(c)
            if c in cat_set:
                feat[c] = _encode_cat_batch(col, code_maps[c])
            else:
                arr = col.to_numpy(zero_copy_only=False)
                if arr.dtype != np.float32:
                    arr = arr.astype(np.float32, copy=False)
                feat[c] = arr
        bucket_idx = _bucket_array(accessions, bucket_count)
        for b in np.unique(bucket_idx):
            row_mask = bucket_idx == b
            idx = np.flatnonzero(row_mask)
            cols_data: list[pa.Array] = [
                accessions.take(pa.array(idx)),
                pa.array(labels[idx], type=pa.int8()),
            ]
            for c in feature_cols:
                a = feat[c][idx]
                if c in cat_set:
                    cols_data.append(pa.array(a, type=pa.int32()))
                else:
                    cols_data.append(pa.array(a, type=pa.float32()))
            table = pa.Table.from_arrays(cols_data, schema=schema)
            writers[int(b)].write_table(table)
