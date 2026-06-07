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

Aspect-conditioned staging (F-RERANK-UNIVERSAL.3)
--------------------------------------------------
When ``StagePlan.aspect_conditioned=True`` the cell-level aspect filter is
removed so a SINGLE fit sees ALL aspects simultaneously.  ``aspect`` is kept
as a LIVE conditioning feature column (written into the bucket parquets and
available to the booster) and the LambdaRank group key switches from
per-protein to per-(protein, aspect).

Bucket routing remains ``crc32(protein_accession) % bucket_count``; this
guarantees that all rows for a protein, across ALL aspects, land in the same
bucket, so no group ever straddles two buckets.  Within each bucket the sort
key becomes ``(protein_accession, aspect)`` so groups are contiguous.

VALID / TEST window plumbing
-----------------------------
``StagePlan.train_snapshot_pairs``  filters the training set to specific
snapshot pairs (e.g. all historical pairs up to v226-v227).

``StagePlan.eval_snapshot_pair``    selects the primary VALID evaluation
window (e.g. ``"v226-v227"``) for selection / threshold-tuning.

``StagePlan.test_snapshot_pairs``   is a LIST of snapshot pairs forming the
multi-window TEST curve (e.g. ``["v227-v228", "v227-v229", "v227-v230"]``).
The TEST window is evaluate-once; design here exposes the plumbing, consumed
by downstream callers (F-RERANK-UNIVERSAL.6+).
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .bucket_io import (
    CAT_MISSING_CODE,
    _bucket_array,
    _build_pass1_schema,
    _encode_cat_batch,
    _encode_feature_arrays,
    _open_bucket_writers,
    _present_columns,
)
from .data import iter_batches
from .propagation import load_parent_map, propagate_labels_to_ancestors
from .splits import StagedSplit, StageResult, _materialise_split


@dataclass
class SourceScan:
    """Where + how to read one cell-filtered parquet stream."""

    parquet_path: Path
    category: str | None
    aspect: str | None
    snapshot_pairs: list[str] | None = None
    batch_size: int = 200_000


@dataclass
class FeatureLayout:
    """Feature/categorical column layout + code maps + the writer schema."""

    feature_cols: list[str]
    categorical_cols: list[str]
    cat_codes: dict[str, list[str]]
    schema: pa.Schema


@dataclass
class BucketRouting:
    """Bucket-routing targets for one pass1 stream."""

    bucket_count: int
    train_writers: list[pq.ParquetWriter]
    val_writers: list[pq.ParquetWriter] | None


def _scan_pass0(
    parquet_path: Path,
    *,
    category: str | None,
    aspect: str | None,
    snapshot_pairs: list[str] | None,
    cat_cols: list[str],
    batch_size: int,
    collect_go_terms: bool = False,
    collect_aspects: bool = False,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, dict[str, list[str]],
    np.ndarray | None, np.ndarray | None,
]:
    """Collect protein, label, snapshot_pair, cat-value vocabulary, and
    optionally go_term_id / aspect (for True-Path-Rule propagation /
    aspect-conditioned group key).

    Returns
    -------
    ``(proteins, labels, snapshot_pairs, cat_codes, go_terms_or_None,
    aspects_or_None)``.
    """
    cols = ["protein_accession", "label", "snapshot_pair", *cat_cols]
    if collect_go_terms:
        cols.append("go_term_id")
    if collect_aspects:
        cols.append("aspect")
    cols = _present_columns(parquet_path, cols)
    cat_seen: dict[str, set] = {c: set() for c in cat_cols}

    prot_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    pair_chunks: list[np.ndarray] = []
    go_chunks: list[np.ndarray] = []
    asp_chunks: list[np.ndarray] = []

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
        if collect_aspects and "aspect" in batch.schema.names:
            asp_chunks.append(batch.column("aspect").to_numpy(zero_copy_only=False))
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
    aspects = np.concatenate(asp_chunks) if asp_chunks else None

    cat_codes = {c: sorted(cat_seen[c]) for c in cat_cols}
    return proteins, labels, pairs, cat_codes, go_terms, aspects


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
# the limit. Highly annotated proteins (e.g. TGF-b1 / P01137) blow past this
# when 12 snapshot pairs x 5 KNN neighbours x dozens of GO candidates pile up.
# In aspect-conditioned mode the group key is (protein, aspect, side) so the
# 9999 cap applies per (protein, aspect) pair; a protein with many aspects may
# have up to 3 * 9999 rows total, which is well within LightGBM's limits.
LGBM_LAMBDARANK_MAX_GROUP = 9999


def _cap_oversized_groups(
    *,
    proteins: np.ndarray,
    labels: np.ndarray,
    keep_mask: np.ndarray,
    val_mask: np.ndarray,
    max_group_size: int,
    seed: int,
    aspects: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    """Drop random non-positive rows from any group bucket that exceeds
    ``max_group_size``.

    The group key is:
    - ``(protein, side)``           when ``aspects is None`` (legacy path).
    - ``(protein, aspect, side)``   when ``aspects`` is provided
                                    (aspect-conditioned path).

    Returns ``(keep_mask, n_dropped)``.
    """
    if max_group_size <= 0 or len(proteins) == 0:
        return keep_mask, 0
    rng = np.random.default_rng(seed + 1)
    side = val_mask.astype(np.int8)

    if aspects is not None:
        order = np.lexsort((side, aspects, proteins))
        sorted_proteins = proteins[order]
        sorted_aspects = aspects[order]
        sorted_side = side[order]
        boundary = (
            (sorted_proteins[1:] != sorted_proteins[:-1])
            | (sorted_aspects[1:] != sorted_aspects[:-1])
            | (sorted_side[1:] != sorted_side[:-1])
        )
    else:
        order = np.lexsort((side, proteins))
        sorted_proteins = proteins[order]
        sorted_side = side[order]
        boundary = (
            (sorted_proteins[1:] != sorted_proteins[:-1])
            | (sorted_side[1:] != sorted_side[:-1])
        )

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


@dataclass
class _BatchColumns:
    """Decoded columns of one Pass-1 batch, ready to slice + write."""

    accessions: pa.Array
    labels: np.ndarray
    feat_arrays: dict[str, np.ndarray]
    go_terms: pa.Array | None
    aspects: pa.Array | None


def _build_bucket_table(
    idx: np.ndarray,
    cols: _BatchColumns,
    layout: FeatureLayout,
    cat_set: set[str],
    *,
    carry_go_terms: bool,
    carry_aspect: bool = False,
) -> pa.Table:
    """Assemble the parquet table for one bucket's selected rows."""
    idx_arr = pa.array(idx)
    cols_data: list[pa.Array] = [
        cols.accessions.take(idx_arr),
        pa.array(cols.labels[idx], type=pa.int8()),
    ]
    if carry_go_terms:
        cols_data.append(
            cols.go_terms.take(idx_arr) if cols.go_terms is not None
            else pa.array([None] * len(idx), type=pa.string())
        )
    if carry_aspect:
        cols_data.append(
            cols.aspects.take(idx_arr) if cols.aspects is not None
            else pa.array([""] * len(idx), type=pa.string())
        )
    for c in layout.feature_cols:
        a = cols.feat_arrays[c][idx]
        cols_data.append(
            pa.array(a, type=pa.int32() if c in cat_set else pa.float32())
        )
    return pa.Table.from_arrays(cols_data, schema=layout.schema)


def _pass1_route_and_write(
    *,
    source: SourceScan,
    layout: FeatureLayout,
    routing: BucketRouting,
    keep_mask: np.ndarray,
    val_mask: np.ndarray,
    override_labels: np.ndarray | None = None,
    carry_go_terms: bool = False,
    carry_aspect: bool = False,
) -> tuple[int, int]:
    """Stream filter + cat-encode + bucket-route. Return ``(n_train, n_val)``.

    If ``override_labels`` is provided (length matching the cell-filtered row
    count produced by Pass 0), it replaces the source's ``label`` column --
    used for True-Path-Rule label propagation. ``carry_go_terms`` adds the
    ``go_term_id`` column so IA sample weighting can map each row to IA(go).
    ``carry_aspect`` adds the ``aspect`` column for aspect-conditioned grouping.
    """
    cols = ["protein_accession", "label", *layout.feature_cols]
    if carry_go_terms:
        cols.append("go_term_id")
    if carry_aspect:
        cols.append("aspect")
    cols = _present_columns(source.parquet_path, cols)
    cat_set = set(layout.categorical_cols)
    code_maps = {
        c: {v: i for i, v in enumerate(layout.cat_codes[c])}
        for c in layout.categorical_cols
    }
    n_train = 0
    n_val = 0
    cursor = 0
    for batch in iter_batches(
        source.parquet_path,
        columns=cols,
        category=source.category,
        aspect=source.aspect,
        snapshot_pairs=source.snapshot_pairs,
        batch_size=source.batch_size,
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
        aspects_col = (
            batch.column("aspect")
            if carry_aspect and "aspect" in batch.schema.names
            else None
        )
        feat_arrays = _encode_feature_arrays(batch, layout.feature_cols, cat_set, code_maps)
        bucket_idx = _bucket_array(accessions, routing.bucket_count)
        batch_cols = _BatchColumns(
            accessions=accessions, labels=labels,
            feat_arrays=feat_arrays, go_terms=go_terms,
            aspects=aspects_col,
        )

        for is_val, writers in (
            (False, routing.train_writers),
            (True, routing.val_writers),
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
                table = _build_bucket_table(
                    idx, batch_cols, layout, cat_set,
                    carry_go_terms=carry_go_terms,
                    carry_aspect=carry_aspect,
                )
                writers[int(b)].write_table(table)
                if is_val:
                    n_val += len(idx)
                else:
                    n_train += len(idx)
    return n_train, n_val

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


@dataclass
class StagePlan:
    """Split + snapshot + bucketing knobs for :func:`stage_for_training`.

    VALID / TEST window fields (F-RERANK-UNIVERSAL.3)
    --------------------------------------------------
    ``train_snapshot_pairs``
        Restrict training rows to these snapshot pairs.  ``None`` = all pairs.
    ``eval_snapshot_pair``
        VALID window: the single snapshot pair used for the primary
        evaluation / selection pass (e.g. ``"v226-v227"``).
    ``test_snapshot_pairs``
        TEST multi-window curve: a list of snapshot pairs for the
        evaluate-once TEST step (e.g. ``["v227-v228", "v227-v229",
        "v227-v230"]``).  ``None`` means no TEST window is staged here.
        Consumed by downstream callers; stage_for_training passes this
        through to callers via ``StageResult`` metadata (not staged as a
        separate split -- that is done by the caller for the frozen model).

    Aspect-conditioned grouping (F-RERANK-UNIVERSAL.3)
    ---------------------------------------------------
    ``aspect_conditioned``
        When ``True``, staging removes the per-cell aspect filter so all
        aspects flow through together.  The ``aspect`` column is written to
        bucket parquets as a conditioning feature and the LambdaRank group
        key changes from per-protein to per-(protein, aspect).  The bucket
        router remains ``crc32(protein_accession) % bucket_count`` so group
        contiguity across buckets is never violated.
    """

    val_strategy: str
    val_fraction: float
    val_holdout_snapshot: str | None
    neg_pos_ratio: float | None
    seed: int
    train_snapshot_pairs: list[str] | None = None
    eval_snapshot_pair: str | None = None
    test_snapshot_pairs: list[str] | None = None
    bucket_count: int = 32
    batch_size: int = 200_000
    parent_map_path: Path | str | None = None
    carry_go_terms: bool = False
    aspect_conditioned: bool = False


def _propagate_train_labels(
    *,
    proteins0: np.ndarray,
    labels0: np.ndarray,
    go_terms0: np.ndarray | None,
    parent_map_path: Path | str,
) -> tuple[np.ndarray, dict[str, int]]:
    """Apply True-Path-Rule propagation to the train labels in place-of-copy."""
    if go_terms0 is None:
        raise RuntimeError("propagation requested but go_term_id column is missing")
    parent_map = load_parent_map(parent_map_path)
    n_pos_before = int((labels0 > 0).sum())
    labels0, n_promoted = propagate_labels_to_ancestors(
        proteins0, go_terms0, labels0, parent_map,
    )
    stats = {
        "positives_before": n_pos_before,
        "positives_after": int((labels0 > 0).sum()),
        "rows_promoted": n_promoted,
    }
    return labels0, stats


def _eval_override_labels(
    *,
    source_eval_parquet: Path,
    cat: str,
    asp: str | None,
    plan: StagePlan,
    propagation_stats: dict[str, int],
) -> np.ndarray | None:
    """Compute propagated eval labels (and update stats) when propagating.

    When ``asp is None`` (aspect-conditioned path) no aspect filter is applied
    to the eval pass.
    """
    if plan.parent_map_path is None:
        return None
    parent_map_local = load_parent_map(plan.parent_map_path)
    ev_proteins, ev_labels, _, _, ev_go_terms, _ = _scan_pass0(
        source_eval_parquet,
        category=cat, aspect=asp,
        snapshot_pairs=[plan.eval_snapshot_pair] if plan.eval_snapshot_pair else None,
        cat_cols=[],
        batch_size=plan.batch_size,
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
    return ev_labels


def stage_for_training(
    *,
    source_train_parquet: Path,
    source_eval_parquet: Path,
    cell: tuple[str, str],
    feature_cols: list[str],
    categorical_cols: list[str],
    out_dir: Path,
    plan: StagePlan,
) -> StageResult:
    """Stage training + val + eval splits.

    When ``plan.aspect_conditioned=True`` the second element of ``cell`` is
    IGNORED as a filter -- all aspects flow through together.  The ``aspect``
    column is written to bucket parquets as a conditioning feature and the
    LambdaRank group key becomes per-(protein, aspect).

    When ``plan.aspect_conditioned=False`` (default) the behaviour is
    identical to the original per-cell per-aspect path.
    """
    cat = cell[0].lower()
    # Aspect-conditioned path: no aspect filter, aspect col written to buckets.
    # Legacy path: filter rows to the single aspect declared in ``cell``.
    asp_filter: str | None = None if plan.aspect_conditioned else cell[1].lower()
    asp_label = cell[1].lower()  # used only for error messages + JSON metadata

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # The eval split never carries go_term_id (IA weights apply to training
    # only), so train/val and eval use distinct schemas.
    # In aspect-conditioned mode the ``aspect`` column is included in ALL
    # schemas (train, val, eval) so it is available as a feature AND as
    # the group-key discriminator in _sort_bucket.
    eval_schema = _build_pass1_schema(
        feature_cols, categorical_cols,
        carry_aspect=plan.aspect_conditioned,
    )
    train_schema = _build_pass1_schema(
        feature_cols, categorical_cols,
        carry_go_terms=plan.carry_go_terms,
        carry_aspect=plan.aspect_conditioned,
    )
    numeric_cols = [c for c in feature_cols if c not in set(categorical_cols)]

    propagate = plan.parent_map_path is not None
    proteins0, labels0, pairs0, cat_codes, go_terms0, aspects0 = _scan_pass0(
        source_train_parquet,
        category=cat, aspect=asp_filter,
        snapshot_pairs=plan.train_snapshot_pairs,
        cat_cols=categorical_cols,
        batch_size=plan.batch_size,
        collect_go_terms=propagate,
        collect_aspects=plan.aspect_conditioned,
    )

    propagation_stats: dict[str, int] = {}
    if propagate:
        labels0, propagation_stats = _propagate_train_labels(
            proteins0=proteins0, labels0=labels0,
            go_terms0=go_terms0, parent_map_path=plan.parent_map_path,
        )

    keep_mask, val_mask, val_proteins = _decide_split(
        proteins=proteins0, labels=labels0, pairs=pairs0,
        val_strategy=plan.val_strategy,
        val_fraction=plan.val_fraction,
        val_holdout_snapshot=plan.val_holdout_snapshot,
        neg_pos_ratio=plan.neg_pos_ratio,
        seed=plan.seed,
    )

    keep_mask, _ = _cap_oversized_groups(
        proteins=proteins0, labels=labels0,
        keep_mask=keep_mask, val_mask=val_mask,
        max_group_size=LGBM_LAMBDARANK_MAX_GROUP,
        seed=plan.seed,
        aspects=aspects0 if plan.aspect_conditioned else None,
    )

    with tempfile.TemporaryDirectory(prefix="staging_buckets_", dir=out_dir) as tmp:
        tmp_dir = Path(tmp)
        train_writers = _open_bucket_writers(tmp_dir, train_schema, plan.bucket_count, "train")
        val_writers = (
            _open_bucket_writers(tmp_dir, train_schema, plan.bucket_count, "val")
            if plan.val_strategy != "none" else None
        )

        _pass1_route_and_write(
            source=SourceScan(
                parquet_path=source_train_parquet, category=cat, aspect=asp_filter,
                snapshot_pairs=plan.train_snapshot_pairs, batch_size=plan.batch_size,
            ),
            layout=FeatureLayout(
                feature_cols=feature_cols, categorical_cols=categorical_cols,
                cat_codes=cat_codes, schema=train_schema,
            ),
            routing=BucketRouting(
                bucket_count=plan.bucket_count,
                train_writers=train_writers, val_writers=val_writers,
            ),
            keep_mask=keep_mask, val_mask=val_mask,
            override_labels=labels0 if propagate else None,
            carry_go_terms=plan.carry_go_terms,
            carry_aspect=plan.aspect_conditioned,
        )

        train_split = _materialise_split(
            bucket_writers=train_writers,
            bucket_dir=tmp_dir,
            bucket_count=plan.bucket_count,
            out_dir=out_dir / "train",
            name="train",
        )
        val_split = None
        if val_writers is not None:
            val_split = _materialise_split(
                bucket_writers=val_writers,
                bucket_dir=tmp_dir,
                bucket_count=plan.bucket_count,
                out_dir=out_dir / "val",
                name="val",
            )

        eval_writers = _open_bucket_writers(tmp_dir, eval_schema, plan.bucket_count, "eval")
        eval_override_labels = _eval_override_labels(
            source_eval_parquet=source_eval_parquet, cat=cat, asp=asp_filter,
            plan=plan, propagation_stats=propagation_stats,
        )
        _stream_eval(
            source_eval_parquet=source_eval_parquet,
            category=cat, aspect=asp_filter,
            snapshot_pair=plan.eval_snapshot_pair,
            feature_cols=feature_cols,
            categorical_cols=categorical_cols,
            cat_codes=cat_codes,
            bucket_count=plan.bucket_count,
            writers=eval_writers,
            schema=eval_schema,
            batch_size=plan.batch_size,
            override_labels=eval_override_labels,
            carry_aspect=plan.aspect_conditioned,
        )
        eval_split = _materialise_split(
            bucket_writers=eval_writers,
            bucket_dir=tmp_dir,
            bucket_count=plan.bucket_count,
            out_dir=out_dir / "eval",
            name="eval",
        )

    if eval_split is None:
        raise RuntimeError(
            f"eval split is empty for cell {cat}-{asp_label} "
            f"(snapshot_pair={plan.eval_snapshot_pair})"
        )
    if train_split is None:
        raise RuntimeError(f"train split is empty for cell {cat}-{asp_label}")

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
            json.dumps(
                {"parent_map": str(plan.parent_map_path), **propagation_stats},
                indent=2,
            )
        )
    result.to_json(out_dir / "staging.json")
    return result


def _stream_eval(
    *,
    source_eval_parquet: Path,
    category: str,
    aspect: str | None,
    snapshot_pair: str | None,
    feature_cols: list[str],
    categorical_cols: list[str],
    cat_codes: dict[str, list[str]],
    bucket_count: int,
    writers: list[pq.ParquetWriter],
    schema: pa.Schema,
    batch_size: int,
    override_labels: np.ndarray | None = None,
    carry_aspect: bool = False,
) -> None:
    """Stream the eval parquet into bucket writers.

    When ``carry_aspect=True`` the ``aspect`` column is read and written into
    the bucket so ``_sort_bucket`` can form per-(protein, aspect) groups.
    """
    cols = ["protein_accession", "label", *feature_cols]
    if carry_aspect:
        cols.append("aspect")
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
        aspects_col = (
            batch.column("aspect") if carry_aspect and "aspect" in batch.schema.names
            else None
        )
        bucket_idx = _bucket_array(accessions, bucket_count)
        for b in np.unique(bucket_idx):
            row_mask = bucket_idx == b
            idx = np.flatnonzero(row_mask)
            idx_arr = pa.array(idx)
            cols_data: list[pa.Array] = [
                accessions.take(idx_arr),
                pa.array(labels[idx], type=pa.int8()),
            ]
            if carry_aspect:
                cols_data.append(
                    aspects_col.take(idx_arr) if aspects_col is not None
                    else pa.array([""] * len(idx), type=pa.string())
                )
            for c in feature_cols:
                a = feat[c][idx]
                if c in cat_set:
                    cols_data.append(pa.array(a, type=pa.int32()))
                else:
                    cols_data.append(pa.array(a, type=pa.float32()))
            table = pa.Table.from_arrays(cols_data, schema=schema)
            writers[int(b)].write_table(table)
