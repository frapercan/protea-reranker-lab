"""Pooled multi-manifest staging for the universal reranker (F-RERANK-UNIVERSAL.5a).

Streams each of the 24 v226-lineage manifests (8 PLM x K{3,5,10}) through
shared bucket writers without writing a physical combined parquet (avoids
write-OOM). ``plm_id`` and ``k_context`` are injected as constants per source.

Memory-bounded design: per-source Pass 0 scans ONLY
(protein_accession, label, snapshot_pair, aspect), computes keep_mask / val_mask
immediately, then frees the row arrays. Peak RAM per source is ~1 GB; global
state is only the list of boolean masks + the tiny cat-vocab sets.

Public entry point: :func:`stage_for_training_pooled`.
"""

from __future__ import annotations

import json
import logging
import resource
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from .bucket_io import (
    CAT_MISSING_CODE,
    _bucket_array,
    _build_pass1_schema,
    _encode_cat_batch,
    _open_bucket_writers,
    _present_columns,
)
from .data import iter_batches
from .splits import StagedSplit, StageResult, _materialise_split
from .staging import (
    BucketRouting,
    _CarryFlags,
    FeatureLayout,
    LGBM_LAMBDARANK_MAX_GROUP,
    StagePlan,
    _BatchColumns,
    _build_bucket_table,
    _cap_oversized_groups,
    _decide_split,
    _eval_override_labels,
)

if TYPE_CHECKING:
    from .multi_source import ManifestSource, MultiManifestSpec

log = logging.getLogger(__name__)

# Columns that are NOT present in the raw parquet but are injected
# per-source as constants (plm_id, k_context).
_INJECTABLE_COLS = frozenset({"plm_id", "k_context"})

# Maximum total training rows across all sources.  If the keep masks
# select more rows, we uniformly downsample the negatives to stay within
# this budget.  At ~30 bytes/row in memory this costs ~12 GB during
# staging (the actual LightGBM binned dataset is much smaller).
# Set to None to disable the budget.
_GLOBAL_ROW_BUDGET: int | None = 80_000_000


def _rss_gb() -> float:
    """Return current process RSS in GB (Linux maxrss is kB)."""
    try:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# Per-source vocabulary scan (Pass 0a): global cat vocab only
# ---------------------------------------------------------------------------


def _build_global_cat_codes(
    multi_spec: "MultiManifestSpec",
    *,
    categorical_cols: list[str],
    plan: StagePlan,
) -> dict[str, list[str]]:
    """Scan all sources to collect global categorical vocab (tiny data only)."""
    plm_ids_vocab = sorted({src.plm_id for src in multi_spec.sources})
    other_cat_cols = [c for c in categorical_cols if c not in _INJECTABLE_COLS]
    global_cat_codes: dict[str, list[str]] = {"plm_id": plm_ids_vocab}
    other_cat_seen: dict[str, set] = {c: set() for c in other_cat_cols}

    for src in multi_spec.sources:
        train_pq = src.parquet_dir / "train.parquet"
        if not train_pq.exists():
            continue
        scan_cols = _present_columns(train_pq, other_cat_cols)
        if not scan_cols:
            continue
        for batch in iter_batches(
            train_pq, columns=scan_cols, category=None, aspect=None,
            snapshot_pairs=plan.train_snapshot_pairs,
            batch_size=plan.batch_size,
        ):
            for c in other_cat_cols:
                if c not in batch.schema.names:
                    continue
                for v in pc.unique(batch.column(c)).to_pylist():
                    if v is not None:
                        other_cat_seen[c].add(v)

    global_cat_codes.update({c: sorted(other_cat_seen[c]) for c in other_cat_cols})
    return global_cat_codes


# ---------------------------------------------------------------------------
# Per-source split decision (Pass 0b): keep + val masks, one source at a time
# ---------------------------------------------------------------------------


@dataclass
class _SourceMasks:
    """Keep + val boolean masks for one training source."""

    keep: np.ndarray  # dtype=bool, len = n_rows in source
    val: np.ndarray   # dtype=bool, same length


def _compute_source_masks(
    train_pq: Path,
    *,
    plan: StagePlan,
) -> _SourceMasks:
    """Run a minimal scan of one source to build per-row keep + val masks.

    Only reads protein_accession, label, snapshot_pair (+ aspect when needed
    for group-cap).  Frees the row arrays before returning.
    """
    # Collect proteins / labels / snapshot_pairs in streaming batches.
    scan_cols = _present_columns(train_pq, ["protein_accession", "label",
                                             "snapshot_pair", "aspect"])
    prot_chunks: list[np.ndarray] = []
    lab_chunks: list[np.ndarray] = []
    pair_chunks: list[np.ndarray] = []
    asp_chunks: list[np.ndarray] = []
    for batch in iter_batches(
        train_pq, columns=scan_cols, category=None, aspect=None,
        snapshot_pairs=plan.train_snapshot_pairs,
        batch_size=plan.batch_size,
    ):
        prot_chunks.append(batch.column("protein_accession").to_numpy(zero_copy_only=False))
        lab_chunks.append(batch.column("label").to_numpy(zero_copy_only=False).astype(np.int8, copy=False))
        if "snapshot_pair" in batch.schema.names:
            pair_chunks.append(batch.column("snapshot_pair").to_numpy(zero_copy_only=False))
        if "aspect" in batch.schema.names:
            asp_chunks.append(batch.column("aspect").to_numpy(zero_copy_only=False))

    n = sum(len(c) for c in prot_chunks)
    if n == 0:
        return _SourceMasks(
            keep=np.empty(0, dtype=bool),
            val=np.empty(0, dtype=bool),
        )

    proteins = np.concatenate(prot_chunks) if prot_chunks else np.empty(0, dtype=object)
    labels = np.concatenate(lab_chunks) if lab_chunks else np.empty(0, dtype=np.int8)
    pairs = (np.concatenate(pair_chunks) if pair_chunks
             else np.empty(len(proteins), dtype=object))
    aspects = np.concatenate(asp_chunks) if asp_chunks else None

    # Free chunk lists immediately.
    del prot_chunks, lab_chunks, pair_chunks, asp_chunks

    keep, val, _ = _decide_split(
        proteins=proteins, labels=labels, pairs=pairs,
        val_strategy=plan.val_strategy, val_fraction=plan.val_fraction,
        val_holdout_snapshot=plan.val_holdout_snapshot,
        neg_pos_ratio=plan.neg_pos_ratio, seed=plan.seed,
    )
    keep, _ = _cap_oversized_groups(
        proteins=proteins, labels=labels, keep_mask=keep, val_mask=val,
        max_group_size=LGBM_LAMBDARANK_MAX_GROUP, seed=plan.seed, aspects=aspects,
    )

    # Free large row arrays immediately.
    del proteins, labels, pairs, aspects

    return _SourceMasks(keep=keep, val=val)


# ---------------------------------------------------------------------------
# Bounded pass-0: vocab scan + per-source masks
# ---------------------------------------------------------------------------


@dataclass
class _BoundedPooledScan:
    """Memory-bounded Pass-0 result: per-source masks + global vocab."""

    per_source_masks: list[_SourceMasks | None]  # None when source parquet missing
    global_cat_codes: dict[str, list[str]]
    n_rows_per_source: list[int]
    n_total_kept: int
    downsample_factor: float  # 1.0 = no downsample, < 1.0 = negatives downsampled


def _collect_source_masks(
    multi_spec: "MultiManifestSpec",
    plan: StagePlan,
) -> tuple[list[_SourceMasks | None], list[int], int]:
    """Compute per-source keep/val masks and free row arrays immediately."""
    per_source_masks: list[_SourceMasks | None] = []
    n_rows_per_source: list[int] = []
    n_total_kept = 0
    for i, src in enumerate(multi_spec.sources):
        train_pq = src.parquet_dir / "train.parquet"
        if not train_pq.exists():
            per_source_masks.append(None)
            n_rows_per_source.append(0)
            log.warning("[pooled_pass0] source %d missing: %s", i, train_pq)
            continue
        log.info(
            "[pooled_pass0] source %d/%d %s K%d (RSS %.1f GB)",
            i + 1, len(multi_spec.sources), src.plm_id, src.k_context, _rss_gb(),
        )
        masks = _compute_source_masks(train_pq, plan=plan)
        n_kept = int(masks.keep.sum())
        per_source_masks.append(masks)
        n_rows_per_source.append(n_kept)
        n_total_kept += n_kept
        log.info("[pooled_pass0]   -> %d rows kept (RSS %.1f GB)", n_kept, _rss_gb())
    return per_source_masks, n_rows_per_source, n_total_kept


def _apply_budget(
    per_source_masks: list[_SourceMasks | None],
    n_total_kept: int,
    budget: int,
    seed: int,
) -> tuple[int, float]:
    """Downsample train negatives uniformly across sources to meet budget.

    Returns (n_after, downsample_factor).  Positives and val rows are never
    dropped.  Drops are logged; callers must surface this in run.json.
    """
    downsample_factor = budget / n_total_kept
    log.warning(
        "[pooled_pass0] GLOBAL ROW BUDGET hit: %d rows > budget %d "
        "(factor=%.3f). Downsampling negatives uniformly.",
        n_total_kept, budget, downsample_factor,
    )
    rng = np.random.default_rng(seed + 99)
    n_after = 0
    for masks in per_source_masks:
        if masks is None:
            continue
        train_neg_idx = np.flatnonzero(masks.keep & ~masks.val)
        n_keep_neg = max(1, int(len(train_neg_idx) * downsample_factor))
        if n_keep_neg < len(train_neg_idx):
            drop_idx = rng.choice(
                train_neg_idx, size=len(train_neg_idx) - n_keep_neg, replace=False,
            )
            masks.keep[drop_idx] = False
        n_after += int(masks.keep.sum())
    log.info("[pooled_pass0] After budget: %d rows (target %d)", n_after, budget)
    return n_after, downsample_factor


def _bounded_pass0(
    multi_spec: "MultiManifestSpec",
    *,
    categorical_cols: list[str],
    plan: StagePlan,
    propagate: bool,
) -> _BoundedPooledScan:
    """Memory-bounded Pass 0: per-source masks + global cat vocab.

    Never holds more than one source worth of row arrays in RAM at a time.
    """
    if propagate:
        raise NotImplementedError(
            "Label propagation is not supported in the memory-bounded pooled path."
        )
    log.info("[pooled_pass0] building global cat vocab (RSS %.1f GB)", _rss_gb())
    global_cat_codes = _build_global_cat_codes(
        multi_spec, categorical_cols=categorical_cols, plan=plan,
    )
    per_source_masks, n_rows_per_source, n_total_kept = _collect_source_masks(
        multi_spec, plan,
    )
    downsample_factor = 1.0
    if _GLOBAL_ROW_BUDGET is not None and n_total_kept > _GLOBAL_ROW_BUDGET:
        n_total_kept, downsample_factor = _apply_budget(
            per_source_masks, n_total_kept, _GLOBAL_ROW_BUDGET, plan.seed,
        )
    return _BoundedPooledScan(
        per_source_masks=per_source_masks,
        global_cat_codes=global_cat_codes,
        n_rows_per_source=n_rows_per_source,
        n_total_kept=n_total_kept,
        downsample_factor=downsample_factor,
    )


# ---------------------------------------------------------------------------
# Source-level encoding context
# ---------------------------------------------------------------------------


@dataclass
class _SrcEncoder:
    """Encoding context for one source: cat codes + injected constants."""

    cat_set: set[str]
    code_maps: dict[str, dict[str, int]]
    plm_code: int
    k_val_f32: np.float32

    @classmethod
    def from_layout_and_src(
        cls, layout: FeatureLayout, src: "ManifestSource",
    ) -> "_SrcEncoder":
        cat_set = set(layout.categorical_cols)
        code_maps = {
            c: {v: i for i, v in enumerate(layout.cat_codes[c])}
            for c in layout.categorical_cols
        }
        return cls(
            cat_set=cat_set, code_maps=code_maps,
            plm_code=code_maps.get("plm_id", {}).get(src.plm_id, CAT_MISSING_CODE),
            k_val_f32=np.float32(src.k_context),
        )

    @classmethod
    def from_cat_codes_and_src(
        cls,
        categorical_cols: list[str],
        cat_codes: dict[str, list[str]],
        src: "ManifestSource",
    ) -> "_SrcEncoder":
        cat_set = set(categorical_cols)
        code_maps = {c: {v: i for i, v in enumerate(cat_codes[c])} for c in categorical_cols}
        return cls(
            cat_set=cat_set, code_maps=code_maps,
            plm_code=code_maps.get("plm_id", {}).get(src.plm_id, CAT_MISSING_CODE),
            k_val_f32=np.float32(src.k_context),
        )


def _inject_src_features(
    batch: pa.RecordBatch, feature_cols: list[str], enc: "_SrcEncoder",
) -> dict[str, np.ndarray]:
    """Build feat_arrays for one batch, injecting plm_id and k_context constants."""
    m = batch.num_rows
    feat_arrays: dict[str, np.ndarray] = {}
    for c in feature_cols:
        if c == "plm_id":
            feat_arrays[c] = np.full(m, enc.plm_code, dtype=np.int32)
        elif c == "k_context":
            feat_arrays[c] = np.full(m, enc.k_val_f32, dtype=np.float32)
        elif c not in batch.schema.names:
            feat_arrays[c] = (
                np.full(m, CAT_MISSING_CODE, dtype=np.int32)
                if c in enc.cat_set else np.full(m, np.nan, dtype=np.float32)
            )
        elif c in enc.cat_set:
            feat_arrays[c] = _encode_cat_batch(batch.column(c), enc.code_maps[c])
        else:
            arr = batch.column(c).to_numpy(zero_copy_only=False)
            feat_arrays[c] = arr.astype(np.float32, copy=False) if arr.dtype != np.float32 else arr
    return feat_arrays


# ---------------------------------------------------------------------------
# Pass-1: route each source through bucket writers
# ---------------------------------------------------------------------------


@dataclass
class _Pass1PooledCtx:
    """Context for :func:`_pass1_route_pooled_source` (reduces param count)."""

    parquet_path: Path
    src: "ManifestSource"
    layout: FeatureLayout
    routing: BucketRouting
    keep_mask: np.ndarray
    val_mask: np.ndarray
    override_labels: np.ndarray | None
    carry_go_terms: bool
    snapshot_pairs: list[str] | None
    batch_size: int


def _pass1_route_batch(
    batch: pa.RecordBatch, ctx: _Pass1PooledCtx, enc: _SrcEncoder, cursor_slice: slice,
) -> None:
    """Route one batch from a pooled source into train/val bucket writers."""
    local_keep = ctx.keep_mask[cursor_slice]
    local_val = ctx.val_mask[cursor_slice]
    labels = (
        ctx.override_labels[cursor_slice].astype(np.int8, copy=False)
        if ctx.override_labels is not None
        else batch.column("label").to_numpy(zero_copy_only=False).astype(np.int8, copy=False)
    )
    if not local_keep.any():
        return
    accessions = batch.column("protein_accession")
    go_terms_col = (
        batch.column("go_term_id")
        if ctx.carry_go_terms and "go_term_id" in batch.schema.names else None
    )
    aspects_col = batch.column("aspect") if "aspect" in batch.schema.names else None
    # Carry snapshot_pair for the extended 4-tuple group key (F-RERANK-UNIVERSAL.5d).
    snapshot_pairs_col = (
        batch.column("snapshot_pair") if "snapshot_pair" in batch.schema.names else None
    )
    feat_arrays = _inject_src_features(batch, ctx.layout.feature_cols, enc)
    bucket_idx = _bucket_array(accessions, ctx.routing.bucket_count)
    batch_cols = _BatchColumns(
        accessions=accessions, labels=labels,
        feat_arrays=feat_arrays, go_terms=go_terms_col, aspects=aspects_col,
        snapshot_pairs=snapshot_pairs_col,
    )
    _route_to_writers(batch_cols, bucket_idx, ctx, local_keep, local_val, enc.cat_set)


def _route_to_writers(
    batch_cols: _BatchColumns,
    bucket_idx: np.ndarray,
    ctx: _Pass1PooledCtx,
    local_keep: np.ndarray,
    local_val: np.ndarray,
    cat_set: set[str],
) -> None:
    """Write batch rows to the appropriate train/val bucket writers."""
    for is_val, writers in ((False, ctx.routing.train_writers), (True, ctx.routing.val_writers)):
        if writers is None:
            continue
        sel = local_keep & (local_val if is_val else ~local_val)
        if not sel.any():
            continue
        for b in np.unique(bucket_idx[sel]):
            row_mask = sel & (bucket_idx == b)
            if not row_mask.any():
                continue
            table = _build_bucket_table(
                np.flatnonzero(row_mask), batch_cols, ctx.layout, cat_set,
                _CarryFlags(go_terms=ctx.carry_go_terms, aspect=True, snapshot_pair=True),
            )
            writers[int(b)].write_table(table)


def _pass1_route_pooled_source(ctx: _Pass1PooledCtx) -> None:
    """Stream one pooled source through the global bucket writers.

    Injects plm_id and k_context as constants (absent from the raw parquet).
    """
    # Build deduped column list preserving order (aspect may already be in feature_cols).
    seen: set[str] = set()
    cols: list[str] = []
    for c in ["protein_accession", "label", *ctx.layout.feature_cols]:
        if c not in seen:
            cols.append(c)
            seen.add(c)
    if ctx.carry_go_terms and "go_term_id" not in seen:
        cols.append("go_term_id")
    if "aspect" not in seen:
        cols.append("aspect")
    # Carry snapshot_pair for the extended 4-tuple group key (F-RERANK-UNIVERSAL.5d).
    if "snapshot_pair" not in seen:
        cols.append("snapshot_pair")
    scan_cols = _present_columns(ctx.parquet_path, [c for c in cols if c not in _INJECTABLE_COLS])
    enc = _SrcEncoder.from_layout_and_src(ctx.layout, ctx.src)
    cursor = 0
    for batch in iter_batches(
        ctx.parquet_path, columns=scan_cols, category=None, aspect=None,
        snapshot_pairs=ctx.snapshot_pairs, batch_size=ctx.batch_size,
    ):
        m = batch.num_rows
        _pass1_route_batch(batch, ctx, enc, slice(cursor, cursor + m))
        cursor += m


# ---------------------------------------------------------------------------
# Eval streaming
# ---------------------------------------------------------------------------


@dataclass
class _EvalPooledCtx:
    """Context for :func:`_stream_eval_pooled` (reduces param count)."""

    eval_pq: Path
    src: "ManifestSource"
    feature_cols: list[str]
    categorical_cols: list[str]
    cat_codes: dict[str, list[str]]
    bucket_count: int
    writers: list
    schema: pa.Schema
    snapshot_pair: str | None
    batch_size: int
    override_labels: np.ndarray | None


def _stream_eval_pooled(ctx: _EvalPooledCtx) -> None:
    """Stream one eval source into bucket writers, injecting plm_id + k_context."""
    # Deduplicate (aspect / snapshot_pair may already be in feature_cols).
    seen: set[str] = {"protein_accession", "label"}
    scan_cols_raw: list[str] = ["protein_accession", "label"]
    for c in ctx.feature_cols:
        if c not in seen:
            scan_cols_raw.append(c)
            seen.add(c)
    if "aspect" not in seen:
        scan_cols_raw.append("aspect")
    # Carry snapshot_pair for the extended 4-tuple group key (F-RERANK-UNIVERSAL.5d).
    if "snapshot_pair" not in seen:
        scan_cols_raw.append("snapshot_pair")
    scan_cols = _present_columns(ctx.eval_pq,
                                 [c for c in scan_cols_raw if c not in _INJECTABLE_COLS])
    enc = _SrcEncoder.from_cat_codes_and_src(ctx.categorical_cols, ctx.cat_codes, ctx.src)
    cursor = 0
    for batch in iter_batches(
        ctx.eval_pq, columns=scan_cols, category=None, aspect=None,
        snapshot_pair=ctx.snapshot_pair, batch_size=ctx.batch_size,
    ):
        m = batch.num_rows
        if m == 0:
            continue
        _write_eval_batch(batch, ctx, enc, cursor)
        cursor += m


def _write_eval_batch(
    batch: pa.RecordBatch, ctx: _EvalPooledCtx, enc: _SrcEncoder, cursor: int,
) -> None:
    """Write one eval batch into bucket writers."""
    m = batch.num_rows
    accessions = batch.column("protein_accession")
    if ctx.override_labels is not None:
        labels = ctx.override_labels[cursor:cursor + m].astype(np.int8, copy=False)
    else:
        labels = batch.column("label").to_numpy(zero_copy_only=False).astype(np.int8, copy=False)

    feat = _inject_src_features(batch, ctx.feature_cols, enc)
    aspects_col = batch.column("aspect") if "aspect" in batch.schema.names else None
    # Carry snapshot_pair for the extended 4-tuple group key (F-RERANK-UNIVERSAL.5d).
    snapshot_pairs_col = (
        batch.column("snapshot_pair") if "snapshot_pair" in batch.schema.names else None
    )
    bucket_idx = _bucket_array(accessions, ctx.bucket_count)
    for b in np.unique(bucket_idx):
        idx = np.flatnonzero(bucket_idx == b)
        idx_arr = pa.array(idx)
        cols_data: list[pa.Array] = [
            accessions.take(idx_arr),
            pa.array(labels[idx], type=pa.int8()),
            (aspects_col.take(idx_arr) if aspects_col is not None
             else pa.array([""] * len(idx), type=pa.string())),
            (snapshot_pairs_col.take(idx_arr) if snapshot_pairs_col is not None
             else pa.array([""] * len(idx), type=pa.string())),
        ]
        for c in ctx.feature_cols:
            a = feat[c][idx]
            cols_data.append(pa.array(a, type=pa.int32() if c in enc.cat_set else pa.float32()))
        ctx.writers[int(b)].write_table(pa.Table.from_arrays(cols_data, schema=ctx.schema))


# ---------------------------------------------------------------------------
# Write splits: open writers, route, materialise
# ---------------------------------------------------------------------------


@dataclass
class _BoundedWriteCtx:
    """Parameters for :func:`_bounded_write_splits` (reduces param count)."""

    multi_spec: "MultiManifestSpec"
    eval_source: "ManifestSource"
    scan: _BoundedPooledScan
    feature_cols: list[str]
    categorical_cols: list[str]
    plan: StagePlan
    out_dir: Path


def _route_train_splits_bounded(
    ctx: _BoundedWriteCtx,
    layout: FeatureLayout,
    routing: BucketRouting,
    tmp_dir: Path,
) -> tuple[StagedSplit | None, StagedSplit | None]:
    """Route all sources using per-source masks and materialise train/val splits."""
    for i, src in enumerate(ctx.multi_spec.sources):
        masks = ctx.scan.per_source_masks[i]
        if masks is None:
            continue
        train_pq = src.parquet_dir / "train.parquet"
        if not train_pq.exists():
            continue
        log.info(
            "[pass1] routing source %d/%d %s K%d (RSS %.1f GB)",
            i + 1, len(ctx.multi_spec.sources), src.plm_id, src.k_context, _rss_gb(),
        )
        _pass1_route_pooled_source(_Pass1PooledCtx(
            parquet_path=train_pq, src=src, layout=layout, routing=routing,
            keep_mask=masks.keep, val_mask=masks.val,
            override_labels=None,  # propagation not supported in pooled path
            carry_go_terms=ctx.plan.carry_go_terms,
            snapshot_pairs=ctx.plan.train_snapshot_pairs,
            batch_size=ctx.plan.batch_size,
        ))
    train_split = _materialise_split(
        bucket_writers=routing.train_writers, bucket_dir=tmp_dir,
        bucket_count=ctx.plan.bucket_count, out_dir=ctx.out_dir / "train", name="train",
    )
    val_split = (
        _materialise_split(
            bucket_writers=routing.val_writers, bucket_dir=tmp_dir,
            bucket_count=ctx.plan.bucket_count,
            out_dir=ctx.out_dir / "val", name="val",
        )
        if routing.val_writers is not None else None
    )
    return train_split, val_split


def _route_eval_split_bounded(
    ctx: _BoundedWriteCtx,
    eval_schema: pa.Schema,
    tmp_dir: Path,
    feat_cols: list[str],
    cat_cols: list[str],
) -> StagedSplit | None:
    """Route eval source and materialise eval split."""
    eval_source_pq = ctx.eval_source.parquet_dir / "eval.parquet"
    eval_override = (
        _eval_override_labels(
            source_eval_parquet=eval_source_pq,
            cat=None, asp=None, plan=ctx.plan,
            propagation_stats={},
        )
        if ctx.plan.parent_map_path is not None else None
    )
    eval_writers = _open_bucket_writers(tmp_dir, eval_schema, ctx.plan.bucket_count, "eval")
    _stream_eval_pooled(_EvalPooledCtx(
        eval_pq=eval_source_pq, src=ctx.eval_source,
        feature_cols=feat_cols, categorical_cols=cat_cols,
        cat_codes=ctx.scan.global_cat_codes, bucket_count=ctx.plan.bucket_count,
        writers=eval_writers, schema=eval_schema,
        snapshot_pair=ctx.plan.eval_snapshot_pair, batch_size=ctx.plan.batch_size,
        override_labels=eval_override,
    ))
    return _materialise_split(
        bucket_writers=eval_writers, bucket_dir=tmp_dir,
        bucket_count=ctx.plan.bucket_count, out_dir=ctx.out_dir / "eval", name="eval",
    )


_BUCKET_RESERVED = frozenset(
    ("protein_accession", "label", "go_term_id", "aspect", "snapshot_pair")
)


def _bounded_write_splits(
    ctx: _BoundedWriteCtx,
) -> tuple[StagedSplit | None, StagedSplit | None, StagedSplit | None]:
    """Open bucket writers, route all sources, materialise train/val/eval splits."""
    # Strip reserved bucket columns from feature_cols: aspect / go_term_id /
    # snapshot_pair are carried as dedicated reserved columns; including them
    # in feature_cols would create duplicate field names in the bucket parquet
    # schema.
    feat_cols = [c for c in ctx.feature_cols if c not in _BUCKET_RESERVED]
    cat_cols = [c for c in ctx.categorical_cols if c not in _BUCKET_RESERVED]
    # The pooled path always uses the extended 4-tuple group key
    # (snapshot_pair, protein, aspect, plm_id) per F-RERANK-UNIVERSAL.5d.
    eval_schema = _build_pass1_schema(
        feat_cols, cat_cols, carry_aspect=True, carry_snapshot_pair=True,
    )
    train_schema = _build_pass1_schema(
        feat_cols, cat_cols,
        carry_go_terms=ctx.plan.carry_go_terms,
        carry_aspect=True, carry_snapshot_pair=True,
    )
    with tempfile.TemporaryDirectory(prefix="staging_pooled_", dir=ctx.out_dir) as tmp:
        tmp_dir = Path(tmp)
        train_writers = _open_bucket_writers(tmp_dir, train_schema, ctx.plan.bucket_count, "train")
        val_writers = (
            _open_bucket_writers(tmp_dir, train_schema, ctx.plan.bucket_count, "val")
            if ctx.plan.val_strategy != "none" else None
        )
        layout = FeatureLayout(
            feature_cols=feat_cols, categorical_cols=cat_cols,
            cat_codes=ctx.scan.global_cat_codes, schema=train_schema,
        )
        routing = BucketRouting(
            bucket_count=ctx.plan.bucket_count,
            train_writers=train_writers, val_writers=val_writers,
        )
        train_split, val_split = _route_train_splits_bounded(ctx, layout, routing, tmp_dir)
        eval_split = _route_eval_split_bounded(ctx, eval_schema, tmp_dir, feat_cols, cat_cols)
    return train_split, val_split, eval_split


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _build_stage_result_bounded(
    ctx: "_BoundedWriteCtx",
    scan: "_BoundedPooledScan",
    train_split: StagedSplit,
    val_split: StagedSplit | None,
    eval_split: StagedSplit,
) -> StageResult:
    """Assemble StageResult and persist staging.json."""
    # Use filtered feature/cat cols (reserved cols stripped) to match what's
    # actually written into the bucket parquet files.
    feat_cols = [c for c in ctx.feature_cols if c not in _BUCKET_RESERVED]
    cat_cols = [c for c in ctx.categorical_cols if c not in _BUCKET_RESERVED]
    numeric_cols = [c for c in feat_cols if c not in set(cat_cols)]
    result = StageResult(
        train=train_split, val=val_split, eval=eval_split,
        cat_codes=scan.global_cat_codes, feature_cols=feat_cols,
        numeric_cols=numeric_cols, categorical_cols=cat_cols,
        n_proteins_train=int(train_split.n_groups),
        n_proteins_val=int(val_split.n_groups) if val_split else 0,
        n_proteins_eval=int(eval_split.n_groups), out_dir=ctx.out_dir,
    )
    staging_meta = {
        "n_total_train_rows": int(train_split.n_rows),
        "n_total_val_rows": int(val_split.n_rows) if val_split else 0,
        "global_row_budget": _GLOBAL_ROW_BUDGET,
        "downsample_factor": float(scan.downsample_factor),
        "peak_rss_gb": _rss_gb(),
    }
    (ctx.out_dir / "pooled_staging_meta.json").write_text(
        json.dumps(staging_meta, indent=2)
    )
    result.to_json(ctx.out_dir / "staging.json")
    return result


def stage_for_training_pooled(
    *,
    multi_spec: "MultiManifestSpec",
    eval_source: "ManifestSource",
    feature_cols: list[str],
    categorical_cols: list[str],
    out_dir: Path,
    plan: StagePlan,
) -> StageResult:
    """Stage training + val + eval splits over ALL sources in ``multi_spec``.

    Memory-bounded: processes each source independently in Pass 0, holds at
    most one source worth of row arrays in RAM at a time, and stores only
    per-source boolean masks (1 byte/row).  Applies a global row budget
    (``_GLOBAL_ROW_BUDGET``) with negative downsampling if the total exceeds
    the budget.  Peak RSS is logged into ``pooled_staging_meta.json``.

    ``plm_id`` and ``k_context`` are injected as constants per source. Eval
    uses ``eval_source`` only (prot_t5 K10 preferred).

    Label propagation (``parent_map_path``) is NOT supported in the pooled
    path because it requires materialising the full concatenated protein+go
    arrays, which defeats the memory budget.  Set ``plan.parent_map_path=None``
    for pooled training.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    propagate = plan.parent_map_path is not None

    log.info("[pooled] starting bounded pass0 (RSS %.1f GB)", _rss_gb())
    scan = _bounded_pass0(
        multi_spec, categorical_cols=categorical_cols, plan=plan, propagate=propagate,
    )
    log.info(
        "[pooled] pass0 complete: %d total kept rows, downsample=%.3f (RSS %.1f GB)",
        scan.n_total_kept, scan.downsample_factor, _rss_gb(),
    )

    write_ctx = _BoundedWriteCtx(
        multi_spec=multi_spec, eval_source=eval_source, scan=scan,
        feature_cols=feature_cols, categorical_cols=categorical_cols,
        plan=plan, out_dir=out_dir,
    )
    train_split, val_split, eval_split = _bounded_write_splits(write_ctx)
    if eval_split is None:
        raise RuntimeError("eval split is empty for pooled staging")
    if train_split is None:
        raise RuntimeError("train split is empty for pooled staging")
    return _build_stage_result_bounded(write_ctx, scan, train_split, val_split, eval_split)
