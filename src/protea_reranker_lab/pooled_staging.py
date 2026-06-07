"""Pooled multi-manifest staging for the universal reranker (F-RERANK-UNIVERSAL.5a).

Streams each of the 24 v226-lineage manifests (8 PLM x K{3,5,10}) through
shared bucket writers without writing a physical combined parquet (avoids
write-OOM). ``plm_id`` and ``k_context`` are injected as constants per source.

This module is intentionally separated from :mod:`staging` to keep file LOC
within the smell budget.  The public entry point is
:func:`stage_for_training_pooled`; ``staging.py`` re-exports it.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa
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
    FeatureLayout,
    LGBM_LAMBDARANK_MAX_GROUP,
    StagePlan,
    _BatchColumns,
    _build_bucket_table,
    _cap_oversized_groups,
    _decide_split,
    _eval_override_labels,
    _propagate_train_labels,
    _scan_pass0,
)

if TYPE_CHECKING:
    from .multi_source import ManifestSource, MultiManifestSpec


# Columns that are NOT present in the raw parquet but are injected
# per-source as constants (plm_id, k_context).
_INJECTABLE_COLS = frozenset({"plm_id", "k_context"})


# ---------------------------------------------------------------------------
# Pass-0: global vocab scan across all sources
# ---------------------------------------------------------------------------


@dataclass
class _PooledScanResult:
    """Aggregated Pass-0 result across all pooled sources."""

    proteins0: np.ndarray
    labels0: np.ndarray
    pairs0: np.ndarray
    go_terms0: np.ndarray | None
    aspects0: np.ndarray | None
    global_cat_codes: dict[str, list[str]]
    source_offsets: list[tuple[int, int]]


def _pooled_pass0(
    multi_spec: "MultiManifestSpec",
    *,
    categorical_cols: list[str],
    plan: StagePlan,
    propagate: bool,
) -> _PooledScanResult:
    """Run Pass 0 over all sources, collecting global vocab + row arrays."""
    plm_ids_vocab = sorted({src.plm_id for src in multi_spec.sources})
    other_cat_cols = [c for c in categorical_cols if c not in _INJECTABLE_COLS]
    global_cat_codes: dict[str, list[str]] = {"plm_id": plm_ids_vocab}

    all_proteins: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_pairs: list[np.ndarray] = []
    all_go: list[np.ndarray] = []
    all_asp: list[np.ndarray] = []
    other_cat_seen: dict[str, set] = {c: set() for c in other_cat_cols}

    for src in multi_spec.sources:
        train_pq = src.parquet_dir / "train.parquet"
        if not train_pq.exists():
            continue
        prot_i, lab_i, pair_i, cat_i, go_i, asp_i = _scan_pass0(
            train_pq, category=None, aspect=None,
            snapshot_pairs=plan.train_snapshot_pairs,
            cat_cols=other_cat_cols, batch_size=plan.batch_size,
            collect_go_terms=propagate, collect_aspects=True,
        )
        all_proteins.append(prot_i)
        all_labels.append(lab_i)
        all_pairs.append(pair_i)
        if go_i is not None:
            all_go.append(go_i)
        if asp_i is not None:
            all_asp.append(asp_i)
        for c in other_cat_cols:
            other_cat_seen[c].update(cat_i.get(c, set()))

    global_cat_codes.update({c: sorted(other_cat_seen[c]) for c in other_cat_cols})

    cursor = 0
    source_offsets: list[tuple[int, int]] = []
    for chunk in all_proteins:
        source_offsets.append((cursor, cursor + len(chunk)))
        cursor += len(chunk)

    return _PooledScanResult(
        proteins0=np.concatenate(all_proteins) if all_proteins else np.empty(0, dtype=object),
        labels0=np.concatenate(all_labels) if all_labels else np.empty(0, dtype=np.int8),
        pairs0=np.concatenate(all_pairs) if all_pairs else np.empty(0, dtype=object),
        go_terms0=np.concatenate(all_go) if all_go else None,
        aspects0=np.concatenate(all_asp) if all_asp else None,
        global_cat_codes=global_cat_codes,
        source_offsets=source_offsets,
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
    feat_arrays = _inject_src_features(batch, ctx.layout.feature_cols, enc)
    bucket_idx = _bucket_array(accessions, ctx.routing.bucket_count)
    batch_cols = _BatchColumns(
        accessions=accessions, labels=labels,
        feat_arrays=feat_arrays, go_terms=go_terms_col, aspects=aspects_col,
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
                carry_go_terms=ctx.carry_go_terms, carry_aspect=True,
            )
            writers[int(b)].write_table(table)


def _pass1_route_pooled_source(ctx: _Pass1PooledCtx) -> None:
    """Stream one pooled source through the global bucket writers.

    Injects plm_id and k_context as constants (absent from the raw parquet).
    """
    cols = ["protein_accession", "label", *ctx.layout.feature_cols]
    if ctx.carry_go_terms:
        cols.append("go_term_id")
    cols.append("aspect")
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
    scan_cols_raw = ["protein_accession", "label", *ctx.feature_cols, "aspect"]
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
    bucket_idx = _bucket_array(accessions, ctx.bucket_count)
    for b in np.unique(bucket_idx):
        idx = np.flatnonzero(bucket_idx == b)
        idx_arr = pa.array(idx)
        cols_data: list[pa.Array] = [
            accessions.take(idx_arr),
            pa.array(labels[idx], type=pa.int8()),
            (aspects_col.take(idx_arr) if aspects_col is not None
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
class _WriteCtx:
    """Parameters for :func:`_pooled_write_splits` (reduces param count)."""

    multi_spec: "MultiManifestSpec"
    eval_source: "ManifestSource"
    scan: _PooledScanResult
    keep_mask: np.ndarray
    val_mask: np.ndarray
    feature_cols: list[str]
    categorical_cols: list[str]
    propagate: bool
    plan: StagePlan
    out_dir: Path
    propagation_stats: dict[str, int]


def _route_train_splits(
    ctx: _WriteCtx,
    layout: FeatureLayout,
    routing: BucketRouting,
    tmp_dir: Path,
) -> tuple[StagedSplit | None, StagedSplit | None]:
    """Route all sources and materialise train/val splits."""
    for i, src in enumerate(ctx.multi_spec.sources):
        train_pq = src.parquet_dir / "train.parquet"
        if not train_pq.exists():
            continue
        start, end = ctx.scan.source_offsets[i]
        _pass1_route_pooled_source(_Pass1PooledCtx(
            parquet_path=train_pq, src=src, layout=layout, routing=routing,
            keep_mask=ctx.keep_mask[start:end], val_mask=ctx.val_mask[start:end],
            override_labels=ctx.scan.labels0[start:end] if ctx.propagate else None,
            carry_go_terms=ctx.plan.carry_go_terms,
            snapshot_pairs=ctx.plan.train_snapshot_pairs, batch_size=ctx.plan.batch_size,
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


def _route_eval_split(
    ctx: _WriteCtx,
    eval_schema: pa.Schema,
    tmp_dir: Path,
) -> StagedSplit | None:
    """Route eval source and materialise eval split."""
    eval_source_pq = ctx.eval_source.parquet_dir / "eval.parquet"
    eval_override = (
        _eval_override_labels(
            source_eval_parquet=eval_source_pq,
            cat=None, asp=None, plan=ctx.plan,
            propagation_stats=ctx.propagation_stats,
        )
        if ctx.plan.parent_map_path is not None else None
    )
    eval_writers = _open_bucket_writers(tmp_dir, eval_schema, ctx.plan.bucket_count, "eval")
    _stream_eval_pooled(_EvalPooledCtx(
        eval_pq=eval_source_pq, src=ctx.eval_source,
        feature_cols=ctx.feature_cols, categorical_cols=ctx.categorical_cols,
        cat_codes=ctx.scan.global_cat_codes, bucket_count=ctx.plan.bucket_count,
        writers=eval_writers, schema=eval_schema,
        snapshot_pair=ctx.plan.eval_snapshot_pair, batch_size=ctx.plan.batch_size,
        override_labels=eval_override,
    ))
    return _materialise_split(
        bucket_writers=eval_writers, bucket_dir=tmp_dir,
        bucket_count=ctx.plan.bucket_count, out_dir=ctx.out_dir / "eval", name="eval",
    )


def _pooled_write_splits(
    ctx: _WriteCtx,
) -> tuple[StagedSplit | None, StagedSplit | None, StagedSplit | None]:
    """Open bucket writers, route all sources, materialise train/val/eval splits."""
    eval_schema = _build_pass1_schema(ctx.feature_cols, ctx.categorical_cols, carry_aspect=True)
    train_schema = _build_pass1_schema(
        ctx.feature_cols, ctx.categorical_cols,
        carry_go_terms=ctx.plan.carry_go_terms, carry_aspect=True,
    )
    with tempfile.TemporaryDirectory(prefix="staging_pooled_", dir=ctx.out_dir) as tmp:
        tmp_dir = Path(tmp)
        train_writers = _open_bucket_writers(tmp_dir, train_schema, ctx.plan.bucket_count, "train")
        val_writers = (
            _open_bucket_writers(tmp_dir, train_schema, ctx.plan.bucket_count, "val")
            if ctx.plan.val_strategy != "none" else None
        )
        layout = FeatureLayout(
            feature_cols=ctx.feature_cols, categorical_cols=ctx.categorical_cols,
            cat_codes=ctx.scan.global_cat_codes, schema=train_schema,
        )
        routing = BucketRouting(
            bucket_count=ctx.plan.bucket_count,
            train_writers=train_writers, val_writers=val_writers,
        )
        train_split, val_split = _route_train_splits(ctx, layout, routing, tmp_dir)
        eval_split = _route_eval_split(ctx, eval_schema, tmp_dir)
    return train_split, val_split, eval_split


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _build_stage_result(
    ctx: "_WriteCtx",
    train_split: StagedSplit,
    val_split: StagedSplit | None,
    eval_split: StagedSplit,
) -> StageResult:
    """Assemble StageResult and persist staging.json + propagation.json."""
    numeric_cols = [c for c in ctx.feature_cols if c not in set(ctx.categorical_cols)]
    result = StageResult(
        train=train_split, val=val_split, eval=eval_split,
        cat_codes=ctx.scan.global_cat_codes, feature_cols=ctx.feature_cols,
        numeric_cols=numeric_cols, categorical_cols=ctx.categorical_cols,
        n_proteins_train=int(train_split.n_groups),
        n_proteins_val=int(val_split.n_groups) if val_split else 0,
        n_proteins_eval=int(eval_split.n_groups), out_dir=ctx.out_dir,
    )
    if ctx.propagation_stats:
        (ctx.out_dir / "propagation.json").write_text(
            json.dumps(
                {"parent_map": str(ctx.plan.parent_map_path), **ctx.propagation_stats},
                indent=2,
            )
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

    Streams each source through shared bucket writers so no physical combined
    parquet is written (avoids write-OOM). ``plm_id`` and ``k_context`` are
    injected as constants per source. Eval uses ``eval_source`` only (prot_t5
    K10 preferred).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    propagate = plan.parent_map_path is not None
    scan = _pooled_pass0(multi_spec, categorical_cols=categorical_cols,
                         plan=plan, propagate=propagate)
    propagation_stats: dict[str, int] = {}
    if propagate:
        assert plan.parent_map_path is not None
        scan.labels0, propagation_stats = _propagate_train_labels(
            proteins0=scan.proteins0, labels0=scan.labels0,
            go_terms0=scan.go_terms0, parent_map_path=plan.parent_map_path,
        )
    keep_mask, val_mask, _ = _decide_split(
        proteins=scan.proteins0, labels=scan.labels0, pairs=scan.pairs0,
        val_strategy=plan.val_strategy, val_fraction=plan.val_fraction,
        val_holdout_snapshot=plan.val_holdout_snapshot,
        neg_pos_ratio=plan.neg_pos_ratio, seed=plan.seed,
    )
    keep_mask, _ = _cap_oversized_groups(
        proteins=scan.proteins0, labels=scan.labels0,
        keep_mask=keep_mask, val_mask=val_mask,
        max_group_size=LGBM_LAMBDARANK_MAX_GROUP, seed=plan.seed, aspects=scan.aspects0,
    )
    write_ctx = _WriteCtx(
        multi_spec=multi_spec, eval_source=eval_source, scan=scan,
        keep_mask=keep_mask, val_mask=val_mask,
        feature_cols=feature_cols, categorical_cols=categorical_cols,
        propagate=propagate, plan=plan, out_dir=out_dir,
        propagation_stats=propagation_stats,
    )
    train_split, val_split, eval_split = _pooled_write_splits(write_ctx)
    if eval_split is None:
        raise RuntimeError("eval split is empty for pooled staging")
    if train_split is None:
        raise RuntimeError("train split is empty for pooled staging")
    return _build_stage_result(write_ctx, train_split, val_split, eval_split)
