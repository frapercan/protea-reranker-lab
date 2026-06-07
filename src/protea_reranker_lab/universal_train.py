"""Training helpers for the universal multi-PLM reranker (F-RERANK-UNIVERSAL.5).

Internal module extracted from :mod:`universal_runner` to keep file LOC
within the smell budget. Not part of the public API.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .ia_weighting import ia_weights, load_ia_table
from .reranker import FitContext, SplitArrays, TrainConfig, fit, predict_streaming
from .runner import _build_train_config
from .sequences import ParquetFeatureSequence
from .pooled_staging import stage_for_training_pooled
from .staging import StagePlan

if TYPE_CHECKING:
    from .multi_source import MultiManifestSpec
    from .universal_runner import UniversalRunSpec


@dataclass
class TrainResult:
    """Output from a single-category training run."""

    booster: Any
    raw_scores: np.ndarray
    eval_labels: np.ndarray
    eval_groups: np.ndarray
    eval_proteins: np.ndarray
    best_iteration: int
    feature_importance: dict[str, float]
    eval_pq: Path
    primary_plm: str
    primary_k: int


@dataclass
class CategoryCtx:
    """Context bundle for train_category (keeps param count <=6)."""

    category: str
    multi_spec: "MultiManifestSpec"
    feature_cols: list[str]
    categorical_cols: list[str]
    spec: "UniversalRunSpec"
    staging_root: Path
    ia_table: dict | None
    out_dir: Path


def _pick_primary_source(multi_spec: "MultiManifestSpec") -> Any:
    """Prefer prot_t5 K10, else any K10, else first available source."""
    for src in multi_spec.sources:
        if src.plm_id == "prot_t5" and src.k_context == 10:
            return src
    for src in multi_spec.sources:
        if src.k_context == 10:
            return src
    return multi_spec.sources[0]


def _make_train_config(spec: "UniversalRunSpec") -> TrainConfig:
    """Build TrainConfig from UniversalRunSpec model_defaults."""
    from .experiment import ExperimentSpec, ModelSpec, SweepRef, TrainingSpec, DatasetRef

    defaults = {
        "objective": spec.model_defaults.get("objective", "lambdarank"),
        "num_boost_round": spec.model_defaults.get("num_boost_round", 3000),
        "early_stopping_rounds": spec.model_defaults.get("early_stopping_rounds", 50),
        "learning_rate": spec.model_defaults.get("learning_rate", 0.05),
        "num_leaves": spec.model_defaults.get("num_leaves", 63),
        "min_data_in_leaf": spec.model_defaults.get("min_data_in_leaf", 100),
        "feature_fraction": spec.model_defaults.get("feature_fraction", 0.9),
        "bagging_fraction": spec.model_defaults.get("bagging_fraction", 0.9),
        "neg_pos_ratio": spec.model_defaults.get("neg_pos_ratio", None),
        "val_fraction": spec.model_defaults.get("val_fraction", 0.2),
        "ia_weighting": spec.ia_weighting,
        "ia_feval_mode": spec.ia_feval_mode,
        "seed": spec.seed,
        "drop_features": [],
        "ia_path": str(spec.ia_path) if spec.ia_path else None,
    }
    experiment_spec = ExperimentSpec(
        name=spec.name,
        dataset=DatasetRef(manifest=Path(".")),
        model=ModelSpec(defaults=defaults),
        training=TrainingSpec(
            cell=f"{spec.train_cell}-all",
            val_strategy=spec.val_strategy,
            val_fraction=defaults["val_fraction"],
            val_holdout_snapshot=spec.val_holdout_snapshot,
        ),
        sweep=SweepRef(backend="none"),
    )
    return _build_train_config(experiment_spec)


def _build_ia_weights_pair(
    cfg: TrainConfig,
    ia_path_resolved: Path,
    stage: Any,
    train_labels: np.ndarray,
    ia_table: dict | None,
) -> tuple[np.ndarray | None, np.ndarray | None, dict | None]:
    """Return (train_weights, val_weights, ia_table)."""
    if cfg.ia_weighting == "none" or stage.train.go_terms_path is None:
        return None, None, ia_table
    if ia_table is None:
        ia_table = load_ia_table(ia_path_resolved)
    train_go = np.load(stage.train.go_terms_path, allow_pickle=True)
    train_w = ia_weights(train_go, train_labels, ia_table, mode=cfg.ia_weighting)
    val_w = None
    if (stage.val is not None and stage.val.go_terms_path is not None
            and stage.val.labels_path is not None):
        val_labels_arr = np.load(stage.val.labels_path)
        val_go = np.load(stage.val.go_terms_path, allow_pickle=True)
        val_w = ia_weights(val_go, val_labels_arr, ia_table, mode=cfg.ia_weighting)
    return train_w, val_w, ia_table


def _build_fit_ctx(
    cfg: TrainConfig,
    stage: Any,
    ia_table: dict | None,
    ia_path_resolved: Path,
    val_split: SplitArrays | None,
) -> FitContext:
    """Build the FitContext for the IA feval path."""
    if cfg.ia_feval_mode not in ("feval", "combined") or val_split is None:
        return FitContext()
    group_sizes = val_split.groups
    offsets = np.zeros(len(group_sizes), dtype=np.int64)
    if len(group_sizes) > 1:
        np.cumsum(group_sizes[:-1], out=offsets[1:])
    if (cfg.ia_weighting != "none" and stage.val is not None
            and stage.val.go_terms_path is not None):
        if ia_table is None:
            ia_table = load_ia_table(ia_path_resolved)
        val_go = np.load(stage.val.go_terms_path, allow_pickle=True)
        ia_flat = np.fromiter(
            (ia_table.get(str(g), 0.0) for g in val_go),
            dtype=np.float64, count=len(val_go),
        )
        edges = np.concatenate(([0], np.cumsum(group_sizes)))
        ia_per_group = np.array([
            float(ia_flat[edges[i]:edges[i + 1]].mean())
            for i in range(len(group_sizes))
        ], dtype=np.float64)
    else:
        ia_per_group = np.ones(len(group_sizes), dtype=np.float64)
    return FitContext(
        val_group_offsets=offsets,
        val_ia_per_group=ia_per_group,
        val_labels_flat=val_split.labels,
    )


def _build_splits(
    stage: Any,
    feature_cols: list[str],
    train_weights: np.ndarray | None,
    val_weights: np.ndarray | None,
) -> tuple[SplitArrays, SplitArrays | None]:
    """Construct SplitArrays from a staged StageResult."""
    train_labels = np.load(stage.train.labels_path)
    train_split = SplitArrays(
        seq=ParquetFeatureSequence(
            [str(p) for p in stage.train.bucket_paths], feature_cols
        ),
        labels=train_labels,
        groups=np.load(stage.train.groups_path),
        weights=train_weights,
    )
    val_split: SplitArrays | None = None
    if stage.val is not None and stage.val.n_rows > 0:
        val_labels = np.load(stage.val.labels_path)
        val_split = SplitArrays(
            seq=ParquetFeatureSequence(
                [str(p) for p in stage.val.bucket_paths], feature_cols
            ),
            labels=val_labels,
            groups=np.load(stage.val.groups_path),
            weights=val_weights,
        )
    return train_split, val_split


def _stage_category(ctx: CategoryCtx, cfg: TrainConfig) -> Any:
    """Stage train/eval data for all pooled sources. Returns stage result.

    Uses :func:`stage_for_training_pooled` to stream ALL 24 v226-lineage
    manifests (8 PLM x K{3,5,10}) through shared bucket writers without
    writing a physical combined parquet.  The eval split is taken from the
    primary source (prot_t5 K10 or best available).
    """
    primary_src = _pick_primary_source(ctx.multi_spec)
    eval_pq = primary_src.parquet_dir / "eval.parquet"
    if not eval_pq.exists():
        raise FileNotFoundError(
            f"Primary source for '{ctx.category}' missing eval parquet: "
            f"{primary_src.parquet_dir}"
        )
    plan = StagePlan(
        val_strategy=ctx.spec.val_strategy,
        val_fraction=cfg.val_fraction,
        val_holdout_snapshot=ctx.spec.val_holdout_snapshot,
        neg_pos_ratio=cfg.neg_pos_ratio,
        seed=cfg.seed,
        carry_go_terms=(cfg.ia_weighting != "none"),
        aspect_conditioned=True,
    )
    stage_dir = ctx.staging_root / ctx.category
    cat_cols = [c for c in ctx.categorical_cols if c in set(ctx.feature_cols)]
    return stage_for_training_pooled(
        multi_spec=ctx.multi_spec,
        eval_source=primary_src,
        feature_cols=ctx.feature_cols,
        categorical_cols=cat_cols,
        out_dir=stage_dir,
        plan=plan,
    )


def _fit_and_predict(
    ctx: CategoryCtx,
    cfg: TrainConfig,
    stage: Any,
    ia_path_resolved: Path,
) -> tuple[Any, dict, np.ndarray]:
    """Fit booster and predict eval. Returns (booster, train_metrics, raw_scores)."""
    cat_cols = [c for c in ctx.categorical_cols if c in set(ctx.feature_cols)]
    train_labels = np.load(stage.train.labels_path)
    train_w, val_w, _ = _build_ia_weights_pair(
        cfg, ia_path_resolved, stage, train_labels, ctx.ia_table
    )
    train_split, val_split = _build_splits(stage, ctx.feature_cols, train_w, val_w)
    fit_ctx = _build_fit_ctx(cfg, stage, ctx.ia_table, ia_path_resolved, val_split)
    booster, train_metrics = fit(
        train_split, val_split, cfg,
        feature_names=ctx.feature_cols,
        categorical_features=cat_cols,
        fit_ctx=fit_ctx,
    )
    eval_seq = ParquetFeatureSequence(
        [str(p) for p in stage.eval.bucket_paths], ctx.feature_cols
    )
    raw_scores = predict_streaming(booster, eval_seq)
    return booster, train_metrics, raw_scores


def train_category(ctx: CategoryCtx) -> TrainResult:
    """Stage + fit one category. Returns TrainResult with booster + predictions."""
    primary_src = _pick_primary_source(ctx.multi_spec)
    eval_pq = primary_src.parquet_dir / "eval.parquet"
    cfg = _make_train_config(ctx.spec)
    ia_path_resolved = (
        ctx.spec.ia_path or Path(__file__).resolve().parents[2]
        / "datasets" / "ia" / "IA-swissprot-exp-v227.txt"
    )
    stage = _stage_category(ctx, cfg)
    booster, train_metrics, raw_scores = _fit_and_predict(ctx, cfg, stage, ia_path_resolved)
    eval_labels = np.load(stage.eval.labels_path)
    eval_groups = np.load(stage.eval.groups_path)
    eval_proteins = np.load(stage.eval.proteins_path, allow_pickle=True)
    shutil.rmtree(ctx.staging_root / ctx.category, ignore_errors=True)
    fi = {k: float(v) for k, v in sorted(
        train_metrics.get("feature_importance", {}).items(), key=lambda kv: -kv[1]
    )}
    return TrainResult(
        booster=booster, raw_scores=raw_scores, eval_labels=eval_labels,
        eval_groups=eval_groups, eval_proteins=eval_proteins,
        best_iteration=int(train_metrics.get("best_iteration", 0)),
        feature_importance=fi, eval_pq=eval_pq,
        primary_plm=primary_src.plm_id, primary_k=primary_src.k_context,
    )
