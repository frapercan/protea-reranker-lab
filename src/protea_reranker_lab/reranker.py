"""LightGBM model training and streaming inference.

Feature schema is imported from ``protea-contracts`` (the canonical source
of truth). Training is **streaming**: ``fit`` consumes :class:`lgb.Sequence`
instances (see :mod:`protea_reranker_lab.sequences`) plus pre-computed numpy
label / group arrays. No pandas DataFrame is materialised end-to-end.

IA-weighted LambdaMART (F-RERANK-UNIVERSAL.4)
----------------------------------------------
Three IA integration mechanisms are supported and compared via bake-off:

``ia_feval_mode="none"``
    No custom feval; use standard NDCG/MAP from LightGBM.  Baseline.

``ia_feval_mode="weight"``
    Per-row sample weights proportional to IA (via ``ia_weighting`` +
    ``ia_scale``).  The booster objective stays ``lambdarank``; weights
    re-price each row's contribution but the internal ranking loss remains
    NDCG-style.  This is "palanca 1" from prior slices extended to the
    universal model.

``ia_feval_mode="feval"``
    A custom LightGBM ``feval`` that computes IA-weighted f_micro_w on the
    (protein, aspect) groups provided in ``FitContext`` (see below).  The
    feval is called by LightGBM on the val split at every boosting round and
    can drive early stopping.  ``objective`` stays ``lambdarank``; feval is
    supplementary.

``ia_feval_mode="combined"``
    Both per-row weights AND the custom feval active simultaneously.  This is
    the LIKELY-CORRECT mechanism per the acceptance criteria.

The feval is implemented as a closure over per-group IA arrays so no global
state is needed.  Because LightGBM's feval receives a flat score array (not
grouped), the function recomputes group boundaries from a precomputed
``group_offsets`` vector supplied by the caller via :class:`FitContext`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import lightgbm as lgb
import numpy as np
from protea_contracts import (
    CATEGORICAL_FEATURES,
    FEATURE_FAMILIES,
)

from .contracts import DEFAULT_TRAINING_FEATURES


@dataclass
class TrainConfig:
    objective: str = "lambdarank"
    num_boost_round: int = 5000
    early_stopping_rounds: int | None = 50
    # Period (in boosting rounds) for lgb.log_evaluation; 0/None disables it.
    # Surfaces the tree phase instead of leaving it a black box.
    log_period: int | None = 50
    learning_rate: float = 0.05
    num_leaves: int = 63
    min_data_in_leaf: int = 100
    feature_fraction: float = 0.9
    bagging_fraction: float = 0.9
    bagging_freq: int = 5
    neg_pos_ratio: float | None = None
    val_fraction: float = 0.2
    seed: int = 42
    enabled_feature_families: list[str] | None = None
    drop_features: list[str] = field(default_factory=list)
    # Combiner mode (MR-2): when set, the booster trains over EXACTLY this
    # explicit, ordered column list (the small score vector) instead of the
    # contracts feature families. Bypasses ``enabled_feature_families`` /
    # ``FEATURE_FAMILIES`` resolution entirely; ``drop_features`` still applies
    # so an operator can ablate one score from the vector. ``None`` keeps the
    # historical monolith behaviour byte-for-byte.
    feature_override: list[str] | None = None
    # Palanca 1: Information-Accretion sample weighting. ``none`` keeps the
    # historical uniform-weight behaviour (the v26/v27-binary champion).
    ia_weighting: str = "none"
    ia_path: str | None = None
    ia_scale: float = 1.0
    # IA feval mechanism (F-RERANK-UNIVERSAL.4). One of:
    #   "none"     - standard NDCG/MAP feval (baseline)
    #   "weight"   - per-row weights only (palanca 1 extended)
    #   "feval"    - custom f_micro_w feval on (protein,aspect) groups
    #   "combined" - both weight + feval (recommended)
    ia_feval_mode: str = "none"
    # K-augmentation policy for training (F-RERANK-UNIVERSAL.4).
    # k_aug_seed: RNG seed for bounded K sampling across pooled sources.
    # k_aug_bounds: (k_min, k_max) inclusive; None = no augmentation.
    # k_inference_policy: "fixed" uses the full pool at inference; "adaptive"
    #   selects K by protein coverage. Captured in ExperimentSpec.hash().
    k_aug_seed: int = 42
    k_aug_bounds: tuple[int, int] | None = None
    k_inference_policy: str = "fixed"

    def selected_features(self) -> list[str]:
        if self.feature_override is not None:
            # Combiner mode: train over exactly the supplied score-vector
            # columns, in order. These are explicit raw parquet columns that may
            # NOT live in the contracts ALL_FEATURES schema (the score-vector
            # signals ride GOPrediction.features per PROTEA #643); staging reads
            # any named parquet column directly, so no contracts-family lookup.
            feats = list(self.feature_override)
        elif self.enabled_feature_families is None:
            # The lab default, not the contracts catalogue. See
            # UNADOPTED_FEATURE_FAMILIES for what is held out and why.
            feats = list(DEFAULT_TRAINING_FEATURES)
        else:
            feats = []
            for fam in self.enabled_feature_families:
                feats.extend(FEATURE_FAMILIES[fam])
        return [f for f in feats if f not in self.drop_features]

    def split_features(self) -> tuple[list[str], list[str]]:
        feats = self.selected_features()
        cat_set = set(CATEGORICAL_FEATURES)
        numeric = [f for f in feats if f not in cat_set]
        categorical = [f for f in feats if f in cat_set]
        return numeric, categorical


@dataclass
class SplitArrays:
    """One LightGBM split: feature sequence(s) + row-aligned label/group/weight."""

    seq: "lgb.Sequence | list[lgb.Sequence] | None"
    labels: np.ndarray | None
    groups: np.ndarray | None = None
    weights: np.ndarray | None = None
    # Per-group IA arrays for the feval closure (F-RERANK-UNIVERSAL.4).
    # ia_per_group[i] is the mean IA value for group i (one element per
    # LambdaRank group, NOT per row); used by the custom feval to weight
    # the per-group contribution to f_micro_w.  None = uniform weighting.
    ia_per_group: np.ndarray | None = None


@dataclass
class FitContext:
    """Extra arrays needed by the IA-feval closure (F-RERANK-UNIVERSAL.4).

    ``val_group_offsets`` is a 1-D int64 array of start positions for each
    val group in the flat score/label arrays LightGBM passes to feval.
    ``val_ia_per_group`` is the mean IA for each group (used to weight the
    per-group precision/recall contribution so the feval mirrors the
    ``f_micro_w`` cafaeval metric).
    ``val_labels_flat`` is the ground-truth label array aligned to the flat
    score array passed by LightGBM.

    All three are ``None`` when the feval is disabled.
    """

    val_group_offsets: np.ndarray | None = None
    val_ia_per_group: np.ndarray | None = None
    val_labels_flat: np.ndarray | None = None


def _lgb_params(cfg: TrainConfig) -> dict[str, Any]:
    params: dict[str, Any] = {
        "objective": cfg.objective,
        "learning_rate": cfg.learning_rate,
        "num_leaves": cfg.num_leaves,
        "min_data_in_leaf": cfg.min_data_in_leaf,
        "feature_fraction": cfg.feature_fraction,
        "bagging_fraction": cfg.bagging_fraction,
        "bagging_freq": cfg.bagging_freq,
        "seed": cfg.seed,
        "verbose": -1,
    }
    if cfg.objective == "lambdarank":
        params["metric"] = ["ndcg", "map"]
        params["ndcg_eval_at"] = [5, 10]
        params["label_gain"] = [0, 1]
    return params


def _make_dataset(
    split: SplitArrays,
    cfg: TrainConfig,
    *,
    feature_names: list[str],
    categorical_features: list[str],
    reference: "lgb.Dataset | None" = None,
) -> lgb.Dataset:
    seq_list = split.seq if isinstance(split.seq, list) else [split.seq]
    return lgb.Dataset(
        seq_list,
        label=split.labels,
        group=split.groups if cfg.objective == "lambdarank" else None,
        weight=split.weights,
        feature_name=feature_names,
        categorical_feature=categorical_features or "auto",
        reference=reference,
        free_raw_data=True,
    )


def _build_ia_feval(
    group_offsets: np.ndarray,
    ia_per_group: np.ndarray,
    labels_flat: np.ndarray,
    *,
    n_thresholds: int = 51,
) -> Any:
    """Return a LightGBM-compatible feval closure for IA-weighted f_micro_w.

    The closure captures ``group_offsets``, ``ia_per_group``, and
    ``labels_flat`` from the val split so LightGBM can call it without
    any extra state.  It returns ``("ia_f_micro_w", value, is_higher_better)``.

    ``group_offsets`` must be a sorted int64 array of start positions for each
    group (length == n_groups); the sentinel at the end is appended
    internally.
    """
    offsets = np.asarray(group_offsets, dtype=np.int64)
    ia = np.asarray(ia_per_group, dtype=np.float64)
    gt = np.asarray(labels_flat, dtype=np.int8)
    n_rows = len(gt)

    def _feval(preds: np.ndarray, _dataset: lgb.Dataset) -> tuple[str, float, bool]:
        if len(preds) != n_rows or len(offsets) == 0:
            return ("ia_f_micro_w", 0.0, True)
        ends = np.append(offsets[1:], n_rows)
        lo = float(preds.min())
        hi = float(preds.max())
        if hi <= lo:
            return ("ia_f_micro_w", 0.0, True)
        thresholds = np.linspace(lo, hi, n_thresholds)
        best = 0.0
        for t in thresholds:
            pred_pos = preds >= t
            ia_prec_sum = 0.0
            ia_rec_sum = 0.0
            ia_sum = ia.sum()
            if ia_sum == 0.0:
                ia_sum = 1.0
            for i, (start, end) in enumerate(zip(offsets, ends)):
                grp_pred = pred_pos[start:end]
                grp_gt = gt[start:end]
                tp = int((grp_pred & (grp_gt > 0)).sum())
                fp = int((grp_pred & (grp_gt == 0)).sum())
                fn = int((~grp_pred & (grp_gt > 0)).sum())
                w = float(ia[i])
                if tp + fp > 0:
                    ia_prec_sum += w * tp / (tp + fp)
                if tp + fn > 0:
                    ia_rec_sum += w * tp / (tp + fn)
            mean_p = ia_prec_sum / ia_sum
            mean_r = ia_rec_sum / ia_sum
            if mean_p + mean_r > 0:
                f = 2 * mean_p * mean_r / (mean_p + mean_r)
                if f > best:
                    best = f
        return ("ia_f_micro_w", float(best), True)

    return _feval


def _select_feval(
    cfg: TrainConfig,
    ctx: FitContext,
    n_valid_sets: int,
) -> Any:
    """Return the IA feval closure or None based on ``cfg.ia_feval_mode``."""
    if cfg.ia_feval_mode not in ("feval", "combined") or n_valid_sets <= 1:
        return None
    if (
        ctx.val_group_offsets is None
        or ctx.val_ia_per_group is None
        or ctx.val_labels_flat is None
    ):
        return None
    return _build_ia_feval(
        ctx.val_group_offsets, ctx.val_ia_per_group, ctx.val_labels_flat,
    )


def _build_train_runtime(
    cfg: TrainConfig,
    n_valid_sets: int,
) -> list[Callable[..., Any]]:
    """Resolve early-stopping/logging callbacks for ``lgb.train``.

    Preserves the established early-stopping behaviour (track the built-in
    lambdarank metrics on the held-out ``val`` set). A ``log_evaluation``
    callback surfaces the tree phase per ``cfg.log_period`` so the boosting
    run is observable instead of a black box.
    """
    callbacks: list[Callable[..., Any]] = []
    if cfg.early_stopping_rounds and n_valid_sets > 1:
        callbacks.append(
            lgb.early_stopping(cfg.early_stopping_rounds, verbose=False)
        )
    if cfg.log_period and cfg.log_period > 0:
        callbacks.append(lgb.log_evaluation(period=cfg.log_period))
    return callbacks


def fit(
    train: SplitArrays,
    val: SplitArrays | None,
    cfg: TrainConfig,
    *,
    feature_names: list[str],
    categorical_features: list[str],
    fit_ctx: FitContext | None = None,
) -> tuple[lgb.Booster, dict[str, Any]]:
    params = _lgb_params(cfg)
    train_ds = _make_dataset(
        train, cfg, feature_names=feature_names,
        categorical_features=categorical_features,
    )
    valid_sets: list[lgb.Dataset] = [train_ds]
    valid_names = ["train"]
    if val is not None and val.labels is not None and len(val.labels):
        valid_sets.append(_make_dataset(
            val, cfg, feature_names=feature_names,
            categorical_features=categorical_features, reference=train_ds,
        ))
        valid_names.append("val")

    ctx = fit_ctx or FitContext()
    feval = _select_feval(cfg, ctx, len(valid_sets))
    callbacks = _build_train_runtime(cfg, len(valid_sets))

    booster = lgb.train(
        params, train_ds,
        num_boost_round=cfg.num_boost_round,
        valid_sets=valid_sets, valid_names=valid_names,
        callbacks=callbacks,
        feval=feval,
    )
    metrics = {
        "best_iteration": booster.best_iteration or booster.current_iteration(),
        "feature_importance": dict(zip(
            booster.feature_name(),
            booster.feature_importance(importance_type="gain"),
            strict=False,
        )),
    }
    return booster, metrics


def predict_streaming(
    booster: lgb.Booster,
    eval_seq: lgb.Sequence,
    *,
    batch_size: int = 100_000,
) -> np.ndarray:
    """Score the eval Sequence in batches; returns one float32 per row."""
    n = len(eval_seq)
    out = np.empty(n, dtype=np.float32)
    if n == 0:
        return out
    n_iter = booster.best_iteration or booster.current_iteration()
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        x = eval_seq[start:stop]
        out[start:stop] = booster.predict(x, num_iteration=n_iter)
    return out
