"""LightGBM model training and streaming inference.

Feature schema is imported from ``protea-contracts`` (the canonical source
of truth). Training is **streaming**: ``fit`` consumes :class:`lgb.Sequence`
instances (see :mod:`protea_reranker_lab.sequences`) plus pre-computed numpy
label / group arrays. No pandas DataFrame is materialised end-to-end.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import lightgbm as lgb
import numpy as np
from protea_contracts import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURE_FAMILIES,
)


@dataclass
class TrainConfig:
    objective: str = "lambdarank"
    num_boost_round: int = 5000
    early_stopping_rounds: int | None = 50
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
    # Palanca 1: Information-Accretion sample weighting. ``none`` keeps the
    # historical uniform-weight behaviour (the v26/v27-binary champion).
    ia_weighting: str = "none"
    ia_path: str | None = None
    ia_scale: float = 1.0

    def selected_features(self) -> list[str]:
        if self.enabled_feature_families is None:
            feats = list(ALL_FEATURES)
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


def fit(
    train: SplitArrays,
    val: SplitArrays | None,
    cfg: TrainConfig,
    *,
    feature_names: list[str],
    categorical_features: list[str],
) -> tuple[lgb.Booster, dict[str, Any]]:
    params = _lgb_params(cfg)

    train_ds = _make_dataset(
        train, cfg, feature_names=feature_names,
        categorical_features=categorical_features,
    )
    valid_sets = [train_ds]
    valid_names = ["train"]

    if val is not None and val.labels is not None and len(val.labels):
        val_ds = _make_dataset(
            val, cfg, feature_names=feature_names,
            categorical_features=categorical_features, reference=train_ds,
        )
        valid_sets.append(val_ds)
        valid_names.append("val")

    callbacks = []
    if cfg.early_stopping_rounds and len(valid_sets) > 1:
        callbacks.append(lgb.early_stopping(cfg.early_stopping_rounds, verbose=False))

    booster = lgb.train(
        params, train_ds,
        num_boost_round=cfg.num_boost_round,
        valid_sets=valid_sets, valid_names=valid_names,
        callbacks=callbacks,
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
