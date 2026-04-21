"""Feature schema + LightGBM fit, mirroring PROTEA's reranker.py exactly.

Kept as a standalone module (not a git submodule of PROTEA) to keep this repo
installable without the PROTEA dependency tree. The schema is version-pinned
to PROTEA commit tagged in the dataset manifest — if PROTEA's feature set
changes, bump the schema here and regenerate the dump.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

NUMERIC_FEATURES: list[str] = [
    "distance",
    "identity_nw", "similarity_nw", "alignment_score_nw", "gaps_pct_nw", "alignment_length_nw",
    "identity_sw", "similarity_sw", "alignment_score_sw", "gaps_pct_sw", "alignment_length_sw",
    "length_query", "length_ref",
    "taxonomic_distance", "taxonomic_common_ancestors",
    "vote_count", "k_position", "go_term_frequency", "ref_annotation_density",
    "neighbor_distance_std", "neighbor_vote_fraction",
    "neighbor_min_distance", "neighbor_mean_distance",
    "anc2vec_neighbor_cos", "anc2vec_neighbor_maxcos", "anc2vec_has_emb",
    "anc2vec_query_known_cos", "anc2vec_query_known_maxcos", "anc2vec_query_known_count",
    "tax_voters_same_frac", "tax_voters_close_frac", "tax_voters_mean_common_ancestors",
    *[f"emb_pca_query_{i}" for i in range(16)],
]

CATEGORICAL_FEATURES: list[str] = [
    "qualifier", "evidence_code", "taxonomic_relation", "aspect",
]

ALL_FEATURES: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

FEATURE_FAMILIES: dict[str, list[str]] = {
    "knn": ["distance", "k_position", "vote_count", "neighbor_vote_fraction",
            "neighbor_min_distance", "neighbor_mean_distance", "neighbor_distance_std"],
    "alignment_nw": ["identity_nw", "similarity_nw", "alignment_score_nw",
                     "gaps_pct_nw", "alignment_length_nw"],
    "alignment_sw": ["identity_sw", "similarity_sw", "alignment_score_sw",
                     "gaps_pct_sw", "alignment_length_sw"],
    "length": ["length_query", "length_ref"],
    "taxonomy_pair": ["taxonomic_distance", "taxonomic_common_ancestors", "taxonomic_relation"],
    "taxonomy_voters": ["tax_voters_same_frac", "tax_voters_close_frac",
                        "tax_voters_mean_common_ancestors"],
    "go_context": ["go_term_frequency", "ref_annotation_density"],
    "anc2vec_neighbor": ["anc2vec_neighbor_cos", "anc2vec_neighbor_maxcos", "anc2vec_has_emb"],
    "anc2vec_query": ["anc2vec_query_known_cos", "anc2vec_query_known_maxcos",
                      "anc2vec_query_known_count"],
    "emb_pca": [f"emb_pca_query_{i}" for i in range(16)],
    "annotation_meta": ["qualifier", "evidence_code", "aspect"],
}


@dataclass
class TrainConfig:
    objective: str = "lambdarank"            # or "binary"
    num_boost_round: int = 5000
    early_stopping_rounds: int | None = 50
    learning_rate: float = 0.05
    num_leaves: int = 63
    min_data_in_leaf: int = 100
    feature_fraction: float = 0.9
    bagging_fraction: float = 0.9
    bagging_freq: int = 5
    neg_pos_ratio: float | None = None        # downsample negatives
    val_fraction: float = 0.2
    seed: int = 42
    enabled_feature_families: list[str] | None = None  # None = use all
    drop_features: list[str] = field(default_factory=list)

    def selected_features(self) -> list[str]:
        if self.enabled_feature_families is None:
            feats = list(ALL_FEATURES)
        else:
            feats = []
            for fam in self.enabled_feature_families:
                feats.extend(FEATURE_FAMILIES[fam])
        return [f for f in feats if f not in self.drop_features]


def encode_categoricals(df: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    """Label-encode string categoricals in place (missing → -1)."""
    out = df.copy()
    for col in cat_cols:
        if col not in out.columns:
            continue
        s = out[col].astype("object").where(out[col].notna(), None)
        codes, _ = pd.factorize(s, use_na_sentinel=True)
        out[col] = codes  # int64
    return out


def prepare_matrix(
    df: pd.DataFrame, cfg: TrainConfig
) -> tuple[pd.DataFrame, list[str], list[str]]:
    feats = cfg.selected_features()
    used_num = [f for f in feats if f in NUMERIC_FEATURES and f in df.columns]
    used_cat = [f for f in feats if f in CATEGORICAL_FEATURES and f in df.columns]
    X = encode_categoricals(df[used_num + used_cat], used_cat)
    for col in used_num:
        if X[col].dtype == object:
            X[col] = pd.to_numeric(X[col], errors="coerce")
    return X, used_num, used_cat


def fit(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame | None,
    cfg: TrainConfig,
    *,
    group_col: str = "protein_accession",
) -> tuple[lgb.Booster, dict[str, Any]]:
    X_tr, used_num, used_cat = prepare_matrix(df_train, cfg)
    y_tr = df_train["label"].to_numpy()

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

    train_groups = _groups(df_train, group_col) if cfg.objective == "lambdarank" else None
    train_ds = lgb.Dataset(
        X_tr, label=y_tr, group=train_groups,
        categorical_feature=used_cat, free_raw_data=False,
    )
    valid_sets = [train_ds]
    valid_names = ["train"]
    if df_val is not None and len(df_val):
        X_va, _, _ = prepare_matrix(df_val, cfg)
        y_va = df_val["label"].to_numpy()
        val_groups = _groups(df_val, group_col) if cfg.objective == "lambdarank" else None
        val_ds = lgb.Dataset(
            X_va, label=y_va, group=val_groups,
            categorical_feature=used_cat, reference=train_ds, free_raw_data=False,
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
            booster.feature_name(), booster.feature_importance(importance_type="gain"), strict=False
        )),
    }
    return booster, metrics


def _groups(df: pd.DataFrame, group_col: str) -> np.ndarray:
    """LightGBM expects ``group`` as contiguous sizes. Requires df sorted by group_col."""
    sizes = df.groupby(group_col, sort=False).size().to_numpy()
    return sizes


def predict(booster: lgb.Booster, df: pd.DataFrame, cfg: TrainConfig) -> np.ndarray:
    X, _, _ = prepare_matrix(df, cfg)
    feat_names = booster.feature_name()
    X = X[[c for c in feat_names if c in X.columns]]
    return booster.predict(X, num_iteration=booster.best_iteration or booster.current_iteration())
