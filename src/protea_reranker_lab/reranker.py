"""Feature schema + LightGBM fit, mirroring PROTEA's reranker.py exactly.

Kept as a standalone module (not a git submodule of PROTEA) to keep this repo
installable without the PROTEA dependency tree. The schema is version-pinned
to PROTEA commit tagged in the dataset manifest — if PROTEA's feature set
changes, bump the schema here and regenerate the dump.

Training is **streaming**: ``fit`` consumes :class:`lgb.Sequence` instances
(see :mod:`protea_reranker_lab.sequences`) plus pre-computed numpy label /
group arrays. No pandas DataFrame is materialised end-to-end.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import lightgbm as lgb
import numpy as np


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
    "knn_distance": ["distance", "neighbor_min_distance", "neighbor_mean_distance",
                     "neighbor_distance_std"],
    "knn_vote": ["k_position", "vote_count", "neighbor_vote_fraction"],
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


def fit(
    train_seq: lgb.Sequence | list[lgb.Sequence],
    train_labels: np.ndarray,
    train_groups: np.ndarray | None,
    val_seq: lgb.Sequence | list[lgb.Sequence] | None,
    val_labels: np.ndarray | None,
    val_groups: np.ndarray | None,
    cfg: TrainConfig,
    *,
    feature_names: list[str],
    categorical_features: list[str],
) -> tuple[lgb.Booster, dict[str, Any]]:
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

    train_seq_list = train_seq if isinstance(train_seq, list) else [train_seq]
    train_ds = lgb.Dataset(
        train_seq_list,
        label=train_labels,
        group=train_groups if cfg.objective == "lambdarank" else None,
        feature_name=feature_names,
        categorical_feature=categorical_features or "auto",
        free_raw_data=True,
    )
    valid_sets = [train_ds]
    valid_names = ["train"]

    if val_seq is not None and val_labels is not None and len(val_labels):
        val_seq_list = val_seq if isinstance(val_seq, list) else [val_seq]
        val_ds = lgb.Dataset(
            val_seq_list,
            label=val_labels,
            group=val_groups if cfg.objective == "lambdarank" else None,
            feature_name=feature_names,
            categorical_feature=categorical_features or "auto",
            reference=train_ds,
            free_raw_data=True,
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
