"""Train the 3 per-category (NK/LK/PK) LightGBM boosters for the NATIVE
PROTEA reranker pipeline.

These boosters run inside PROTEA's live ``apply_reranker`` predict path, so
they are trained on the parity export parquet (the full numeric feature schema
incl. the classifier / self_prior / association columns) rather than on the
offline 16-feature composite. They REPLACE that offline composite, which the
predict-time ``SchemaShaMismatchError`` guard rejects.

Memory-safe by construction: a parity ``train.parquet`` is ~54M rows x ~78
cols, so a naive ``ParquetFile.read().to_pandas()`` blows past the box RAM.
Here we stream row-group batches from a LOCAL parquet file, keep only the
needed feature columns as ``float32``, and split into per-category
accumulators so peak RSS stays well under the box. Binary objective (the
export's ``reranker_objective`` default), early-stopping on the eval split.

The trained boosters are saved as ``ensemble_gbm_{NK,LK,PK}.txt`` alongside a
``summary.json`` carrying the ``feature_schema_sha`` the live predict guard
expects.

This is the lab-native home of what used to be the ad-hoc
``storage/fullgo_models/train_native_boosters.py``. It depends only on
``protea-contracts`` (a declared lab dependency) for the feature schema, never
on PROTEA / protea-method directly, so it stays installable as a thin lab dev
dependency.
"""

from __future__ import annotations

import contextlib
import gc
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import lightgbm as lgb
import numpy as np
import pyarrow.parquet as pq
from protea_contracts import (
    CATEGORICAL_FEATURES,
    compute_feature_schema_sha,
)

from .contracts import DEFAULT_TRAINING_FEATURES

if TYPE_CHECKING:
    from .native_boosters_mlflow import MlflowLogger

log = logging.getLogger(__name__)

CATEGORIES: tuple[str, ...] = ("nk", "lk", "pk")

#: LightGBM params mirroring the offline ad-hoc script (binary objective, the
#: export's ``reranker_objective`` default; AUC metric for early stopping).
DEFAULT_PARAMS: dict[str, float | int | str] = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "verbosity": -1,
}
DEFAULT_NUM_BOOST_ROUND = 5000
DEFAULT_EARLY_STOP = 50


def infer_active_feature_families(
    *,
    compute_alignments: bool = True,
    compute_taxonomy: bool = True,
    compute_v6_features: bool = False,
) -> list[str]:
    """Return the sorted lab feature families active under the given flags.

    Mirrors ``protea_method.reranker.infer_active_feature_families`` so the
    lab can compute the predict-time ``feature_schema_sha`` without importing
    PROTEA. Keep in sync with that function and with
    ``protea_reranker_lab.contracts.FEATURE_FAMILIES``.
    """
    families: list[str] = ["knn", "annotation_meta"]
    if compute_alignments:
        families += ["alignment_nw", "length"]
    if compute_taxonomy:
        families.append("taxonomy_pair")
    if compute_v6_features:
        families += [
            "anc2vec_neighbor",
            "anc2vec_query",
            "emb_pca",
            "taxonomy_voters",
            "go_context",
        ]
    return sorted(set(families))


@dataclass
class NativeBoosterConfig:
    """Inputs for one native-booster training job.

    ``train_parquet`` / ``eval_parquet`` are LOCAL paths (pulled beforehand
    via ``scripts/pull_dataset.py`` or a direct MinIO fetch). Streaming reads
    from disk keep peak RAM bounded, unlike loading the whole object first.
    """

    train_parquet: Path
    eval_parquet: Path
    out_dir: Path
    dataset_name: str = ""
    features: list[str] | None = None  # explicit numeric subset (MR-2 flat combiner)
    params: dict[str, float | int | str] = field(
        default_factory=lambda: dict(DEFAULT_PARAMS)
    )
    num_boost_round: int = DEFAULT_NUM_BOOST_ROUND
    early_stopping_rounds: int = DEFAULT_EARLY_STOP
    batch_rows: int = 500_000


def _to_float(col: list) -> np.ndarray:
    """Numeric column -> float32, coercing ''/None to NaN (slow per-row path)."""
    arr = np.asarray(col, dtype=object)
    out = np.empty(len(arr), dtype=np.float32)
    for i, value in enumerate(arr):
        try:
            out[i] = float(value)
        except (TypeError, ValueError):
            out[i] = np.nan
    return out


def resolve_numeric_features(parquet_path: Path, features: list[str] | None) -> list[str]:
    """Pick the numeric feature columns to train on (categoricals dropped).

    The generic ``apply_reranker`` predict path coerces string categoricals to
    NaN, so training on them would create a train/predict mismatch. When an
    explicit ``features`` subset is given (e.g. the MR-2 flat combiner over the
    component-score vector) it is intersected with the columns actually present.
    """
    present = set(pq.read_schema(str(parquet_path)).names)
    cat_set = set(CATEGORICAL_FEATURES)
    if features:
        return [f for f in features if f.strip() and f in present]
    return [
        f for f in DEFAULT_TRAINING_FEATURES if f in present and f not in cat_set
    ]


def feature_schema_sha() -> str:
    """The predict-time schema sha the live guard expects (alignments+taxonomy)."""
    return compute_feature_schema_sha(infer_active_feature_families())


def _accumulate(parquet_path: Path, feat: list[str], batch_rows: int) -> dict[str, tuple]:
    """Stream row groups; return ``{cat: (X float32, y int8)}`` for the file.

    ``feat`` is NUMERIC-only. Memory stays bounded: each batch is cast to
    float32, split per category, and freed before the next one is read.
    """
    cols = feat + ["label", "category"]
    parts: dict[str, list] = {c: [] for c in CATEGORIES}
    ys: dict[str, list] = {c: [] for c in CATEGORIES}
    pf = pq.ParquetFile(str(parquet_path))
    seen = 0
    for batch in pf.iter_batches(batch_size=batch_rows, columns=cols):
        rec = batch.to_pydict()
        cat = np.asarray([str(x) for x in rec["category"]])
        lab = np.asarray(rec["label"], dtype=np.int8)
        xb = np.empty((len(cat), len(feat)), dtype=np.float32)
        for j, name in enumerate(feat):
            col = rec[name]
            try:
                xb[:, j] = np.asarray(col, dtype=np.float32)
            except (TypeError, ValueError):
                xb[:, j] = _to_float(col)
        for c in CATEGORIES:
            mask = cat == c
            if mask.any():
                parts[c].append(xb[mask])
                ys[c].append(lab[mask])
        seen += len(cat)
        del rec, xb, cat, lab
        gc.collect()
        log.info("    ...%d rows streamed from %s", seen, parquet_path.name)
    out: dict[str, tuple] = {}
    for c in CATEGORIES:
        if parts[c]:
            out[c] = (np.vstack(parts[c]), np.concatenate(ys[c]))
        parts[c] = []
    gc.collect()
    return out


def _train_one_category(
    cat: str,
    train: dict[str, tuple],
    ev: dict[str, tuple],
    cfg: NativeBoosterConfig,
    callbacks_extra: list | None,
) -> tuple[lgb.Booster, dict]:
    """Train a single per-category booster (eval-only valid set)."""
    feat = cfg.features or []  # only used for feature_name below
    x_tr, y_tr = train[cat]
    d_tr = lgb.Dataset(x_tr, label=y_tr, feature_name=feat or "auto", free_raw_data=False)
    # eval-only: do NOT put the (up to ~50M-row) train set in valid_sets.
    # Computing train-AUC every iteration dominates PK runtime and the trained
    # booster is identical either way; early stopping only needs the eval set.
    valid, names = [], []
    callbacks = [lgb.log_evaluation(period=50), *(callbacks_extra or [])]
    if cat in ev:
        x_ev, y_ev = ev[cat]
        d_ev = lgb.Dataset(
            x_ev, label=y_ev, feature_name=feat or "auto", reference=d_tr, free_raw_data=False
        )
        valid.append(d_ev)
        names.append("eval")
        callbacks.append(lgb.early_stopping(cfg.early_stopping_rounds, verbose=True))
    booster = lgb.train(
        cfg.params, d_tr,
        num_boost_round=cfg.num_boost_round,
        valid_sets=valid, valid_names=names, callbacks=callbacks,
    )
    best = booster.best_score.get("eval", {}).get("auc") if cat in ev else None
    info = {
        "train_rows": int(len(y_tr)),
        "pos": int(y_tr.sum()),
        "best_iter": int(booster.best_iteration),
        "eval_auc": best,
    }
    return booster, info


def _train_and_save_category(
    cat: str,
    train: dict[str, tuple],
    ev: dict[str, tuple],
    cfg: NativeBoosterConfig,
    mlflow_logger: "MlflowLogger | None",
) -> dict:
    """Train one category booster, save it, and return its summary entry."""
    cb_extra = mlflow_logger.metric_callback(cat) if mlflow_logger else None
    with (mlflow_logger.category_run(cat) if mlflow_logger else contextlib.nullcontext()):
        if mlflow_logger:
            _, y_tr = train[cat]
            mlflow_logger.log_category_params(cat, int(len(y_tr)), int(y_tr.sum()))
        booster, info = _train_one_category(cat, train, ev, cfg, cb_extra)
        path = cfg.out_dir / f"ensemble_gbm_{cat.upper()}.txt"
        booster.save_model(str(path))
        info["path"] = str(path)
        log.info(
            "  %s: train=%d pos=%d best_iter=%d eval_auc=%s -> %s",
            cat, info["train_rows"], info["pos"], info["best_iter"], info["eval_auc"], path,
        )
        if mlflow_logger:
            mlflow_logger.log_category_result(cat, info, path)
    return info


def train_native_boosters(
    cfg: NativeBoosterConfig,
    mlflow_logger: "MlflowLogger | None" = None,
) -> dict:
    """Train and save the 3 per-category boosters; return the run summary.

    Writes ``ensemble_gbm_{NK,LK,PK}.txt`` + ``summary.json`` under
    ``cfg.out_dir``. ``mlflow_logger`` is optional; when ``None`` no tracking
    happens (the function is fully usable offline).
    """
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    feat = resolve_numeric_features(cfg.train_parquet, cfg.features)
    cfg.features = feat
    sha = feature_schema_sha()
    cat_set = set(CATEGORICAL_FEATURES)
    log.info(
        "numeric_features=%d (dropped %d categoricals) schema_sha=%s",
        len(feat), len(cat_set), sha,
    )

    log.info("streaming TRAIN from %s ...", cfg.train_parquet)
    train = _accumulate(cfg.train_parquet, feat, cfg.batch_rows)
    log.info("streaming EVAL from %s ...", cfg.eval_parquet)
    ev = _accumulate(cfg.eval_parquet, feat, cfg.batch_rows)

    summary: dict = {
        "feature_schema_sha": sha,
        "features": feat,
        "dropped_categoricals": sorted(cat_set),
        "dataset_name": cfg.dataset_name,
        "boosters": {},
    }
    for cat in CATEGORIES:
        if cat not in train:
            log.warning("!! %s: 0 train rows", cat)
            continue
        summary["boosters"][cat] = _train_and_save_category(cat, train, ev, cfg, mlflow_logger)
        del train[cat]  # release the per-category arrays before the next booster
        gc.collect()

    summary_path = cfg.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    if mlflow_logger:
        mlflow_logger.log_summary_artifact(summary_path)
    log.info("DONE -> %s", cfg.out_dir)
    return summary
