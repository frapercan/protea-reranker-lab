"""Reproduce the 0.391 clean15 (15-feat champion subset) recipe on the
association-CORRECTED export 46be427a. Native self_prior + IA, NO v5 overlay
(the overlay-indexed original crashed on the new row counts). MLflow-tracked.

This is the DECISIVE 0.391 reproduction test: the 16-feat recipe trained on the
matched, corrected-association union+expand pool. Eval the resulting boosters on
prediction set 5347c7cb (already matched: union + expand_votes + 7-seed +
corrected association) to compare vs 0.3745 (native S2) and 0.3911 (offline).
"""
from __future__ import annotations

import gc
import io
import os
import time

import lightgbm as lgb
import numpy as np
import pyarrow.parquet as pq
from minio import Minio

BASE = "datasets/fullgo-union-SELECT-160-220-227-c99db18-assoc"
BUCKET = "protea"
IA_TSV = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv"
OUT = "/home/frapercan/Thesis2/storage/fullgo_models/native_boosters_clean15_assocfix"
os.makedirs(OUT, exist_ok=True)
CLIENT = Minio("localhost:9000", access_key="minioadmin", secret_key="minioadmin", secure=False)
CATS = ("nk", "lk", "pk")

LEAN = [
    "distance", "identity_nw", "similarity_nw", "alignment_score_nw", "gaps_pct_nw",
    "alignment_length_nw", "identity_sw", "similarity_sw", "alignment_score_sw",
    "gaps_pct_sw", "alignment_length_sw", "length_query", "length_ref",
    "taxonomic_distance", "taxonomic_common_ancestors", "vote_count", "k_position",
    "go_term_frequency", "ref_annotation_density", "neighbor_distance_std",
    "neighbor_vote_fraction", "neighbor_min_distance", "neighbor_mean_distance",
    "knn_present", "classifier_score", "classifier_present", "self_prior_score",
    "association_total", "association_cross", "association_present",
]
SP_IDX = LEAN.index("self_prior_score")
IA_COL = len(LEAN)  # IA appended after the 30 LEAN cols
CLEAN = [
    "neighbor_vote_fraction", "distance", "identity_nw", "identity_sw", "taxonomic_distance",
    "vote_count", "classifier_score", "knn_present", "classifier_present",
    "association_total", "association_cross", "association_present", "go_term_frequency",
]
CLEAN_IDX = [LEAN.index(c) for c in CLEAN]
NAMES = CLEAN + ["self_prior_score", "IA"]  # 15, exact serve-matching names

PARAMS = dict(objective="binary", metric="auc", learning_rate=0.05, num_leaves=31,
              min_data_in_leaf=50, feature_fraction=0.9, bagging_fraction=0.9,
              bagging_freq=1, verbosity=-1, num_threads=8, seed=42)
NUM_BOOST = 1500
EARLY = 50


def free_g():
    with open("/proc/meminfo") as fh:
        for ln in fh:
            if ln.startswith("MemAvailable"):
                return int(ln.split()[1]) / 1024 / 1024
    return 99.0


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] free={free_g():.1f}G  {m}", flush=True)


def load_ia():
    ia = {}
    for ln in open(IA_TSV):
        p = ln.rstrip("\n").split("\t")
        if len(p) >= 2:
            try:
                ia[p[0]] = float(p[1])
            except ValueError:
                pass
    return ia


def load_split_for_cat(raw, cat, ia):
    """One split, one category -> (X[30 LEAN + IA], y). self_prior + association
    read NATIVELY from the parquet; IA looked up per-row by go_term_id."""
    pf = pq.ParquetFile(io.BytesIO(raw))
    cols = LEAN + ["category", "label", "go_term_id"]
    Xs, ys = [], []
    for b in pf.iter_batches(batch_size=2_000_000, columns=cols):
        cat_arr = np.asarray(b.column("category").to_numpy(zero_copy_only=False), dtype=object)
        m = cat_arr == cat
        if not m.any():
            continue
        idx = np.nonzero(m)[0]
        X = np.empty((idx.size, len(LEAN) + 1), dtype=np.float32)
        for j, f in enumerate(LEAN):
            col = b.column(f).to_numpy(zero_copy_only=False)
            X[:, j] = np.asarray(col, dtype=np.float64)[idx].astype(np.float32)
        goid = b.column("go_term_id").to_pylist()
        X[:, IA_COL] = np.fromiter((ia.get(goid[k], 0.0) for k in idx.tolist()),
                                   dtype=np.float32, count=idx.size)
        Xs.append(X)
        ys.append(b.column("label").to_numpy(zero_copy_only=False)[idx].astype(np.int8))
    del pf
    gc.collect()
    return np.concatenate(Xs), np.concatenate(ys)


def build(X):
    return np.hstack([X[:, CLEAN_IDX], X[:, SP_IDX:SP_IDX + 1], X[:, IA_COL:IA_COL + 1]])


def main():
    import mlflow

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("native-reranker-clean15-assocfix")
    log("loading IA.tsv")
    ia = load_ia()
    log(f"IA terms: {len(ia)}")
    log("downloading train.parquet")
    rawtr = CLIENT.get_object(BUCKET, f"{BASE}/train.parquet").read()
    log(f"train.parquet {len(rawtr) // 1024 // 1024}MB")
    log("downloading eval.parquet")
    rawte = CLIENT.get_object(BUCKET, f"{BASE}/eval.parquet").read()
    with mlflow.start_run(run_name="clean15-assocfix-16feat"):
        mlflow.log_param("dataset", BASE)
        mlflow.log_param("recipe", "clean15 15-feat, corrected association, native self_prior+IA")
        for cat in CATS:
            log(f"{cat}: load train")
            Xtr, ytr = load_split_for_cat(rawtr, cat, ia)
            log(f"{cat}: load eval")
            Xte, yte = load_split_for_cat(rawte, cat, ia)
            Xtr2, Xte2 = build(Xtr), build(Xte)
            del Xtr, Xte
            gc.collect()
            log(f"{cat}: train {Xtr2.shape} pos={int(ytr.sum())} | dev {Xte2.shape}")
            dtr = lgb.Dataset(Xtr2, label=ytr, feature_name=NAMES, free_raw_data=False)
            dev = lgb.Dataset(Xte2, label=yte, reference=dtr, feature_name=NAMES, free_raw_data=False)
            bst = lgb.train(PARAMS, dtr, num_boost_round=NUM_BOOST, valid_sets=[dev],
                            callbacks=[lgb.early_stopping(EARLY), lgb.log_evaluation(200)])
            bst.save_model(f"{OUT}/ensemble_gbm_{cat.upper()}.txt")
            auc = bst.best_score["valid_0"]["auc"]
            mlflow.log_metric(f"{cat}_auc", auc)
            mlflow.log_metric(f"{cat}_best_iter", bst.best_iteration)
            imp = dict(sorted(zip(NAMES, bst.feature_importance("gain").tolist()), key=lambda x: -x[1]))
            log(f"{cat}: SAVED best_iter={bst.best_iteration} auc={auc:.4f} top={list(imp.items())[:4]}")
            del Xtr2, Xte2, dtr, dev, bst
            gc.collect()
    log("DONE")
    return 0


raise SystemExit(main())
