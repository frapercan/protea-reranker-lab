"""Per-category LightGBM lambdarank reranker on the NEW clf+assoc platform
dataset (clean-learned-clfassoc-train227-test230). Identical protocol to the
prior clean_227230 per-category setup, EXCEPT the feature set now INCLUDES the
populated classifier_score, classifier_present, association_total,
association_cross, association_present (zero-filled and excluded before).

Reads train per-category via parquet filters to keep RAM peak bounded
(PK = 37.5M rows). Early-stop val = latest train pair v225-v227. Predicts on
eval.parquet (v227-v230) and writes eval_scores.parquet.
"""
import gc
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

S = Path("/tmp/claude-1000/-home-frapercan-Thesis2/afd2c43a-ede7-46dc-94dd-9745808d2194/scratchpad")
TRAIN = S / "ds_clfassoc" / "train.parquet"
EVAL = S / "ds_clfassoc" / "eval.parquet"
OUT = S / "clfassoc_out"
OUT.mkdir(exist_ok=True)
VAL_PAIR = "v225-v227"
CATS = ["nk", "lk", "pk"]

# Metadata / non-feature columns. The 5 clf+assoc features are NOT excluded now.
EXCLUDE = {
    "protein_accession", "go_term_id", "label", "category", "snapshot_pair",
    "qualifier", "evidence_code", "taxonomic_relation", "aspect",
}

PARAMS = {
    "objective": "lambdarank",
    "metric": ["ndcg", "map"],
    "ndcg_eval_at": [5, 10],
    "label_gain": [0, 1],
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 5,
    "seed": 42,
    "verbose": -1,
    "num_threads": 12,
}
NUM_BOOST_ROUND = 3000
EARLY_STOP = 50


def feature_cols():
    names = pq.ParquetFile(str(TRAIN)).schema_arrow.names
    return [c for c in names if c not in EXCLUDE]


def read_cat(path, feats, cat, want_ids=False):
    cols = feats + ["snapshot_pair", "protein_accession", "aspect", "label"]
    if want_ids:
        cols += ["go_term_id", "distance"]
    cols = list(dict.fromkeys(cols))
    t = pq.read_table(str(path), columns=cols,
                      filters=[("category", "==", cat)])
    n = t.num_rows
    X = np.empty((n, len(feats)), dtype=np.float32)
    for j, c in enumerate(feats):
        X[:, j] = t.column(c).to_numpy(zero_copy_only=False).astype(np.float32)
    meta = {
        "snapshot_pair": np.asarray(t.column("snapshot_pair").to_pylist()),
        "protein_accession": np.asarray(t.column("protein_accession").to_pylist()),
        "aspect": np.asarray(t.column("aspect").to_pylist()),
        "label": t.column("label").to_numpy(zero_copy_only=False).astype(np.int64),
    }
    if want_ids:
        meta["go_term_id"] = np.asarray(t.column("go_term_id").to_pylist())
        meta["distance"] = t.column("distance").to_numpy(zero_copy_only=False).astype(np.float64)
    del t
    return X, meta


def group_sizes_sorted(snap, prot, asp):
    snap_c = pd.factorize(snap)[0].astype(np.int64)
    prot_c = pd.factorize(prot)[0].astype(np.int64)
    asp_c = pd.factorize(asp)[0].astype(np.int64)
    n_asp = asp_c.max() + 1
    n_prot = prot_c.max() + 1
    combo = (snap_c * n_prot + prot_c) * n_asp + asp_c
    order = np.argsort(combo, kind="stable")
    combo_sorted = combo[order]
    _, sizes = np.unique(combo_sorted, return_counts=True)
    return order, sizes.astype(np.int32)


def train_one(cat, feats):
    print(f"[{cat}] reading...", flush=True)
    X, meta = read_cat(TRAIN, feats, cat)
    y = (meta["label"] > 0).astype(np.int32)
    snap = meta["snapshot_pair"]
    prot = meta["protein_accession"]
    asp = meta["aspect"]
    is_val = snap == VAL_PAIR
    tr = ~is_val
    print(f"[{cat}] rows={len(y)} train={tr.sum()} val={is_val.sum()} "
          f"pos_frac={y.mean():.4f}", flush=True)

    o_tr, g_tr = group_sizes_sorted(snap[tr], prot[tr], asp[tr])
    o_va, g_va = group_sizes_sorted(snap[is_val], prot[is_val], asp[is_val])
    Xtr = X[tr][o_tr]
    ytr = y[tr][o_tr]
    Xva = X[is_val][o_va]
    yva = y[is_val][o_va]
    del X, meta, snap, prot, asp, is_val, tr, y
    gc.collect()

    dtr = lgb.Dataset(Xtr, label=ytr, group=g_tr, feature_name=feats, free_raw_data=True)
    dva = lgb.Dataset(Xva, label=yva, group=g_va, reference=dtr,
                      feature_name=feats, free_raw_data=True)
    del Xtr, Xva
    gc.collect()
    t = time.time()
    booster = lgb.train(
        PARAMS, dtr, num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[dtr, dva], valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False),
                   lgb.log_evaluation(period=200)],
    )
    best = booster.best_iteration or booster.current_iteration()
    print(f"[{cat}] best_iter={best} time={time.time()-t:.0f}s", flush=True)
    fi = dict(sorted(zip(booster.feature_name(),
                         booster.feature_importance(importance_type="gain").tolist()),
                     key=lambda kv: -kv[1]))
    booster.save_model(str(OUT / f"booster_{cat}.txt"), num_iteration=best)
    del dtr, dva
    gc.collect()
    return best, fi


def main():
    feats = feature_cols()
    print(f"n_features={len(feats)}", flush=True)
    new5 = ["classifier_score", "classifier_present", "association_total",
            "association_cross", "association_present"]
    print("new5 in feats:", [f in feats for f in new5], flush=True)
    (OUT / "features.json").write_text(json.dumps(feats, indent=1))

    info = {}
    for cat in CATS:
        best, fi = train_one(cat, feats)
        info[cat] = {"best_iteration": best,
                     "top_features": dict(list(fi.items())[:20]),
                     "new5_importance": {k: fi.get(k) for k in new5}}
    (OUT / "train_info.json").write_text(json.dumps(info, indent=1))

    print("predicting on eval...", flush=True)
    boosters = {c: lgb.Booster(model_file=str(OUT / f"booster_{c}.txt")) for c in CATS}
    cols = feats + ["protein_accession", "go_term_id", "label", "category",
                    "aspect", "distance"]
    cols = list(dict.fromkeys(cols))
    t = pq.read_table(str(EVAL), columns=cols)
    n = t.num_rows
    Xev = np.empty((n, len(feats)), dtype=np.float32)
    for j, c in enumerate(feats):
        Xev[:, j] = t.column(c).to_numpy(zero_copy_only=False).astype(np.float32)
    cat = np.asarray(t.column("category").to_pylist())
    rerank = np.empty(n, dtype=np.float64)
    for c in CATS:
        m = cat == c
        if m.sum() == 0:
            continue
        rerank[m] = boosters[c].predict(Xev[m], num_iteration=info[c]["best_iteration"])
    tbl = pa.table({
        "protein_accession": t.column("protein_accession"),
        "go_term_id": t.column("go_term_id"),
        "label": t.column("label"),
        "category": t.column("category"),
        "aspect": t.column("aspect"),
        "distance": t.column("distance"),
        "reranker_score": rerank,
    })
    pq.write_table(tbl, str(OUT / "eval_scores.parquet"))
    print("DONE wrote", OUT / "eval_scores.parquet", flush=True)


if __name__ == "__main__":
    main()
