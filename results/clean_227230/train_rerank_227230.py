"""Per-category LightGBM reranker on the frozen clean-learned-train227-test230
dataset. Trains one lambdarank booster per category (NK/LK/PK), grouped by
(snapshot_pair, protein, aspect), held-out val = latest train pair v225-v227 for
early stopping. Predicts reranker scores on eval.parquet (v227-v230) for every
candidate row. Writes eval_scores.parquet (ids + distance + label + category +
aspect + reranker_score) for the downstream cafaeval scoring step.

Run with the lab venv. Config mirrors the lab TrainConfig defaults
(reranker.py / universal_train._make_train_config): lambdarank, num_leaves 63,
lr 0.05, min_data_in_leaf 100, feature_fraction 0.9, bagging 0.9/freq5,
num_boost_round 3000, early_stopping 50, metric ndcg/map, label_gain [0,1].
"""
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

S = Path("/tmp/claude-1000/-home-frapercan-Thesis2/afd2c43a-ede7-46dc-94dd-9745808d2194/scratchpad")
TRAIN = S / "ds227230" / "train.parquet"
EVAL = S / "ds227230" / "eval.parquet"
OUT = S / "rerank_out"
OUT.mkdir(exist_ok=True)
VAL_PAIR = "v225-v227"
CATS = ["nk", "lk", "pk"]

EXCLUDE = {
    "protein_accession", "go_term_id", "label", "category", "snapshot_pair",
    "qualifier", "evidence_code", "taxonomic_relation", "aspect",
    # zero-filled in this export (compute_association/classifier = False)
    "classifier_score", "classifier_present",
    "association_total", "association_cross", "association_present",
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


def load_matrix(path, feats, want_ids=False):
    cols = feats + ["category", "snapshot_pair", "protein_accession", "aspect", "label"]
    if want_ids:
        cols += ["go_term_id", "distance"]
    cols = list(dict.fromkeys(cols))
    t = pq.read_table(str(path), columns=cols)
    n = t.num_rows
    X = np.empty((n, len(feats)), dtype=np.float32)
    for j, c in enumerate(feats):
        arr = t.column(c).to_numpy(zero_copy_only=False)
        X[:, j] = arr.astype(np.float32)
    meta = {
        "category": np.asarray(t.column("category").to_pylist()),
        "snapshot_pair": np.asarray(t.column("snapshot_pair").to_pylist()),
        "protein_accession": np.asarray(t.column("protein_accession").to_pylist()),
        "aspect": np.asarray(t.column("aspect").to_pylist()),
        "label": t.column("label").to_numpy(zero_copy_only=False).astype(np.int64),
    }
    if want_ids:
        meta["go_term_id"] = np.asarray(t.column("go_term_id").to_pylist())
        meta["distance"] = t.column("distance").to_numpy(zero_copy_only=False).astype(np.float64)
    return X, meta


def group_sizes_sorted(snap, prot, asp):
    """Return (order, group_sizes) so rows[order] are contiguous per
    (snap,prot,asp) group; group_sizes aligns to the sorted blocks."""
    snap_c = pd.factorize(snap)[0].astype(np.int64)
    prot_c = pd.factorize(prot)[0].astype(np.int64)
    asp_c = pd.factorize(asp)[0].astype(np.int64)
    n_asp = asp_c.max() + 1
    n_prot = prot_c.max() + 1
    combo = (snap_c * n_prot + prot_c) * n_asp + asp_c
    order = np.argsort(combo, kind="stable")
    combo_sorted = combo[order]
    _, sizes = np.unique(combo_sorted, return_counts=True)
    # unique returns sorted-by-value; combo_sorted is sorted so counts align
    return order, sizes.astype(np.int32)


def train_one(cat, X, meta, feats):
    m = meta["category"] == cat
    Xc = X[m]
    y = (meta["label"][m] > 0).astype(np.int32)
    snap = meta["snapshot_pair"][m]
    prot = meta["protein_accession"][m]
    asp = meta["aspect"][m]
    is_val = snap == VAL_PAIR
    tr = ~is_val
    print(f"[{cat}] rows={m.sum()} train={tr.sum()} val={is_val.sum()} "
          f"pos_frac={y.mean():.4f}", flush=True)

    o_tr, g_tr = group_sizes_sorted(snap[tr], prot[tr], asp[tr])
    o_va, g_va = group_sizes_sorted(snap[is_val], prot[is_val], asp[is_val])
    Xtr = Xc[tr][o_tr]
    ytr = y[tr][o_tr]
    Xva = Xc[is_val][o_va]
    yva = y[is_val][o_va]

    dtr = lgb.Dataset(Xtr, label=ytr, group=g_tr, feature_name=feats, free_raw_data=True)
    dva = lgb.Dataset(Xva, label=yva, group=g_va, reference=dtr,
                      feature_name=feats, free_raw_data=True)
    t = time.time()
    booster = lgb.train(
        PARAMS, dtr, num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[dtr, dva], valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False),
                   lgb.log_evaluation(period=100)],
    )
    best = booster.best_iteration or booster.current_iteration()
    print(f"[{cat}] best_iter={best} time={time.time()-t:.0f}s", flush=True)
    fi = dict(sorted(zip(booster.feature_name(),
                         booster.feature_importance(importance_type="gain").tolist()),
                     key=lambda kv: -kv[1]))
    booster.save_model(str(OUT / f"booster_{cat}.txt"), num_iteration=best)
    return booster, best, fi


def main():
    feats = feature_cols()
    print(f"n_features={len(feats)}", flush=True)
    (OUT / "features.json").write_text(json.dumps(feats, indent=1))

    print("loading train matrix...", flush=True)
    Xtr_all, meta_tr = load_matrix(TRAIN, feats)
    boosters, info = {}, {}
    for cat in CATS:
        b, best, fi = train_one(cat, Xtr_all, meta_tr, feats)
        boosters[cat] = b
        info[cat] = {"best_iteration": best,
                     "top_features": dict(list(fi.items())[:15])}
    del Xtr_all, meta_tr

    print("loading eval matrix + predicting...", flush=True)
    Xev, meta_ev = load_matrix(EVAL, feats, want_ids=True)
    n = Xev.shape[0]
    rerank = np.empty(n, dtype=np.float64)
    for cat in CATS:
        m = meta_ev["category"] == cat
        if m.sum() == 0:
            continue
        best = info[cat]["best_iteration"]
        rerank[m] = boosters[cat].predict(Xev[m], num_iteration=best)
    # save eval scores table
    import pyarrow as pa
    tbl = pa.table({
        "protein_accession": meta_ev["protein_accession"],
        "go_term_id": meta_ev["go_term_id"],
        "label": meta_ev["label"],
        "category": meta_ev["category"],
        "aspect": meta_ev["aspect"],
        "distance": meta_ev["distance"],
        "reranker_score": rerank,
    })
    pq.write_table(tbl, str(OUT / "eval_scores.parquet"))
    (OUT / "train_info.json").write_text(json.dumps(info, indent=1))
    print("DONE. wrote", OUT / "eval_scores.parquet", flush=True)


if __name__ == "__main__":
    main()
