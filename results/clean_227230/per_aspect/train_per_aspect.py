"""Train per-(category x aspect) LightGBM lambdarank rerankers on the frozen
clean-learned-train227-test230 dataset. Up to 9 models (NK/LK/PK x mfo/bpo/cco).
Each model is grouped by (snapshot_pair, protein) WITHIN its aspect (all rows of
that model share one aspect, so the aspect axis collapses out of the group key).
Early-stop on the latest train cut v225-v227. Predicts the routed raw booster
score on eval.parquet (each eval row scored by the booster matching its own
(category, aspect)). Writes eval_scores_per_aspect.parquet with the RAW routed
score so the downstream step can apply different score calibrations.

Mirrors the per-category reranker params exactly (lambdarank, num_leaves 63,
lr 0.05, min_data_in_leaf 100, ff 0.9, bagging 0.9/freq5, 3000 rounds, ES 50).
Loads the full train matrix ONCE (~4GB), trains sequentially, frees the lgb
Dataset between models to keep peak RAM modest.
"""
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

S = Path("/tmp/claude-1000/-home-frapercan-Thesis2/afd2c43a-ede7-46dc-94dd-9745808d2194/scratchpad")
TRAIN = S / "ds227230" / "train.parquet"
EVAL = S / "ds227230" / "eval.parquet"
OUT = S / "per_aspect"
BOOST = OUT / "boosters"
BOOST.mkdir(parents=True, exist_ok=True)
VAL_PAIR = "v225-v227"
CATS = ["nk", "lk", "pk"]
ASPECTS = ["mfo", "bpo", "cco"]

EXCLUDE = {
    "protein_accession", "go_term_id", "label", "category", "snapshot_pair",
    "qualifier", "evidence_code", "taxonomic_relation", "aspect",
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
        X[:, j] = t.column(c).to_numpy(zero_copy_only=False).astype(np.float32)
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


def group_sizes_sorted(snap, prot):
    """rows[order] contiguous per (snap,prot); sizes aligns to sorted blocks."""
    snap_c = pd.factorize(snap)[0].astype(np.int64)
    prot_c = pd.factorize(prot)[0].astype(np.int64)
    n_prot = prot_c.max() + 1
    combo = snap_c * n_prot + prot_c
    order = np.argsort(combo, kind="stable")
    _, sizes = np.unique(combo[order], return_counts=True)
    return order, sizes.astype(np.int32)


def train_one(cat, asp, X, meta, feats):
    m = (meta["category"] == cat) & (meta["aspect"] == asp)
    Xc = X[m]
    y = (meta["label"][m] > 0).astype(np.int32)
    snap = meta["snapshot_pair"][m]
    prot = meta["protein_accession"][m]
    is_val = snap == VAL_PAIR
    tr = ~is_val
    print(f"[{cat}-{asp}] rows={m.sum()} train={tr.sum()} val={is_val.sum()} "
          f"pos_frac={y.mean():.4f}", flush=True)

    o_tr, g_tr = group_sizes_sorted(snap[tr], prot[tr])
    o_va, g_va = group_sizes_sorted(snap[is_val], prot[is_val])
    dtr = lgb.Dataset(Xc[tr][o_tr], label=y[tr][o_tr], group=g_tr,
                      feature_name=feats, free_raw_data=True)
    dva = lgb.Dataset(Xc[is_val][o_va], label=y[is_val][o_va], group=g_va,
                      reference=dtr, feature_name=feats, free_raw_data=True)
    t = time.time()
    booster = lgb.train(
        PARAMS, dtr, num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[dva], valid_names=["val"],
        callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False),
                   lgb.log_evaluation(period=200)],
    )
    best = booster.best_iteration or booster.current_iteration()
    print(f"[{cat}-{asp}] best_iter={best} time={time.time()-t:.0f}s", flush=True)
    fi = dict(sorted(zip(booster.feature_name(),
                         booster.feature_importance(importance_type="gain").tolist()),
                     key=lambda kv: -kv[1]))
    booster.save_model(str(BOOST / f"booster_{cat}_{asp}.txt"), num_iteration=best)
    del dtr, dva, Xc
    return booster, best, fi


def main():
    feats = feature_cols()
    print(f"n_features={len(feats)}", flush=True)
    (OUT / "features.json").write_text(json.dumps(feats, indent=1))

    print("loading train matrix...", flush=True)
    Xtr, meta_tr = load_matrix(TRAIN, feats)
    info = {}
    bests = {}
    for cat in CATS:
        for asp in ASPECTS:
            b, best, fi = train_one(cat, asp, Xtr, meta_tr, feats)
            bests[(cat, asp)] = best
            info[f"{cat}-{asp}"] = {"best_iteration": best,
                                    "top_features": dict(list(fi.items())[:12])}
            del b
    del Xtr, meta_tr

    print("loading eval matrix + routed predict...", flush=True)
    Xev, meta_ev = load_matrix(EVAL, feats, want_ids=True)
    n = Xev.shape[0]
    routed = np.full(n, np.nan, dtype=np.float64)
    for cat in CATS:
        for asp in ASPECTS:
            m = (meta_ev["category"] == cat) & (meta_ev["aspect"] == asp)
            if m.sum() == 0:
                continue
            bm = lgb.Booster(model_file=str(BOOST / f"booster_{cat}_{asp}.txt"))
            routed[m] = bm.predict(Xev[m])
            del bm
    assert not np.isnan(routed).any(), "some eval rows unrouted"
    tbl = pa.table({
        "protein_accession": meta_ev["protein_accession"],
        "go_term_id": meta_ev["go_term_id"],
        "label": meta_ev["label"],
        "category": meta_ev["category"],
        "aspect": meta_ev["aspect"],
        "distance": meta_ev["distance"],
        "raw_score": routed,
    })
    pq.write_table(tbl, str(OUT / "eval_scores_per_aspect.parquet"))
    (OUT / "train_info.json").write_text(json.dumps(info, indent=1))
    print("DONE wrote eval_scores_per_aspect.parquet", flush=True)


if __name__ == "__main__":
    main()
