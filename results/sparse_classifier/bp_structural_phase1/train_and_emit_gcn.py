"""Phase 1 GATE: retrain the BP reranker (LK + PK) IN-FRAME, base vs union-pool, with the
GCN JOINT SCORE replacing the frozen two-tower signal, and emit BP test predictions for
board-faithful scoring.

Identical LightGBM setup/seed as the champion train_rerank.py; only pool / feature set
differs (single-variable change vs Phase 0):

  PK  base  = percut knn-only (champion; clf candidates excluded).
      union = percut FULL pool + gcn_score + gcn_rank (the JOINT joint scorer's
              per-cut compatibility for EVERY candidate, replacing Phase 0's clf_rank).

  LK  base  = baseline-M2 full pool (champion).
      union = union_lk_gcn_*.parquet (same Phase 0 pool, tt_clf_score := gcn_score) +
              tt_clf_score / tt_clf_rank / tt_has_clf.

Lever (champion keeper): pminmax on LK-bpo only. Outputs rerank_bp_{cat}_{variant}.parquet
(acc, go, s_rerank; BP rows) + feature-importance json.

Run:  python train_and_emit_gcn.py [pk|lk] [base|union]   (single-variant, process isolation)
"""
import os, sys, json, time, gc
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import lightgbm as lgb

HERE = os.path.dirname(os.path.abspath(__file__))
SC = os.path.dirname(HERE)
PERCUT = os.path.join(SC, "percut_rerank")
BASELINE = os.path.join(SC, "percut_rerank", "baseline")
sys.path.insert(0, HERE)

VALID_PAIR = "v225-v227"
ASPECT_CODE = {"mfo": 0, "bpo": 1, "cco": 2}
SEED = 42
META = {"protein_accession", "go_term_id", "label", "category", "snapshot_pair",
        "qualifier", "evidence_code", "taxonomic_relation", "aspect"}
LK_EXTRA = ["tt_clf_score", "tt_clf_rank", "tt_has_clf"]
PK_EXTRA = ["gcn_score", "gcn_rank"]
KEEP_META = ["protein_accession", "go_term_id", "label", "category", "snapshot_pair", "aspect"]

_SCORER = None


def get_scorer():
    global _SCORER
    if _SCORER is None:
        from gcn_scorer import GcnScorer
        _SCORER = GcnScorer()
    return _SCORER


def base_feature_list(parquet_path):
    sch = pq.ParquetFile(parquet_path).schema_arrow
    cols = list(sch.names)
    extra = set(LK_EXTRA + PK_EXTRA)
    feats = [c for c in cols if c not in META and c not in extra] + ["aspect_code"]
    return feats


def load(parquet_path, category, knn_only, add_gcn):
    sch = pq.ParquetFile(parquet_path).schema_arrow
    allcols = list(sch.names)
    extra = set(LK_EXTRA + PK_EXTRA)
    feat_cols = [c for c in allcols if c not in META and c not in extra]
    present_extra = [c for c in (LK_EXTRA + PK_EXTRA) if c in allcols]
    read_cols = sorted(set(KEEP_META + feat_cols + present_extra))
    df = pq.read_table(parquet_path, columns=read_cols,
                       filters=[("category", "=", category)]).to_pandas()
    if knn_only:
        df = df[df["knn_present"].astype(bool)].copy()
    bool_cols = [n for n in df.columns if str(sch.field(n).type) == "bool"]
    for b in bool_cols:
        df[b] = df[b].astype("int8")
    df["aspect_code"] = df["aspect"].map(ASPECT_CODE).astype("int8")
    for c in df.columns:
        if df[c].dtype == "float64":
            df[c] = df[c].astype("float32")
    if add_gcn:
        t0 = time.time()
        gs = get_scorer().score_frame(
            df[["protein_accession", "go_term_id", "snapshot_pair"]].rename(
                columns={"protein_accession": "acc", "go_term_id": "go"}))
        df["gcn_score"] = gs.astype("float32")
        print(f"    gcn-scored {len(df)} rows in {time.time()-t0:.1f}s "
              f"(finite {np.isfinite(gs).mean()*100:.1f}%)", flush=True)
        # within-(protein, snapshot) rank of the joint score over candidates
        df["gcn_rank"] = np.nan
        m = np.isfinite(df["gcn_score"].to_numpy())
        sub = df.loc[m, ["protein_accession", "snapshot_pair", "gcn_score"]]
        r = sub.groupby(["protein_accession", "snapshot_pair"])["gcn_score"] \
               .rank(ascending=False, method="first")
        df.loc[m, "gcn_rank"] = r.values
        df["gcn_rank"] = df["gcn_rank"].astype("float32")
    return df


def apply_pminmax_lk_bpo(df, col="score"):
    mask = (df["aspect"] == "bpo")
    if not mask.any():
        return df
    sub = df.loc[mask, ["protein_accession", col]].copy()
    g = sub.groupby("protein_accession")[col]
    mn = g.transform("min"); mx = g.transform("max"); rng = mx - mn
    df.loc[mask, col] = np.where(rng > 0, (sub[col] - mn) / rng, 1.0)
    return df


def train_variant(cat, variant):
    if cat == "pk":
        tp = os.path.join(PERCUT, "train.parquet")
        ep = os.path.join(PERCUT, "eval.parquet")
        knn_only = (variant == "base")
        add_gcn = (variant == "union")
        feats = base_feature_list(tp) + (PK_EXTRA if variant == "union" else [])
    else:  # lk
        if variant == "base":
            tp = os.path.join(BASELINE, "train.parquet")
            ep = os.path.join(BASELINE, "eval.parquet")
            feats = base_feature_list(tp)
        else:
            tp = os.path.join(HERE, "union_lk_gcn_train.parquet")
            ep = os.path.join(HERE, "union_lk_gcn_eval.parquet")
            feats = base_feature_list(tp) + LK_EXTRA
        knn_only = False
        add_gcn = False

    t0 = time.time()
    df = load(tp, cat, knn_only, add_gcn)
    tr = df[df["snapshot_pair"] != VALID_PAIR]
    va = df[df["snapshot_pair"] == VALID_PAIR]
    print(f"[{cat} {variant}] train={len(tr)} pos={int(tr.label.sum())} "
          f"valid={len(va)} pos={int(va.label.sum())} feats={len(feats)} "
          f"(load {time.time()-t0:.1f}s)", flush=True)
    Xtr = tr[feats].to_numpy(dtype=np.float32); ytr = tr["label"].astype("int8").values
    Xva = va[feats].to_numpy(dtype=np.float32); yva = va["label"].astype("int8").values
    del df, tr, va; gc.collect()
    dtr = lgb.Dataset(Xtr, label=ytr, free_raw_data=True)
    dva = lgb.Dataset(Xva, label=yva, reference=dtr, free_raw_data=True)
    params = dict(objective="binary", metric=["auc", "average_precision"],
                  learning_rate=0.05, num_leaves=63, min_child_samples=100,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                  max_depth=-1, seed=SEED, verbosity=-1, num_threads=12)
    booster = lgb.train(params, dtr, num_boost_round=3000,
                        valid_sets=[dva], valid_names=["valid"],
                        callbacks=[lgb.early_stopping(80), lgb.log_evaluation(0)])
    best_it = booster.best_iteration
    print(f"[{cat} {variant}] best_iter={best_it} "
          f"auc={booster.best_score['valid']['auc']:.5f} "
          f"ap={booster.best_score['valid']['average_precision']:.5f}", flush=True)
    del Xtr, ytr, Xva, yva, dtr, dva; gc.collect()

    te = load(ep, cat, knn_only, add_gcn)
    te["score"] = booster.predict(te[feats].to_numpy(dtype=np.float32), num_iteration=best_it)
    if cat == "lk":
        te = apply_pminmax_lk_bpo(te)
    bp = te[te["aspect"] == "bpo"][["protein_accession", "go_term_id", "score"]]
    bp = bp.rename(columns={"protein_accession": "acc", "go_term_id": "go", "score": "s_rerank"})
    bp.to_parquet(os.path.join(HERE, f"rerank_bp_{cat}_{variant}.parquet"))
    imp = dict(zip(feats, booster.feature_importance(importance_type="gain")))
    booster.save_model(os.path.join(HERE, f"model_{cat}_{variant}.txt"), num_iteration=best_it)
    top = {k: round(float(v), 1) for k, v in sorted(imp.items(), key=lambda x: -x[1])[:20]}
    new_feats = PK_EXTRA if cat == "pk" else LK_EXTRA
    extra_imp = {k: round(float(imp.get(k, 0)), 1) for k in new_feats}
    return {"best_iter": best_it, "n_bp_pred": len(bp), "top_importance": top,
            "new_feat_importance": extra_imp,
            "auc": round(float(booster.best_score['valid']['auc']), 5),
            "ap": round(float(booster.best_score['valid']['average_precision']), 5)}


def main():
    if len(sys.argv) == 3:
        cat, variant = sys.argv[1], sys.argv[2]
        m = train_variant(cat, variant)
        json.dump({f"{cat}_{variant}": m},
                  open(os.path.join(HERE, f"train_meta_{cat}_{variant}.json"), "w"), indent=2)
        print(f"done {cat} {variant}", flush=True)
        return
    meta = {}
    for cat, variant in [("pk", "base"), ("pk", "union"), ("lk", "base"), ("lk", "union")]:
        meta[f"{cat}_{variant}"] = train_variant(cat, variant)
    json.dump(meta, open(os.path.join(HERE, "train_meta_all.json"), "w"), indent=2)
    print("done", flush=True)


if __name__ == "__main__":
    main()
