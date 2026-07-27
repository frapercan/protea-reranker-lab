"""Retrain the champion BP reranker (LK + PK) IN-FRAME, base vs union-pool, and
emit BP test predictions for board-faithful scoring.

A/B per category (identical LightGBM setup/seed as the champion train_rerank.py;
only the pool / feature set differs):

  PK  base  = percut dataset, knn_present=True  (champion: clf candidates excluded)
      union = percut dataset, FULL pool (admits the two-tower clf candidates that
              raised PK-bpo ceiling 0.470->0.537) + clf_rank feature. classifier_score
              / classifier_present are already champion features.

  LK  base  = baseline-M2 dataset, full pool (champion)
      union = union_lk_*.parquet (champion pool UNION two-tower BP candidates) +
              tt_clf_score / tt_clf_rank / tt_has_clf features.

Lever (champion keeper): pminmax on LK-bpo only. Outputs rerank_bp_{cat}_{variant}.parquet
(acc, go, s_rerank; BP rows) + feature-importance json.
"""
import os, sys, json, time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import lightgbm as lgb

HERE = os.path.dirname(os.path.abspath(__file__))
SC = "/home/frapercan/Thesis2/repositories/protea-reranker-lab/results/sparse_classifier"
PERCUT = os.path.join(SC, "percut_rerank")
BASELINE = os.path.join(SC, "percut_rerank", "baseline")

VALID_PAIR = "v225-v227"
ASPECT_CODE = {"mfo": 0, "bpo": 1, "cco": 2}
SEED = 42
META = {"protein_accession", "go_term_id", "label", "category", "snapshot_pair",
        "qualifier", "evidence_code", "taxonomic_relation", "aspect"}
LK_EXTRA = ["tt_clf_score", "tt_clf_rank", "tt_has_clf"]
PK_EXTRA = ["clf_rank"]


# meta we actually need downstream (drop unused string meta to save memory)
KEEP_META = ["protein_accession", "go_term_id", "label", "category",
             "snapshot_pair", "aspect"]


def base_feature_list(parquet_path):
    sch = pq.ParquetFile(parquet_path).schema_arrow
    cols = list(sch.names)
    bool_cols = [n for n in cols if str(sch.field(n).type) == "bool"]
    extra = set(LK_EXTRA + PK_EXTRA)
    feats = [c for c in cols if c not in META and c not in extra] + ["aspect_code"]
    return feats, bool_cols, None


def load(parquet_path, category, knn_only, add_clf_rank):
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
    if add_clf_rank:
        # within-(protein, snapshot) rank of the two-tower classifier_score over
        # classifier candidates (BP and otherwise); NaN for non-clf rows.
        df["clf_rank"] = np.nan
        m = df["classifier_present"].astype(bool).values
        sub = df.loc[m, ["protein_accession", "snapshot_pair", "classifier_score"]]
        r = sub.groupby(["protein_accession", "snapshot_pair"])["classifier_score"] \
               .rank(ascending=False, method="first")
        df.loc[m, "clf_rank"] = r.values
        df["clf_rank"] = df["clf_rank"].astype("float32")
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
        add_rank = (variant == "union")
        feats0, _, _ = base_feature_list(tp)
        feats = feats0 + (PK_EXTRA if variant == "union" else [])
    else:  # lk
        if variant == "base":
            tp = os.path.join(BASELINE, "train.parquet")
            ep = os.path.join(BASELINE, "eval.parquet")
            feats0, _, _ = base_feature_list(tp)
            feats = feats0
        else:
            tp = os.path.join(HERE, "union_lk_train.parquet")
            ep = os.path.join(HERE, "union_lk_eval.parquet")
            feats0, _, _ = base_feature_list(tp)
            feats = feats0 + LK_EXTRA
        knn_only = False
        add_rank = False

    t0 = time.time()
    df = load(tp, cat, knn_only, add_rank)
    tr = df[df["snapshot_pair"] != VALID_PAIR]
    va = df[df["snapshot_pair"] == VALID_PAIR]
    print(f"[{cat} {variant}] train={len(tr)} pos={int(tr.label.sum())} "
          f"valid={len(va)} pos={int(va.label.sum())} feats={len(feats)} "
          f"(load {time.time()-t0:.1f}s)", flush=True)
    Xtr = tr[feats].to_numpy(dtype=np.float32); ytr = tr["label"].astype("int8").values
    Xva = va[feats].to_numpy(dtype=np.float32); yva = va["label"].astype("int8").values
    del df, tr, va; import gc; gc.collect()
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

    te = load(ep, cat, knn_only, add_rank)
    te["score"] = booster.predict(te[feats].to_numpy(dtype=np.float32),
                                  num_iteration=best_it)
    if cat == "lk":
        te = apply_pminmax_lk_bpo(te)
    bp = te[te["aspect"] == "bpo"][["protein_accession", "go_term_id", "score"]]
    bp = bp.rename(columns={"protein_accession": "acc", "go_term_id": "go",
                            "score": "s_rerank"})
    bp.to_parquet(os.path.join(HERE, f"rerank_bp_{cat}_{variant}.parquet"))
    imp = dict(zip(feats, booster.feature_importance(importance_type="gain")))
    booster.save_model(os.path.join(HERE, f"model_{cat}_{variant}.txt"), num_iteration=best_it)
    top = {k: round(float(v), 1) for k, v in sorted(imp.items(), key=lambda x: -x[1])[:20]}
    extra_imp = {k: round(float(imp.get(k, 0)), 1) for k in (PK_EXTRA if cat == "pk" else LK_EXTRA)}
    return {"best_iter": best_it, "n_bp_pred": len(bp),
            "top_importance": top, "new_feat_importance": extra_imp}


def main():
    # single-variant mode for process isolation: `train_and_emit.py pk union`
    if len(sys.argv) == 3:
        cat, variant = sys.argv[1], sys.argv[2]
        m = train_variant(cat, variant)
        json.dump({f"{cat}_{variant}": m},
                  open(os.path.join(HERE, f"train_meta_{cat}_{variant}.json"), "w"), indent=2)
        print(f"done {cat} {variant}", flush=True)
        return
    plan = [("pk", "base"), ("pk", "union"), ("lk", "base"), ("lk", "union")]
    meta = {}
    for cat, variant in plan:
        meta[f"{cat}_{variant}"] = train_variant(cat, variant)
    json.dump(meta, open(os.path.join(HERE, "train_meta_all.json"), "w"), indent=2)
    print("done", flush=True)


if __name__ == "__main__":
    main()
