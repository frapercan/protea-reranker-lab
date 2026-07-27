"""In-frame literature test: retrain the champion BP reranker on its OWN
multi-snapshot training frame (v160..v225 train, v225-v227 valid for early stop,
v227-v230 test), WITH vs WITHOUT literature features. Per category:
  LK -> baseline dataset (M2 candidates), PK -> percut dataset (per-cut candidates).

Literature features (BP rows only, snapshot-independent text_score join):
  text_score, text_cos, text_rank_pct, text_top1/3/5, has_text
(rank_gap is intentionally excluded: it needs the reranker score, circular inside
the reranker). Same LightGBM setup/seed as train_rerank.py; only the feature set
differs between WITHOUT and WITH literature -> clean A/B.

Outputs the BP test predictions for each variant; graft+score handled separately.
"""
import os, sys, json, time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import lightgbm as lgb

HERE = os.path.dirname(os.path.abspath(__file__))
SC = "/home/frapercan/Thesis2/repositories/protea-reranker-lab/results/sparse_classifier"
TEXT_SCORES = os.path.join(HERE, "text_scores_infame.parquet")

DATASET_DIR = {  # champion source per category
    "lk": os.path.join(SC, "percut_rerank", "baseline"),
    "pk": os.path.join(SC, "percut_rerank"),
}
VALID_PAIR = "v225-v227"
ASPECT_CODE = {"mfo": 0, "bpo": 1, "cco": 2}
SEED = 42
LIT_FEATS = ["text_score", "text_cos", "text_rank_pct",
             "text_top1", "text_top3", "text_top5", "has_text"]


def _meta_cols():
    return {"protein_accession", "go_term_id", "label", "category", "snapshot_pair",
            "qualifier", "evidence_code", "taxonomic_relation", "aspect"}


def base_feature_list(train_parquet):
    sch = pq.ParquetFile(train_parquet).schema_arrow
    cols = list(sch.names)
    meta = _meta_cols()
    bool_cols = [n for n in cols if str(sch.field(n).type) == "bool"]
    feats = [c for c in cols if c not in meta] + ["aspect_code"]
    read_cols = list(meta) + [c for c in feats if c != "aspect_code"]
    return feats, bool_cols, read_cols


def add_lit_feats(df, text_df):
    """df has protein_accession, go_term_id, aspect, snapshot_pair. Adds LIT_FEATS.
    text_df: DataFrame acc,go,cos,text_score. BP rows only get nonzero (vectorized
    merge). Non-BP rows / pairs absent from text_df default to 0."""
    is_bp = (df["aspect"] == "bpo").values
    key = df[["protein_accession", "go_term_id"]].rename(
        columns={"protein_accession": "acc", "go_term_id": "go"})
    merged = key.merge(text_df[["acc", "go", "cos", "text_score"]],
                       on=["acc", "go"], how="left")
    cos = merged["cos"].to_numpy(dtype=np.float32, na_value=0.0)
    ts = merged["text_score"].to_numpy(dtype=np.float32, na_value=0.0)
    # zero out any non-BP rows that happened to match a pair
    cos = np.where(is_bp, cos, 0.0).astype(np.float32)
    ts = np.where(is_bp, ts, 0.0).astype(np.float32)
    df["text_cos"] = cos
    df["text_score"] = ts
    # within-protein-per-snapshot rank features over BP rows
    df["text_rank_pct"] = 0.0
    df["text_top1"] = 0.0
    df["text_top3"] = 0.0
    df["text_top5"] = 0.0
    df["has_text"] = 0.0
    bp = df[is_bp]
    if len(bp):
        grp = bp.groupby(["snapshot_pair", "protein_accession"])
        rank_pct = grp["text_score"].rank(pct=True, method="average")
        rank_desc = grp["text_score"].rank(ascending=False, method="first")
        has = grp["text_score"].transform(lambda s: float((s > 0).any()))
        df.loc[is_bp, "text_rank_pct"] = rank_pct.values
        df.loc[is_bp, "has_text"] = has.values
        for k in (1, 3, 5):
            df.loc[is_bp, f"text_top{k}"] = (
                (rank_desc.values <= k) & (has.values > 0)).astype(np.float32)
        # no-text proteins: zero the percentile so it isn't a spurious signal
        notext = is_bp & (df["has_text"].values <= 0)
        df.loc[notext, "text_rank_pct"] = 0.0
    for c in LIT_FEATS:
        df[c] = df[c].astype("float32")
    return df


def load_category(parquet_path, category, bool_cols, read_cols, base_feats,
                  text_df=None):
    filt = [("category", "=", category)]
    if category == "pk":
        filt.append(("knn_present", "=", True))
    df = pq.read_table(parquet_path, columns=read_cols, filters=filt).to_pandas()
    for b in bool_cols:
        if b in df.columns:
            df[b] = df[b].astype("int8")
    df["aspect_code"] = df["aspect"].map(ASPECT_CODE).astype("int8")
    for c in base_feats:
        if c in df.columns and df[c].dtype == "float64":
            df[c] = df[c].astype("float32")
    if text_df is not None:
        df = add_lit_feats(df, text_df)
    return df


def apply_pminmax_lk_bpo(df, score_col="score"):
    mask = (df["aspect"] == "bpo")
    if not mask.any():
        return df
    sub = df.loc[mask, ["protein_accession", score_col]].copy()
    g = sub.groupby("protein_accession")[score_col]
    mn = g.transform("min"); mx = g.transform("max"); rng = mx - mn
    norm = np.where(rng > 0, (sub[score_col] - mn) / rng, 1.0)
    df.loc[mask, score_col] = norm
    return df


def train_variant(cat, use_lit, text_df):
    ddir = DATASET_DIR[cat]
    train_parquet = os.path.join(ddir, "train.parquet")
    eval_parquet = os.path.join(ddir, "eval.parquet")
    base_feats, bool_cols, read_cols = base_feature_list(train_parquet)
    feats = base_feats + (LIT_FEATS if use_lit else [])

    t0 = time.time()
    df = load_category(train_parquet, cat, bool_cols, read_cols, base_feats,
                       text_df if use_lit else None)
    tr = df[df["snapshot_pair"] != VALID_PAIR]
    va = df[df["snapshot_pair"] == VALID_PAIR]
    print(f"[{cat} lit={use_lit}] train={len(tr)} pos={int(tr.label.sum())} "
          f"valid={len(va)} (load {time.time()-t0:.1f}s)", flush=True)
    dtr = lgb.Dataset(tr[feats], label=tr["label"].astype("int8").values,
                      free_raw_data=False)
    dva = lgb.Dataset(va[feats], label=va["label"].astype("int8").values,
                      reference=dtr, free_raw_data=False)
    params = dict(objective="binary", metric=["auc", "average_precision"],
                  learning_rate=0.05, num_leaves=63, min_child_samples=100,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                  max_depth=-1, seed=SEED, verbosity=-1, num_threads=12)
    booster = lgb.train(params, dtr, num_boost_round=3000,
                        valid_sets=[dva], valid_names=["valid"],
                        callbacks=[lgb.early_stopping(80),
                                   lgb.log_evaluation(0)])
    best_it = booster.best_iteration
    print(f"[{cat} lit={use_lit}] best_iter={best_it} "
          f"valid_auc={booster.best_score['valid']['auc']:.5f} "
          f"valid_ap={booster.best_score['valid']['average_precision']:.5f}",
          flush=True)
    del df, tr, va, dtr, dva

    te = load_category(eval_parquet, cat, bool_cols, read_cols, base_feats,
                       text_df if use_lit else None)
    te["score"] = booster.predict(te[feats], num_iteration=best_it)
    if cat == "lk":
        te = apply_pminmax_lk_bpo(te)
    bp = te[te["aspect"] == "bpo"][["protein_accession", "go_term_id", "score"]]
    bp = bp.rename(columns={"protein_accession": "acc", "go_term_id": "go",
                            "score": "s_rerank"})
    imp = dict(zip(feats, booster.feature_importance(importance_type="gain")))
    suffix = "lit" if use_lit else "base"
    booster.save_model(os.path.join(HERE, f"model_{cat}_{suffix}.txt"),
                       num_iteration=best_it)
    return bp, imp, best_it


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "all"   # lk | pk | all
    cats = ["lk", "pk"] if tag == "all" else [tag]
    text_df = pd.read_parquet(TEXT_SCORES)
    print(f"text pairs: {len(text_df)}", flush=True)
    out = {}
    for cat in cats:
        for use_lit in (False, True):
            bp, imp, best_it = train_variant(cat, use_lit, text_df)
            sfx = "lit" if use_lit else "base"
            bp.to_parquet(os.path.join(HERE, f"rerank_bp_{cat}_{sfx}.parquet"))
            topimp = {k: round(float(v), 1) for k, v in
                      sorted(imp.items(), key=lambda x: -x[1])[:20]}
            out[f"{cat}_{sfx}"] = {"best_iter": best_it, "top_importance": topimp,
                                   "lit_importance": {k: round(float(imp.get(k, 0)), 1)
                                                      for k in LIT_FEATS}}
    json.dump(out, open(os.path.join(HERE, f"train_meta_{tag}.json"), "w"), indent=2)
    print("done", flush=True)


if __name__ == "__main__":
    main()
