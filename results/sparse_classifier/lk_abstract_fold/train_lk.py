"""LK-BP in-frame A/B/C: retrain the champion LK reranker on its OWN multi-snapshot
frame (v160..v225 train, v225-v227 early-stop), three feature sets:
  base      -> no text feature (must reproduce champion LK-BP 0.42807)
  titles    -> literature_infame/text_scores_infame.parquet (desc+func+precut titles)
  abstract  -> lk_abstract_fold/abstract_text_scores.parquet (titles + precut abstracts)

Reuses the EXACT building blocks (feature list, lit-feature engineering, LightGBM
params/seed, pminmax LK-BP normalization) from literature_infame/train_rerank_lit.py
-> clean A/B/C, only the text-score source differs. LK ONLY.
"""
import os
import sys
import json
import time
import pandas as pd
import lightgbm as lgb

HERE = os.path.dirname(os.path.abspath(__file__))
LITINFAME = os.path.join(HERE, "..", "literature_infame")
sys.path.insert(0, LITINFAME)
import train_rerank_lit as T  # noqa: E402

TITLES_SCORES = os.path.join(LITINFAME, "text_scores_infame.parquet")
ABSTRACT_SCORES = os.path.join(HERE, "abstract_text_scores.parquet")
VARIANTS = {
    "base": None,
    "titles": TITLES_SCORES,
    "abstract": ABSTRACT_SCORES,
}


def train_lk(variant, scores_path):
    cat = "lk"
    ddir = T.DATASET_DIR[cat]
    train_parquet = os.path.join(ddir, "train.parquet")
    eval_parquet = os.path.join(ddir, "eval.parquet")
    base_feats, bool_cols, read_cols = T.base_feature_list(train_parquet)
    use_lit = scores_path is not None
    text_df = pd.read_parquet(scores_path) if use_lit else None
    feats = base_feats + (T.LIT_FEATS if use_lit else [])

    t0 = time.time()
    df = T.load_category(train_parquet, cat, bool_cols, read_cols, base_feats,
                         text_df if use_lit else None)
    tr = df[df["snapshot_pair"] != T.VALID_PAIR]
    va = df[df["snapshot_pair"] == T.VALID_PAIR]
    print(f"[lk {variant}] train={len(tr)} pos={int(tr.label.sum())} "
          f"valid={len(va)} (load {time.time()-t0:.1f}s)", flush=True)
    dtr = lgb.Dataset(tr[feats], label=tr["label"].astype("int8").values,
                      free_raw_data=False)
    dva = lgb.Dataset(va[feats], label=va["label"].astype("int8").values,
                      reference=dtr, free_raw_data=False)
    params = dict(objective="binary", metric=["auc", "average_precision"],
                  learning_rate=0.05, num_leaves=63, min_child_samples=100,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                  max_depth=-1, seed=T.SEED, verbosity=-1, num_threads=12)
    booster = lgb.train(params, dtr, num_boost_round=3000,
                        valid_sets=[dva], valid_names=["valid"],
                        callbacks=[lgb.early_stopping(80), lgb.log_evaluation(0)])
    best_it = booster.best_iteration
    print(f"[lk {variant}] best_iter={best_it} "
          f"valid_auc={booster.best_score['valid']['auc']:.5f} "
          f"valid_ap={booster.best_score['valid']['average_precision']:.5f}",
          flush=True)
    del df, tr, va, dtr, dva

    te = T.load_category(eval_parquet, cat, bool_cols, read_cols, base_feats,
                         text_df if use_lit else None)
    te["score"] = booster.predict(te[feats], num_iteration=best_it)
    te = T.apply_pminmax_lk_bpo(te)
    bp = te[te["aspect"] == "bpo"][["protein_accession", "go_term_id", "score"]]
    bp = bp.rename(columns={"protein_accession": "acc", "go_term_id": "go",
                            "score": "s_rerank"})
    imp = dict(zip(feats, booster.feature_importance(importance_type="gain")))
    booster.save_model(os.path.join(HERE, f"model_lk_{variant}.txt"),
                       num_iteration=best_it)
    bp.to_parquet(os.path.join(HERE, f"rerank_bp_lk_{variant}.parquet"))
    lit_imp = {k: round(float(imp.get(k, 0)), 1) for k in T.LIT_FEATS} if use_lit else {}
    top = {k: round(float(v), 1) for k, v in
           sorted(imp.items(), key=lambda x: -x[1])[:15]}
    return {"best_iter": best_it,
            "valid_auc": round(float(booster.best_score["valid"]["auc"]), 5),
            "lit_importance": lit_imp, "top_importance": top}


def main():
    todo = sys.argv[1:] or list(VARIANTS)
    meta = {}
    for v in todo:
        meta[v] = train_lk(v, VARIANTS[v])
    out = os.path.join(HERE, "train_meta_lk.json")
    prev = json.load(open(out)) if os.path.exists(out) else {}
    prev.update(meta)
    json.dump(prev, open(out, "w"), indent=2)
    print("done", list(meta), flush=True)


if __name__ == "__main__":
    main()
