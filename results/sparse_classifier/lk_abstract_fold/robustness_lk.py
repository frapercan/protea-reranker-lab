"""Seed-robustness of the titles-vs-abstract ordering on board-faithful LK-BP.
For each seed, retrain the LK reranker with the TITLES text feature and with the
ABSTRACT text feature, score board-faithful, record both and their delta.
Establishes whether abstract < titles is a stable ordering or seed noise.
"""
import os
import sys
import json
import pandas as pd
import lightgbm as lgb

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "literature_infame"))
import train_rerank_lit as T  # noqa: E402
import graft_score as G  # noqa: E402

TITLES = os.path.join(HERE, "..", "literature_infame", "text_scores_infame.parquet")
ABSTRACT = os.path.join(HERE, "abstract_text_scores.parquet")
SEEDS = [7, 123, 2024]


def train_score(seed, scores_path):
    cat = "lk"
    ddir = T.DATASET_DIR[cat]
    trp = os.path.join(ddir, "train.parquet")
    evp = os.path.join(ddir, "eval.parquet")
    base_feats, bool_cols, read_cols = T.base_feature_list(trp)
    text_df = pd.read_parquet(scores_path)
    feats = base_feats + T.LIT_FEATS
    df = T.load_category(trp, cat, bool_cols, read_cols, base_feats, text_df)
    tr = df[df["snapshot_pair"] != T.VALID_PAIR]
    va = df[df["snapshot_pair"] == T.VALID_PAIR]
    dtr = lgb.Dataset(tr[feats], label=tr["label"].astype("int8").values,
                      free_raw_data=False)
    dva = lgb.Dataset(va[feats], label=va["label"].astype("int8").values,
                      reference=dtr, free_raw_data=False)
    params = dict(objective="binary", metric=["auc", "average_precision"],
                  learning_rate=0.05, num_leaves=63, min_child_samples=100,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                  max_depth=-1, seed=seed, verbosity=-1, num_threads=12)
    booster = lgb.train(params, dtr, num_boost_round=3000, valid_sets=[dva],
                        valid_names=["valid"],
                        callbacks=[lgb.early_stopping(80), lgb.log_evaluation(0)])
    best_it = booster.best_iteration
    del df, tr, va, dtr, dva
    te = T.load_category(evp, cat, bool_cols, read_cols, base_feats, text_df)
    te["score"] = booster.predict(te[feats], num_iteration=best_it)
    te = T.apply_pminmax_lk_bpo(te)
    bp = te[te["aspect"] == "bpo"][["protein_accession", "go_term_id", "score"]]
    bp = bp.rename(columns={"protein_accession": "acc", "go_term_id": "go",
                            "score": "s_rerank"})
    u = G.union_naivemax(bp)
    return G.score_bp(list(zip(u.acc, u.go, u.s_max)), cat)["f_micro_w"]


def main():
    out = []
    for s in SEEDS:
        ft = train_score(s, TITLES)
        fa = train_score(s, ABSTRACT)
        rec = {"seed": s, "titles": ft, "abstract": fa,
               "abstract_minus_titles": round(fa - ft, 5)}
        out.append(rec)
        print(rec, flush=True)
    res = {"seeds": out,
           "mean_titles": round(sum(r["titles"] for r in out) / len(out), 5),
           "mean_abstract": round(sum(r["abstract"] for r in out) / len(out), 5),
           "mean_abstract_minus_titles":
               round(sum(r["abstract_minus_titles"] for r in out) / len(out), 5)}
    json.dump(res, open(os.path.join(HERE, "robustness_lk.json"), "w"), indent=2)
    print("MEAN titles", res["mean_titles"], "abstract", res["mean_abstract"],
          "delta", res["mean_abstract_minus_titles"], flush=True)


if __name__ == "__main__":
    main()
