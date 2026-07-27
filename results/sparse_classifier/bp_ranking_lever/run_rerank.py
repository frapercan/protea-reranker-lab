"""Decisive test: train a thin LightGBM meta-reranker on validation v225-v227 BP rows
with [base_score + term-space features] -> label, apply to TEST, score bpo via cafa_eval.

If term features add real signal beyond base_score, this should beat baseline on test.
Protein-grouped split for early stopping; meta-model never sees test.
"""
import os, sys, json, time
import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from score_bp import score_bp_cell

HERE = os.path.dirname(os.path.abspath(__file__))
FEATS_FULL = ["base_score", "profile_fit", "topk_coherence", "depth", "ia"]
FEATS_TERMONLY = ["profile_fit", "topk_coherence", "depth", "ia"]
SEED = 42


def train_meta(valid, feats):
    prots = valid["protein_accession"].unique()
    rng = np.random.RandomState(SEED)
    rng.shuffle(prots)
    cut = int(0.5 * len(prots))
    tr_p = set(prots[:cut]);
    tr = valid[valid["protein_accession"].isin(tr_p)]
    va = valid[~valid["protein_accession"].isin(tr_p)]
    dtr = lgb.Dataset(tr[feats], label=tr["label"].astype("int8").values)
    dva = lgb.Dataset(va[feats], label=va["label"].astype("int8").values, reference=dtr)
    params = dict(objective="binary", metric="average_precision", learning_rate=0.05,
                  num_leaves=31, min_child_samples=50, feature_fraction=0.9,
                  bagging_fraction=0.9, bagging_freq=1, seed=SEED, verbosity=-1,
                  num_threads=12)
    booster = lgb.train(params, dtr, num_boost_round=2000, valid_sets=[dva],
                        valid_names=["valid"], callbacks=[lgb.early_stopping(60)])
    return booster


def main():
    results = {}
    for cat in ["lk", "pk"]:
        print(f"\n########## {cat.upper()} meta-rerank ##########")
        valid = pd.read_parquet(os.path.join(HERE, f"valid_{cat}_bp_feat.parquet"))
        test = pd.read_parquet(os.path.join(HERE, f"test_{cat}_bp_feat.parquet"))
        cat_res = {}
        for name, feats in [("full", FEATS_FULL), ("term_only", FEATS_TERMONLY)]:
            booster = train_meta(valid, feats)
            imp = dict(zip(feats, booster.feature_importance(importance_type="gain")))
            test["_m"] = booster.predict(test[feats], num_iteration=booster.best_iteration)
            cell = score_bp_cell(cat, test, "_m")
            cat_res[name] = {"cell": cell, "gain_importance": {k: float(v) for k, v in imp.items()},
                             "best_iter": int(booster.best_iteration)}
            print(f"  [{name}] TEST bpo f={cell['f_micro_w']:.4f} tau={cell['tau']:.2f} "
                  f"pr={cell['pr_micro_w']:.3f} rc={cell['rc_micro_w']:.3f}")
            print(f"     gain: {{ " + ", ".join(f'{k}:{v:.0f}' for k,v in imp.items()) + " }")
        results[cat] = cat_res
    json.dump(results, open(os.path.join(HERE, "rerank_meta.json"), "w"), indent=2)
    print("\nwrote rerank_meta.json")


if __name__ == "__main__":
    main()
