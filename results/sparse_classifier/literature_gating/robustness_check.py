"""Robustness probe: does a strongly-regularized, monotone-in-evidence gate (which
cannot distort the score below the naivemax ordering as freely) transfer to test any
better than the flexible gate? If even this regresses, the negative is robust.
Trains on ALL validation, scores TEST bpo board-faithfully.
"""
import os, sys, json, tempfile, shutil
import numpy as np
import pandas as pd
import lightgbm as lgb

HERE = os.path.dirname(os.path.abspath(__file__))
SC = "/home/frapercan/Thesis2/repositories/protea-reranker-lab/results/sparse_classifier"
from cafaeval.evaluation import cafa_eval  # noqa: E402
OBO = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
IA_PATH = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv"
TOI_FILE = os.path.join(SC, "lafa_gt", "groundtruth_terms_of_interest.txt")
GT_DIR = os.path.join(SC, "lafa_gt")
GT = {"lk": (os.path.join(GT_DIR, "groundtruth_LK.tsv"), None),
      "pk": (os.path.join(GT_DIR, "groundtruth_PK.tsv"),
             os.path.join(GT_DIR, "groundtruth_PK_known.tsv"))}
BASE_FEATS = ["s_rerank", "s_interpro", "s_max", "has_rerank", "has_interpro",
              "ia", "depth", "prot_max_rerank", "prot_n_cand"]
LIT_FEATS = ["text_score", "text_cos", "text_rank_pct", "rerank_rank_pct",
             "rank_gap", "text_top1", "text_top3", "text_top5", "has_text"]
FEATS = BASE_FEATS + LIT_FEATS
MONO = [1 if f in ("s_rerank", "s_interpro", "s_max", "text_score") else 0 for f in FEATS]
SEED = 42


def bpo_cell(rows, cat):
    gt, known = GT[cat]
    d = tempfile.mkdtemp(prefix="rob_")
    pd.DataFrame(rows, columns=["acc", "go", "score"]).to_csv(
        os.path.join(d, f"{cat}.tsv"), sep="\t", header=False, index=False, float_format="%.6f")
    _, best = cafa_eval(OBO, d, gt, ia=IA_PATH, no_orphans=True, norm="cafa", prop="fill",
                        exclude=known, toi_file=TOI_FILE, th_step=0.01, n_cpu=8)
    shutil.rmtree(d, ignore_errors=True)
    b = best["f_micro_w"].reset_index()
    for _, r in b.iterrows():
        if r["ns"] == "biological_process":
            return round(float(r["f_micro_w"]), 5)
    return 0.0


def main():
    res = {}
    for cat in ["lk", "pk"]:
        vu = pd.read_parquet(os.path.join(HERE, f"valid_union_{cat}.parquet"))
        tu = pd.read_parquet(os.path.join(HERE, f"test_union_{cat}.parquet"))
        rng = np.random.RandomState(SEED); prots = vu.acc.unique().copy(); rng.shuffle(prots)
        es = set(prots[int(0.8 * len(prots)):])
        a = vu[~vu.acc.isin(es)]; b = vu[vu.acc.isin(es)]
        dtr = lgb.Dataset(a[FEATS], label=a.label.astype("int8").values)
        dva = lgb.Dataset(b[FEATS], label=b.label.astype("int8").values, reference=dtr)
        params = dict(objective="binary", metric="average_precision", learning_rate=0.03,
                      num_leaves=7, min_child_samples=500, feature_fraction=0.7,
                      bagging_fraction=0.7, bagging_freq=1, lambda_l2=10.0,
                      monotone_constraints=MONO, seed=SEED, verbosity=-1, num_threads=12)
        g = lgb.train(params, dtr, num_boost_round=2000, valid_sets=[dva],
                      valid_names=["v"], callbacks=[lgb.early_stopping(60, verbose=False)])
        pr = g.predict(tu[FEATS], num_iteration=g.best_iteration)
        f_reg = bpo_cell(list(zip(tu.acc, tu.go, pr)), cat)
        f_nm = bpo_cell(list(zip(tu.acc, tu.go, tu.s_max)), cat)
        res[cat] = {"naivemax_test_bpo": f_nm, "regularized_monotone_lit_test_bpo": f_reg}
        print(f"[{cat}] naivemax={f_nm}  regularized_monotone_lit={f_reg}")
    json.dump(res, open(os.path.join(HERE, "robustness_check.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
