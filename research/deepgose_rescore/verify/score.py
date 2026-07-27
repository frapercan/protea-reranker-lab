#!/usr/bin/env python
import sys, json
from cafaeval.evaluation import cafa_eval

OBO = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
IA = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv"
TOI = "/home/frapercan/Thesis2/repositories/protea-reranker-lab/results/sparse_classifier/lafa_gt/groundtruth_terms_of_interest.txt"
GT = "/home/frapercan/Thesis2/repositories/protea-reranker-lab/results/sparse_classifier/lafa_gt/groundtruth_LK.tsv"

pred_dir = sys.argv[1]
out_json = sys.argv[2]

df, dfs = cafa_eval(OBO, pred_dir, GT, ia=IA, prop="fill", norm="cafa",
                    no_orphans=True, toi_file=TOI, exclude=None, max_terms=None,
                    th_step=0.01, n_cpu=4, weighted_only=False)

d = dfs["f_micro_w"].reset_index()
d = d[d["ns"] == "biological_process"]
best = d.loc[d["f_micro_w"].idxmax()]
res = {
    "f_micro_w": float(best["f_micro_w"]),
    "tau": float(best["tau"]),
    "n_rows_ns": int(len(d)),
}
with open(out_json, "w") as f:
    json.dump(res, f)
print(json.dumps(res))
