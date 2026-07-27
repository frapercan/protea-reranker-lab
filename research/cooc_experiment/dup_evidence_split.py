"""The pool concatenates its two candidate generators instead of merging them.

40,192 (protein,term) pairs appear TWICE in the pk-bpo pool: once as a knn-only row and
once as a classifier-only row. Only 9 rows in 616,223 ever carry both flags. So when the
KNN and the classifier AGREE on a candidate, the pipeline does not record agreement: it
records two half-views, each blind to the other's evidence. This measures what that costs.
"""
import collections, json
import numpy as np, pyarrow.parquet as pq
from pathlib import Path
DS = "/home/frapercan/Thesis2/repositories/protea-reranker-lab/datasets/protst-global-train227-test230/eval.parquet"
cols = ["category", "aspect", "protein_accession", "go_term_id", "label",
        "knn_present", "classifier_present", "distance", "classifier_score",
        "vote_count", "neighbor_vote_fraction"]
d = pq.read_table(DS, columns=cols)
c = np.asarray(d.column("category").to_pylist()); a = np.asarray(d.column("aspect").to_pylist())
m = (c == "pk") & (a == "bpo")
P = np.asarray(d.column("protein_accession").to_pylist())[m]
G = np.asarray(d.column("go_term_id").to_pylist())[m]
num = {k: d.column(k).to_numpy(zero_copy_only=False).astype(float)[m]
       for k in ("label", "knn_present", "classifier_present", "distance",
                 "classifier_score", "vote_count", "neighbor_vote_fraction")}
keys = list(zip(P, G))
cnt = collections.Counter(keys)
dupset = {k for k, v in cnt.items() if v > 1}          # built ONCE
isdup = np.fromiter((k in dupset for k in keys), bool, len(keys))
lab = num["label"] > 0
res = {}
res["dup_pairs"] = len(dupset); res["dup_rows"] = int(isdup.sum())
res["pos_rate_agreed"] = round(float(lab[isdup].mean()), 4)
res["pos_rate_single"] = round(float(lab[~isdup].mean()), 4)
res["lift"] = round(res["pos_rate_agreed"] / max(res["pos_rate_single"], 1e-9), 2)
print(f"duplicated pairs {len(dupset):,} / rows {isdup.sum():,} ({isdup.mean():.1%})")
print(f"\n=== does AGREEMENT between the two generators predict truth? ===")
print(f"  positive rate when BOTH proposed it (the split rows): {res['pos_rate_agreed']:.4f}")
print(f"  positive rate for single-source rows               : {res['pos_rate_single']:.4f}")
print(f"  -> agreement is {res['lift']}x more likely to be TRUE")
k0 = next(iter(dupset))
idx = [i for i, k in enumerate(keys) if k == k0][:2]
print(f"\n=== the same candidate, {k0[0]} / {k0[1]} ===")
for i in idx:
    print(f"  knn={num['knn_present'][i]:.0f} clf={num['classifier_present'][i]:.0f} "
          f"dist={num['distance'][i]:.4f} clf_score={num['classifier_score'][i]:.4f} "
          f"votes={num['vote_count'][i]:.0f} label={num['label'][i]:.0f}")
print("  -> each row carries HALF the evidence; neither sees both sources.")
json.dump(res, open("/home/frapercan/Thesis2/storage/cooc_experiment/dup_evidence_split.json", "w"), indent=1)
print("DONE", flush=True)
