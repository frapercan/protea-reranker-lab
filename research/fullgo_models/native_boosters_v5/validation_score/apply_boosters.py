"""Apply the native_boosters_v5 per-category LightGBM boosters to the validation
frame (220->227) eval.parquet and emit per-category prediction TSVs + GT TSVs for
cafaeval. Memory-safe: streams row-group batches, accumulates per-category, keeps
only the 69 numeric features in summary.json order.

Prediction collapse: the candidate set has duplicate (protein, term) rows (union of
KNN / classifier / self-prior / association streams). We collapse to ONE score per
(protein, term) by MAX, matching the offline seal which keeps the max per pair at
feature-load time (load_knn / load_clf keep max).

GT derivation: label==1 rows = the CAFA ground truth (leaf terms) for this frame,
partitioned by category. cafaeval propagates GT to ancestors (prop=fill / propagate).
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

import lightgbm as lgb
import numpy as np
import pyarrow.parquet as pq

HERE = "/home/frapercan/Thesis2/storage/fullgo_models/native_boosters_v5"
EVAL = f"{HERE}/validation_score/eval.parquet"
OUT = f"{HERE}/validation_score"
CATS = ("nk", "lk", "pk")

with open(f"{HERE}/summary.json") as fh:
    SUMMARY = json.load(fh)
FEATURES = SUMMARY["features"]  # 69 numeric, exact booster order
assert len(FEATURES) == 69, len(FEATURES)

BOOSTERS = {
    c: lgb.Booster(model_file=f"{HERE}/ensemble_gbm_{c.upper()}.txt") for c in CATS
}
for c in CATS:
    nfeat = BOOSTERS[c].num_feature()
    assert nfeat == 69, f"{c} booster expects {nfeat} features"


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    pf = pq.ParquetFile(EVAL)
    # per (cat): dict (prot,term) -> max score ; and gt set (prot,term) for label==1
    pred: dict[str, dict] = {c: {} for c in CATS}
    gt: dict[str, set] = {c: set() for c in CATS}
    n = 0
    cols = ["protein_accession", "go_term_id", "label", "category"] + FEATURES
    for batch in pf.iter_batches(batch_size=400_000, columns=cols):
        d = batch.to_pydict()
        cat = np.asarray([str(x) for x in d["category"]])
        prot = d["protein_accession"]
        term = d["go_term_id"]
        lab = np.asarray(d["label"], dtype=np.int8)
        # build feature matrix (float32); bools -> 0/1
        X = np.empty((len(cat), len(FEATURES)), dtype=np.float32)
        for j, f in enumerate(FEATURES):
            col = d[f]
            arr = np.asarray(col)
            if arr.dtype == object or arr.dtype == bool:
                X[:, j] = np.asarray([float(v) if v is not None else np.nan for v in col],
                                     dtype=np.float32)
            else:
                X[:, j] = arr.astype(np.float32)
        for c in CATS:
            m = cat == c
            if not m.any():
                continue
            idx = np.nonzero(m)[0]
            scores = BOOSTERS[c].predict(X[idx], num_threads=2)
            pc = pred[c]
            gc_ = gt[c]
            for k, s in zip(idx.tolist(), scores.tolist()):
                key = (prot[k], term[k])
                if key not in pc or s > pc[key]:
                    pc[key] = s
                if lab[k] == 1:
                    gc_.add(key)
        n += len(cat)
        print(f"  ...{n} rows", flush=True)
        del d, X, cat
    # write per-category pred + gt
    stats = {}
    for c in CATS:
        pp = f"{OUT}/pred_{c}.tsv"
        gg = f"{OUT}/gt_{c}.tsv"
        with open(pp, "w") as w:
            for (p, t), s in pred[c].items():
                w.write(f"{p}\t{t}\t{s:.6f}\n")
        with open(gg, "w") as w:
            for (p, t) in sorted(gt[c]):
                w.write(f"{p}\t{t}\n")
        stats[c] = {"pred_pairs": len(pred[c]), "gt_pairs": len(gt[c]),
                    "pred_proteins": len({p for p, _ in pred[c]}),
                    "gt_proteins": len({p for p, _ in gt[c]})}
        print(f"{c}: pred_pairs={stats[c]['pred_pairs']} gt_pairs={stats[c]['gt_pairs']} "
              f"pred_prot={stats[c]['pred_proteins']} gt_prot={stats[c]['gt_proteins']}", flush=True)
    with open(f"{OUT}/apply_stats.json", "w") as w:
        json.dump(stats, w, indent=2)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
