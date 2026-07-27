"""Build the LK union pool = champion LK pool (baseline-M2 dataset, knn + M2-clf +
assoc + interpro candidates) UNIONED with the two-tower sparse-classifier BP
candidates (from the percut dataset, the same candidates that raised the LK-bpo
recall ceiling 0.767 -> 0.823 in p4_recall_ceiling.json).

Temporal honesty: the percut two-tower candidates+scores are built per-cut (no
future leakage) by build_per_cut_codes.py, so every (acc, go, snapshot_pair) score
respects that snapshot's t0. We only join on (acc, go, snapshot_pair); no future
snapshot codes ever touch a row.

The two-tower signal enters as a SEPARATE feature `tt_clf_score` (the baseline
`classifier_score` column stays M2, so we never overwrite the champion feature).
For two-tower candidates absent from the champion pool we APPEND new rows
(retrieval features NaN, M2 classifier_score=0 -> the 0.391 union design).
Added features on every BP row: tt_clf_score, tt_has_clf, tt_clf_rank.

Writes union_lk_train.parquet / union_lk_eval.parquet (78 baseline cols + 3 tt).
"""
import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

SC = "/home/frapercan/Thesis2/repositories/protea-reranker-lab/results/sparse_classifier"
HERE = os.path.dirname(os.path.abspath(__file__))
BASELINE = os.path.join(SC, "percut_rerank", "baseline")   # LK champion (M2)
PERCUT = os.path.join(SC, "percut_rerank")                 # two-tower candidates
KEY = ["protein_accession", "go_term_id", "snapshot_pair"]


def build(split):
    base_path = os.path.join(BASELINE, f"{split}.parquet")
    perc_path = os.path.join(PERCUT, f"{split}.parquet")
    # champion LK pool (all aspects; aspect is a reranker feature)
    base = pq.read_table(base_path, filters=[("category", "=", "lk")]).to_pandas()
    cols = list(base.columns)
    # two-tower BP candidates (classifier_present rows) from the percut dataset
    perc = pq.read_table(
        perc_path,
        filters=[("category", "=", "lk"), ("aspect", "=", "bpo")],
    ).to_pandas()
    perc = perc[perc["classifier_present"].astype(bool)].copy()
    print(f"[{split}] baseline LK rows={len(base)} | two-tower BP cand rows={len(perc)}",
          flush=True)

    # two-tower score lookup over (acc, go, snapshot)
    tt = perc[KEY + ["classifier_score"]].rename(columns={"classifier_score": "tt_clf_score"})
    tt = tt.groupby(KEY, as_index=False)["tt_clf_score"].max()

    # attach tt score to existing champion rows
    base = base.merge(tt, on=KEY, how="left")
    base_keys = set(map(tuple, base[KEY].itertuples(index=False, name=None)))

    # rows to APPEND = two-tower candidates not already in the champion pool
    perc_keyed = perc.set_index(KEY)
    perc_new_mask = ~perc[KEY].apply(tuple, axis=1).isin(base_keys)
    add = perc[perc_new_mask].copy()
    print(f"[{split}] appended new two-tower-only rows={len(add)} "
          f"(pos={int(add.label.sum())})", flush=True)
    # reschema appended rows into baseline columns: M2 classifier_score/present -> 0,
    # tt_clf_score = the two-tower logit. All other (retrieval/freq) cols come from
    # the percut row as-built (retrieval features are NaN for clf-only candidates).
    add = add.reindex(columns=cols)            # keep the 78 baseline cols, same order
    add["tt_clf_score"] = perc.loc[perc_new_mask, "classifier_score"].values
    add["classifier_score"] = 0.0
    add["classifier_present"] = False

    union = pd.concat([base, add], ignore_index=True)
    union["tt_clf_score"] = union["tt_clf_score"].fillna(0.0).astype("float32")
    union["tt_has_clf"] = (union["tt_clf_score"].abs() > 0).astype("int8")
    # within-(protein, snapshot) rank of the two-tower score over BP candidates
    union["tt_clf_rank"] = 0.0
    is_bp = (union["aspect"] == "bpo") & (union["tt_has_clf"] == 1)
    if is_bp.any():
        sub = union.loc[is_bp, ["protein_accession", "snapshot_pair", "tt_clf_score"]]
        rank = sub.groupby(["protein_accession", "snapshot_pair"])["tt_clf_score"] \
                  .rank(ascending=False, method="first")
        union.loc[is_bp, "tt_clf_rank"] = rank.values
    union["tt_clf_rank"] = union["tt_clf_rank"].astype("float32")

    out = os.path.join(HERE, f"union_lk_{split}.parquet")
    union.to_parquet(out)
    print(f"[{split}] union LK rows={len(union)} pos={int(union.label.sum())} -> {out}",
          flush=True)
    # quick BP pool stats
    bp = union[union.aspect == "bpo"]
    print(f"[{split}] BP union rows={len(bp)} pos={int(bp.label.sum())} "
          f"tt_has_clf rows={int((bp.tt_has_clf==1).sum())}", flush=True)


if __name__ == "__main__":
    for split in ("train", "eval"):
        build(split)
