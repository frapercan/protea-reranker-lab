"""Standalone scoring-quality eval of the GCN joint scorer vs the frozen two-tower.

(a) recall@100 on the held-out v227 protein split (from gcn_metrics.json vs p2/metrics.json).
(b) PK-BP true-vs-false candidate RANKING on the 227->230 test: does the GCN joint score
    separate true from false PK-BP candidates better than the frozen two-tower
    classifier_score? Reports AUROC + average precision on the percut PK-BP eval rows
    (per-cut honest gcn_score), restricted to in-vocab candidates for a fair head-to-head.
"""
import os, sys, json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score, average_precision_score

HERE = os.path.dirname(os.path.abspath(__file__))
SC = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from gcn_scorer import GcnScorer  # noqa: E402


def main():
    scorer = GcnScorer()
    ev = pq.read_table(
        os.path.join(SC, "percut_rerank", "eval.parquet"),
        columns=["protein_accession", "go_term_id", "snapshot_pair", "label",
                 "classifier_score", "classifier_present", "category", "aspect"],
        filters=[("category", "=", "pk"), ("aspect", "=", "bpo")]).to_pandas()
    gs = scorer.score_frame(ev.rename(columns={"protein_accession": "acc", "go_term_id": "go"})
                            [["acc", "go", "snapshot_pair"]])
    ev["gcn_score"] = gs
    y = ev["label"].astype(int).to_numpy()
    invocab = np.isfinite(gs)
    res = {"pk_bp_eval_rows": int(len(ev)), "pos": int(y.sum()),
           "in_vocab_frac": round(float(invocab.mean()), 4)}

    # GCN over all in-vocab PK-BP candidates
    m = invocab
    res["gcn_all_invocab"] = {
        "n": int(m.sum()), "pos": int(y[m].sum()),
        "auroc": round(float(roc_auc_score(y[m], gs[m])), 5),
        "ap": round(float(average_precision_score(y[m], gs[m])), 5)}

    # head-to-head on the SAME rows where the frozen two-tower also fired
    cp = ev["classifier_present"].astype(bool).to_numpy() & invocab
    cs = ev["classifier_score"].to_numpy()
    res["head_to_head_clf_present"] = {
        "n": int(cp.sum()), "pos": int(y[cp].sum()),
        "two_tower_auroc": round(float(roc_auc_score(y[cp], cs[cp])), 5),
        "two_tower_ap": round(float(average_precision_score(y[cp], cs[cp])), 5),
        "gcn_auroc": round(float(roc_auc_score(y[cp], gs[cp])), 5),
        "gcn_ap": round(float(average_precision_score(y[cp], gs[cp])), 5)}
    # the clf-ONLY admitted candidates (not in the KNN pool) -- the dilution set
    knn = None
    try:
        knn = pq.read_table(os.path.join(SC, "percut_rerank", "eval.parquet"),
                            columns=["knn_present", "category", "aspect"],
                            filters=[("category", "=", "pk"), ("aspect", "=", "bpo")]).to_pandas()
        clf_only = (~knn["knn_present"].astype(bool).to_numpy()) & invocab
        res["gcn_on_clf_only_admitted"] = {
            "n": int(clf_only.sum()), "pos": int(y[clf_only].sum()),
            "auroc": round(float(roc_auc_score(y[clf_only], gs[clf_only])), 5)
            if y[clf_only].sum() > 0 and y[clf_only].sum() < clf_only.sum() else None,
            "ap": round(float(average_precision_score(y[clf_only], gs[clf_only])), 5)
            if y[clf_only].sum() > 0 else None}
    except Exception as e:
        res["gcn_on_clf_only_admitted"] = {"error": str(e)}

    # recall comparison
    try:
        g = json.load(open(os.path.join(HERE, "gcn_metrics.json")))
        res["recall_at_k_gcn"] = {k: g["recall_at_k"][k]["ensemble"] for k in ["50", "100", "200"]}
        res["recall_at_k_two_tower"] = g.get("two_tower_recall_at_k", {})
    except Exception as e:
        res["recall_note"] = str(e)

    json.dump(res, open(os.path.join(HERE, "eval_standalone.json"), "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
