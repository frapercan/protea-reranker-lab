"""Diagnostic: is the text arm orthogonal to the baseline on HARD proteins?
Candidate-level ranking quality (pool labels) + neighbor-distance stratification.
"""
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
TAG = os.environ.get("TAG", "spubmed")


def per_protein_ap(df, score_col):
    aps = []
    for _, g in df.groupby("protein_accession"):
        if g.label.sum() == 0 or g.label.sum() == len(g):
            continue
        aps.append(average_precision_score(g.label, g[score_col]))
    return float(np.mean(aps)) if aps else float("nan")


def main():
    ev = pd.read_parquet(
        "/home/frapercan/Thesis2/repositories/protea-reranker-lab/results/clean_227230/eval_scores.parquet")
    ev = ev[ev.aspect == "bpo"].copy()
    ts = pd.read_parquet(os.path.join(HERE, f"text_scores_bpo_{TAG}_precut.parquet"))
    ts = ts.rename(columns={"acc": "protein_accession", "go": "go_term_id"})
    m = ev.merge(ts[["protein_accession", "go_term_id", "text_score"]],
                 on=["protein_accession", "go_term_id"], how="inner")
    print("merged BP candidate rows:", len(m))

    rep = {}
    for cat in ("nk", "lk"):
        sub = m[m.category == cat].copy()
        # difficulty by neighbor KNN distance (higher = more remote/harder)
        # per-protein distance is constant per protein (KNN to nearest train)
        pdist = sub.groupby("protein_accession").distance.first()
        q = pdist.quantile([1 / 3, 2 / 3]).values
        def band(d):
            return "easy_near" if d <= q[0] else ("mid" if d <= q[1] else "hard_remote")
        sub["band"] = sub.protein_accession.map(lambda p: band(pdist[p]))

        cat_rep = {"n_proteins": int(sub.protein_accession.nunique()),
                   "n_true_cands": int(sub.label.sum()),
                   "dist_tertiles": [round(float(x), 4) for x in q],
                   "per_protein_AP": {}, "AP_by_band": {}}
        for col, nm in (("reranker_score", "reranker"),
                        ("text_score", "text"), ("distance", "neg_distance")):
            s = sub.copy()
            if col == "distance":
                s["_sc"] = -s["distance"]
                col2 = "_sc"
            else:
                col2 = col
            cat_rep["per_protein_AP"][nm] = round(per_protein_ap(s, col2), 4)
        for b in ("easy_near", "mid", "hard_remote"):
            sb = sub[sub.band == b]
            cat_rep["AP_by_band"][b] = {
                "n_prot": int(sb.protein_accession.nunique()),
                "reranker": round(per_protein_ap(sb, "reranker_score"), 4),
                "text": round(per_protein_ap(sb, "text_score"), 4),
            }
        # orthogonal recovery: true cands the reranker ranks LOW (per-protein
        # bottom 50%) that text ranks HIGH (per-protein top 25%)
        def pct_rank(g, col):
            return g[col].rank(pct=True)
        sub["rr_pct"] = sub.groupby("protein_accession").reranker_score.transform(
            lambda x: x.rank(pct=True))
        sub["tx_pct"] = sub.groupby("protein_accession").text_score.transform(
            lambda x: x.rank(pct=True))
        true = sub[sub.label == 1]
        missed_by_rr = true[true.rr_pct <= 0.5]
        recovered = missed_by_rr[missed_by_rr.tx_pct >= 0.75]
        cat_rep["orthogonal_recovery"] = {
            "true_cands": int(len(true)),
            "true_ranked_low_by_reranker(<=p50)": int(len(missed_by_rr)),
            "of_those_text_ranks_high(>=p75)": int(len(recovered)),
            "recovery_frac": round(len(recovered) / max(len(missed_by_rr), 1), 4),
        }
        rep[cat] = cat_rep
        print(cat, json.dumps(cat_rep, indent=2))

    json.dump(rep, open(os.path.join(HERE, f"diagnostic_{TAG}.json"), "w"), indent=2)
    print("written diagnostic_%s.json" % TAG)


if __name__ == "__main__":
    main()
