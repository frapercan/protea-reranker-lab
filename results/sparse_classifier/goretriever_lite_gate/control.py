"""Control: is the trained retriever's AUROC gain ORTHOGONAL literature signal, or
just the GO-term base-rate / term-frequency prior (a candidate-pool artifact GATE 0
flagged, already implicit in the reranker) leaking through the shared GO embedding?

We compute, per hard-BP category, the AUROC of a pure per-GO base-rate predictor
(each candidate scored by its GO term's positive rate) with OOF-by-protein estimation
(GO pos-rate computed on train proteins only). If this alone approaches the trained
retriever AUROC, the "retriever" is a term-frequency prior, not literature retrieval.
Also reports correlation of the trained OOF score with the per-GO base rate.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
R = HERE.parent
HARD = R / "phaseA_separability" / "hard_frame.parquet"


def main():
    hard = pd.read_parquet(HARD)
    hard = hard[(hard.aspect == "bpo") & (hard.category.isin(["lk", "pk"]))].copy()
    oof = pd.read_parquet(HERE / "retriever_oof_scores.parquet")
    oof = oof.rename(columns={"acc": "protein_accession", "go": "go_term_id"})
    df = hard.merge(oof[["protein_accession", "go_term_id", "cat", "text_score_raw"]],
                    on=["protein_accession", "go_term_id"], how="inner")

    out = {}
    for c in ["lk", "pk"]:
        d = df[df.cat == c].reset_index(drop=True)
        y = d.label.to_numpy().astype(int)
        groups = d.protein_accession.to_numpy()
        go = d.go_term_id.to_numpy()
        # OOF per-GO base rate (train-fold pos rate; global prior fallback)
        base = np.zeros(len(d), dtype=float)
        gkf = GroupKFold(n_splits=min(5, len(np.unique(groups))))
        for tr, te in gkf.split(d, y, groups):
            gp = pd.Series(y[tr]).groupby(go[tr]).mean()
            prior = y[tr].mean()
            base[te] = [gp.get(g, prior) for g in go[te]]
        auroc_base = round(float(roc_auc_score(y, base)), 4)
        auroc_trained = round(float(roc_auc_score(y, d.text_score_raw.to_numpy())), 4)
        corr = round(float(np.corrcoef(base, d.text_score_raw.to_numpy())[0, 1]), 4)
        out[c] = {"auroc_per_go_baserate": auroc_base,
                  "auroc_trained_retriever": auroc_trained,
                  "spearmanlike_corr_trained_vs_baserate": corr,
                  "n": int(len(d)), "pos": int(y.sum())}
        print(f"{c.upper()}: per-GO base-rate AUROC {auroc_base} | trained {auroc_trained} "
              f"| corr(trained,baserate) {corr}", flush=True)
    json.dump(out, open(HERE / "control_baserate.json", "w"), indent=2)
    print("wrote control_baserate.json", flush=True)


if __name__ == "__main__":
    main()
