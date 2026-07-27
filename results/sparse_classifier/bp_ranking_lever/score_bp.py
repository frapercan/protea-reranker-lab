"""Score a TEST category's BP re-ranking via cafa_eval -> bpo f_micro_w.

Reassembles full pred file (non-BP rows unchanged + re-scored BP rows), writes it to a
temp pred dir, runs cafa_eval with the board-faithful settings, returns the bpo cell.
"""
import os, sys, json, tempfile, shutil
import numpy as np
import pandas as pd
from cafaeval.evaluation import cafa_eval

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

GT_DIR = "/home/frapercan/Thesis2/repositories/protea-reranker-lab/results/sparse_classifier/lafa_gt"
GT = {
    "lk": (os.path.join(GT_DIR, "groundtruth_LK.tsv"), None),
    "pk": (os.path.join(GT_DIR, "groundtruth_PK.tsv"),
           os.path.join(GT_DIR, "groundtruth_PK_known.tsv")),
}
TOI = os.path.join(GT_DIR, "groundtruth_terms_of_interest.txt")
HERE = os.path.dirname(os.path.abspath(__file__))


def score_bp_cell(cat, bp_df, score_col):
    """bp_df: protein_accession, go_term_id, <score_col>. Returns bpo cell dict."""
    non_bp = pd.read_parquet(os.path.join(HERE, f"test_{cat}_nonbp.parquet"))
    bp_out = bp_df[["protein_accession", "go_term_id", score_col]].rename(
        columns={score_col: "score"})
    nb_out = non_bp[["protein_accession", "go_term_id", "base_score"]].rename(
        columns={"base_score": "score"})
    full = pd.concat([nb_out, bp_out], ignore_index=True)
    full = full[full["score"] > 0.0]  # cafa convention; sparse

    tmp = tempfile.mkdtemp(prefix=f"bp_{cat}_")
    try:
        full.to_csv(os.path.join(tmp, f"{cat}.tsv"), sep="\t", header=False,
                    index=False, float_format="%.6f")
        gt_file, known = GT[cat]
        _, dfs_best = cafa_eval(
            C.OBO, tmp, gt_file, ia=C.IA_PATH, no_orphans=True, norm="cafa",
            prop="fill", exclude=known, toi_file=TOI, th_step=0.01, n_cpu=8)
        best = dfs_best["f_micro_w"].reset_index()
        row = best[best["ns"] == "biological_process"]
        if len(row) == 0:
            return None
        row = row.iloc[0]
        return {"f_micro_w": float(row["f_micro_w"]), "tau": float(row["tau"]),
                "pr_micro_w": float(row.get("pr_micro_w", float("nan"))),
                "rc_micro_w": float(row.get("rc_micro_w", float("nan"))),
                "cov": float(row.get("cov", float("nan")))}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
