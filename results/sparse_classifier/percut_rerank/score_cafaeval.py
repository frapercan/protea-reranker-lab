"""Score the per-category TEST predictions with cafaeval -> 9-cell f_micro_w.

Replicates: -ia IA.tsv -prop fill -norm cafa -no_orphans -toi <toi> [-known PK_known].
Metric reported per (category, aspect) = best f_micro_w (IA-weighted micro-Fmax).
"""
import os, json
from cafaeval.evaluation import cafa_eval

HERE = os.path.dirname(os.path.abspath(__file__))
PRED_DIR = os.path.join(HERE, "predictions")
GT_DIR = "/home/frapercan/Thesis2/repositories/protea-reranker-lab/results/sparse_classifier/lafa_gt"
OBO = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
IA = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv"
TOI = os.path.join(GT_DIR, "groundtruth_terms_of_interest.txt")

NS2ASP = {"biological_process": "bpo", "molecular_function": "mfo",
          "cellular_component": "cco"}

GT = {
    "nk": (os.path.join(GT_DIR, "groundtruth_NK.tsv"), None),
    "lk": (os.path.join(GT_DIR, "groundtruth_LK.tsv"), None),
    "pk": (os.path.join(GT_DIR, "groundtruth_PK.tsv"),
           os.path.join(GT_DIR, "groundtruth_PK_known.tsv")),
}


def score_category(cat):
    gt_file, known = GT[cat]
    pred_dir = os.path.join(PRED_DIR, cat)
    df, dfs_best = cafa_eval(
        OBO, pred_dir, gt_file,
        ia=IA, no_orphans=True, norm="cafa", prop="fill",
        exclude=known, toi_file=TOI, th_step=0.01, n_cpu=8,
    )
    best = dfs_best["f_micro_w"]
    cells = {}
    # index levels: filename, ns, tau ; column f_micro_w
    bi = best.reset_index()
    for _, row in bi.iterrows():
        asp = NS2ASP.get(row["ns"])
        if asp:
            cells[asp] = {
                "f_micro_w": float(row["f_micro_w"]),
                "tau": float(row["tau"]),
                "pr_micro_w": float(row.get("pr_micro_w", float("nan"))),
                "rc_micro_w": float(row.get("rc_micro_w", float("nan"))),
                "cov": float(row.get("cov", float("nan"))),
            }
    for asp in ("mfo", "bpo", "cco"):
        cells.setdefault(asp, None)
    return cells


def main():
    result = {}
    for cat in ("nk", "lk", "pk"):
        print("=" * 30, "scoring", cat)
        result[cat] = score_category(cat)
        print(cat, {a: (c["f_micro_w"] if c else None) for a, c in result[cat].items()})
    with open(os.path.join(HERE, "result_9cell.json"), "w") as f:
        json.dump(result, f, indent=2)
    print("written result_9cell.json")


if __name__ == "__main__":
    main()
