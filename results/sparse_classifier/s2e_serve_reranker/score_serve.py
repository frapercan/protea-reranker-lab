"""Board-faithful 9-cell scoring of the serve-producible reranker.

Same cafa_eval invocation as percut_rerank/score_cafaeval.py
(-ia IA.tsv -prop fill -norm cafa -no_orphans -toi <toi>, -known PK_known
for PK). Metric = f_micro_w. Writes result_9cell.json comparing the
retrained serve-producible trio against the champion (graft, no interpro)
and the full champion (with interpro BP graft).
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

# Champion references (from percut_rerank + interpro2go_test).
# graft = composite reranker WITHOUT interpro (baseline NK/LK + percut PK).
CHAMP_GRAFT = {
    "nk": {"mfo": 0.66368, "bpo": 0.33108, "cco": 0.47696},
    "lk": {"mfo": 0.56384, "bpo": 0.35119, "cco": 0.46147},
    "pk": {"mfo": 0.24163, "bpo": 0.14024, "cco": 0.26554},
}
# full champion = graft + InterPro BP-only graft (naivemax_bponly).
CHAMP_FULL = {
    "nk": {"mfo": 0.6637, "bpo": 0.3375, "cco": 0.477},
    "lk": {"mfo": 0.5638, "bpo": 0.4281, "cco": 0.4615},
    "pk": {"mfo": 0.2416, "bpo": 0.218, "cco": 0.2655},
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
    for _, row in best.reset_index().iterrows():
        asp = NS2ASP.get(row["ns"])
        if asp:
            cells[asp] = {"f_micro_w": float(row["f_micro_w"]),
                          "tau": float(row["tau"]),
                          "cov": float(row.get("cov", float("nan")))}
    for asp in ("mfo", "bpo", "cco"):
        cells.setdefault(asp, None)
    return cells


def main():
    result = {}
    for cat in ("nk", "lk", "pk"):
        print("=" * 30, "scoring", cat)
        result[cat] = score_category(cat)
        print(cat, {a: (round(c["f_micro_w"], 4) if c else None) for a, c in result[cat].items()})

    flat = []
    dvs_graft = {}
    dvs_full = {}
    for cat in ("nk", "lk", "pk"):
        dvs_graft[cat] = {}
        dvs_full[cat] = {}
        for asp in ("mfo", "bpo", "cco"):
            v = result[cat][asp]["f_micro_w"] if result[cat][asp] else None
            flat.append(v)
            dvs_graft[cat][asp] = round(v - CHAMP_GRAFT[cat][asp], 4) if v is not None else None
            dvs_full[cat][asp] = round(v - CHAMP_FULL[cat][asp], 4) if v is not None else None
    mean = sum(x for x in flat if x is not None) / 9.0
    mean_graft = sum(CHAMP_GRAFT[c][a] for c in CHAMP_GRAFT for a in CHAMP_GRAFT[c]) / 9.0
    mean_full = sum(CHAMP_FULL[c][a] for c in CHAMP_FULL for a in CHAMP_FULL[c]) / 9.0
    out = {
        "serve_producible_9cell": result,
        "champion_graft_no_interpro": CHAMP_GRAFT,
        "champion_full_with_interpro_bpgraft": CHAMP_FULL,
        "delta_vs_graft": dvs_graft,
        "delta_vs_full_champion": dvs_full,
        "mean_serve_producible": round(mean, 4),
        "mean_champion_graft": round(mean_graft, 4),
        "mean_champion_full": round(mean_full, 4),
        "lost_to_interpro_graft_mean": round(mean_graft - mean_full, 4),
    }
    with open(os.path.join(HERE, "result_9cell.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("mean serve-producible:", round(mean, 4),
          "| champion graft:", round(mean_graft, 4),
          "| champion full:", round(mean_full, 4))
    print("written result_9cell.json")


if __name__ == "__main__":
    main()
