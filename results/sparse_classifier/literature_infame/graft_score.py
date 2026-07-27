"""Champion-graft scoring harness (board-faithful, BP only).

champion BP cell = naivemax( reranker_BP_test  UNION  InterPro2GO_BP_test ),
scored with cafa_eval (OBO, IA, TOI, prop=fill, norm=cafa, no_orphans,
PK excludes PK_known). This module is the SHARED scorer used both to reproduce
the champion (Step 1) and to score the literature-augmented reranker (Step 3).

Usage: import score_bp_naivemax(cat, rerank_bp_df) -> dict(f_micro_w, tau, cov).
rerank_bp_df: columns acc, go, s_rerank (BP rows only).
"""
import os, sys, json, tempfile, shutil
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SC = "/home/frapercan/Thesis2/repositories/protea-reranker-lab/results/sparse_classifier"
sys.path.insert(0, os.path.join(SC, "interpro2go_test"))
import interpro_lib as L  # noqa: E402
from cafaeval.evaluation import cafa_eval  # noqa: E402

OBO = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
IA_PATH = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv"
GT_DIR = os.path.join(SC, "lafa_gt")
TOI_FILE = os.path.join(GT_DIR, "groundtruth_terms_of_interest.txt")
NS_JSON = os.path.join(SC, "interpro2go_test", "go_namespace.json")

GT = {
    "lk": (os.path.join(GT_DIR, "groundtruth_LK.tsv"), None),
    "pk": (os.path.join(GT_DIR, "groundtruth_PK.tsv"),
           os.path.join(GT_DIR, "groundtruth_PK_known.tsv")),
}
# champion reranker TEST predictions per category (s_rerank source)
RERANK_TEST = {
    "lk": os.path.join(SC, "percut_rerank/baseline/predictions/lk/lk.tsv"),
    "pk": os.path.join(SC, "percut_rerank/predictions/pk/pk.tsv"),
}
nm = json.load(open(NS_JSON))
_tip = None


def interpro_test_map():
    global _tip
    if _tip is None:
        _tip = L.interpro_preds(json.load(
            open(os.path.join(SC, "interpro2go_test", "protein2ipr.json"))))
    return _tip


def union_naivemax(rerank_bp):
    """rerank_bp: acc,go,s_rerank (BP). Returns union frame with s_max."""
    ip_map = interpro_test_map()
    rr = rerank_bp[["acc", "go", "s_rerank"]].copy()
    rr = rr.groupby(["acc", "go"], as_index=False)["s_rerank"].max()
    pset = set(rr.acc.unique())
    ip_rows = []
    for a, gd in ip_map.items():
        if a not in pset:
            continue
        for g, s in gd.items():
            if nm.get(g) == "bpo":
                ip_rows.append((a, g, float(s)))
    ip_df = pd.DataFrame(ip_rows, columns=["acc", "go", "s_interpro"]) if ip_rows \
        else pd.DataFrame(columns=["acc", "go", "s_interpro"])
    u = pd.merge(rr, ip_df, on=["acc", "go"], how="outer")
    u["s_rerank"] = u["s_rerank"].fillna(0.0).astype("float32")
    u["s_interpro"] = u["s_interpro"].fillna(0.0).astype("float32")
    u = u[u.acc.isin(pset)].copy()
    u["s_max"] = np.maximum(u["s_rerank"], u["s_interpro"]).astype("float32")
    return u


def score_bp(pred_rows, cat):
    """pred_rows: iterable of (acc,go,score). Returns BP cell dict board-faithful."""
    gt_file, known = GT[cat]
    d = tempfile.mkdtemp(prefix=f"infame_{cat}_")
    pd.DataFrame(list(pred_rows), columns=["acc", "go", "score"]).to_csv(
        os.path.join(d, f"{cat}.tsv"), sep="\t", header=False, index=False,
        float_format="%.6f")
    _, dfs_best = cafa_eval(OBO, d, gt_file, ia=IA_PATH, no_orphans=True,
                            norm="cafa", prop="fill", exclude=known,
                            toi_file=TOI_FILE, th_step=0.01, n_cpu=8)
    best = dfs_best["f_micro_w"].reset_index()
    shutil.rmtree(d, ignore_errors=True)
    for _, r in best.iterrows():
        if r["ns"] == "biological_process":
            return {"f_micro_w": round(float(r["f_micro_w"]), 5),
                    "tau": round(float(r["tau"]), 3),
                    "cov": round(float(r.get("cov", float("nan"))), 4)}
    return {"f_micro_w": 0.0, "tau": 0.0, "cov": 0.0}


def load_rerank_test(cat, path=None):
    rr = pd.read_csv(path or RERANK_TEST[cat], sep="\t", header=None,
                     names=["acc", "go", "s_rerank"])
    rr = rr[rr.go.map(nm) == "bpo"][["acc", "go", "s_rerank"]]
    return rr


def champion_bp(cat, rerank_path=None):
    rr = load_rerank_test(cat, rerank_path)
    u = union_naivemax(rr)
    return score_bp(list(zip(u.acc, u.go, u.s_max)), cat)


if __name__ == "__main__":
    out = {}
    for cat in ["lk", "pk"]:
        out[cat] = champion_bp(cat)
        print(cat, out[cat], flush=True)
    json.dump(out, open(os.path.join(HERE, "champion_reproduction.json"), "w"),
              indent=2)
