"""Stage 2: board-faithful scoring of the text arm and noisy-OR blend vs the
current graft naivemax baseline. NK-BP and LK-BP (PK text not fetched).

Reproduces score_cafaeval.py exact call; w=0 must reproduce graft bpo cells.
"""
import os
import json
import tempfile
import shutil
import collections
import numpy as np
import pandas as pd
from cafaeval.evaluation import cafa_eval

HERE = os.path.dirname(os.path.abspath(__file__))
SC = "/home/frapercan/Thesis2/repositories/protea-reranker-lab/results/sparse_classifier"
OBO = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
IA = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv"
GT_DIR = os.path.join(SC, "lafa_gt")
TOI = os.path.join(GT_DIR, "groundtruth_terms_of_interest.txt")
BASE = os.path.join(SC, "interpro2go_test/predictions_graft_interpro_naivemax.tsv")
NS2ASP = {"biological_process": "bpo", "molecular_function": "mfo",
          "cellular_component": "cco"}
GT = {"nk": (os.path.join(GT_DIR, "groundtruth_NK.tsv"), None),
      "lk": (os.path.join(GT_DIR, "groundtruth_LK.tsv"), None),
      "pk": (os.path.join(GT_DIR, "groundtruth_PK.tsv"),
             os.path.join(GT_DIR, "groundtruth_PK_known.tsv"))}
TEXT_MODE = os.environ.get("TEXT_MODE", "precut")
TAG = os.environ.get("TAG", "spubmed")


def load_base():
    df = pd.read_csv(BASE, sep="\t", header=None, names=["acc", "go", "score"])
    bmap = collections.defaultdict(dict)
    for r in df.itertuples(index=False):
        d = bmap[r.acc]
        d[r.go] = max(d.get(r.go, 0.0), float(r.score))
    return bmap


def score_cat(rows, cat):
    gt_file, known = GT[cat]
    d = tempfile.mkdtemp(prefix=f"gr_{cat}_")
    pd.DataFrame(rows, columns=["acc", "go", "score"]).to_csv(
        os.path.join(d, f"{cat}.tsv"), sep="\t", header=False, index=False,
        float_format="%.6f")
    try:
        _, dfs_best = cafa_eval(OBO, d, gt_file, ia=IA, no_orphans=True,
                                norm="cafa", prop="fill", exclude=known,
                                toi_file=TOI, th_step=0.01, n_cpu=8)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    best = dfs_best["f_micro_w"].reset_index()
    cells = {}
    for _, r in best.iterrows():
        a = NS2ASP.get(r["ns"])
        if a:
            cells[a] = {"f_micro_w": round(float(r["f_micro_w"]), 5),
                        "tau": round(float(r["tau"]), 3),
                        "cov": round(float(r.get("cov", float("nan"))), 4)}
    return cells


def build_rows(bmap, text_by_prot, proteins, w):
    """For each protein: union base terms + text(BP) terms; blend BP via noisyor."""
    rows = []
    for p in proteins:
        bm = bmap.get(p, {})
        tm = text_by_prot.get(p, {})
        terms = set(bm) | set(tm)
        for t in terms:
            b = bm.get(t, 0.0)
            ts = tm.get(t)
            if ts is not None and w > 0:
                comb = 1.0 - (1.0 - b) * (1.0 - w * ts)
            else:
                comb = b
            if comb > 0:
                rows.append((p, t, comb))
    return rows


def main():
    bmap = load_base()
    route = json.load(open(os.path.join(HERE, "bp_route.json")))
    ts = pd.read_parquet(os.path.join(HERE, f"text_scores_bpo_{TAG}_{TEXT_MODE}.parquet"))
    text_by_prot = collections.defaultdict(dict)
    for r in ts.itertuples(index=False):
        text_by_prot[r.acc][r.go] = float(r.text_score)
    prots = {cat: [p for p, c in route.items() if c == cat]
             for cat in ("nk", "lk", "pk")}

    grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    out = {"text_mode": TEXT_MODE, "metric": "f_micro_w (board-faithful)",
           "baseline_naivemax_bpo": {}, "text_alone_bpo": {}, "sweep": {},
           "chosen": {}}

    for cat in ("nk", "lk"):
        ps = prots[cat]
        # baseline (w=0 reproduction)
        base_cells = score_cat(build_rows(bmap, text_by_prot, ps, 0.0), cat)
        out["baseline_naivemax_bpo"][cat] = base_cells.get("bpo")
        # text alone: predictions = text scores only (BP), board-faithful
        talone = [(p, g, text_by_prot[p][g]) for p in ps
                  for g in text_by_prot.get(p, {})]
        out["text_alone_bpo"][cat] = score_cat(talone, cat).get("bpo")
        # sweep
        sweep = {}
        base_bp = base_cells["bpo"]["f_micro_w"]
        best_w, best_v = 0.0, base_bp
        for w in grid:
            if w == 0.0:
                sweep["0.0"] = base_bp
                continue
            cells = score_cat(build_rows(bmap, text_by_prot, ps, w), cat)
            v = cells["bpo"]["f_micro_w"]
            sweep[f"{w:.1f}"] = v
            if v > best_v + 1e-9:
                best_v, best_w = v, w
        out["sweep"][cat] = sweep
        out["chosen"][cat] = {"w": best_w, "bpo": best_v,
                              "delta_vs_graft": round(best_v - base_bp, 5)}
        print(cat, "graft", round(base_bp, 5), "text_alone",
              out["text_alone_bpo"][cat]["f_micro_w"] if out["text_alone_bpo"][cat] else None,
              "best_w", best_w, "blend", round(best_v, 5),
              "delta", round(best_v - base_bp, 5))

    json.dump(out, open(os.path.join(HERE, f"blend_9cell_{TAG}_{TEXT_MODE}.json"), "w"),
              indent=2)
    print("written blend_9cell_%s_%s.json" % (TAG, TEXT_MODE))


if __name__ == "__main__":
    main()
