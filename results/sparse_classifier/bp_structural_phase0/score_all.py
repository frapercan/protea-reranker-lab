"""Board-faithful BP scoring for the union-pool A/B, mirroring the champion graft.

For each (cat, variant) BP cell:
  cell = naivemax( reranker_BP_test  UNION  InterPro2GO_BP_test ),
  scored with cafa_eval (OBO, IA, TOI, prop=fill, norm=cafa, no_orphans, PK excludes
  PK_known). Reports best f_micro_w (+tau, pr, rc, cov), the realized-recall at the
  best-Fmax tau, and the propagated recall CEILING (max rc_micro_w over tau, tau->0).

Reuses the literature_infame graft harness (interpro map + union_naivemax + GT/OBO/IA).
"""
import os, sys, json, tempfile, shutil
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SC = "/home/frapercan/Thesis2/repositories/protea-reranker-lab/results/sparse_classifier"
sys.path.insert(0, os.path.join(SC, "literature_infame"))
import graft_score as G  # noqa: E402
from cafaeval.evaluation import cafa_eval  # noqa: E402

TRANSFEW = {"lk": 0.512, "pk": 0.294}
CHAMPION_CANON = {"lk": 0.42807, "pk": 0.21797}


def score_bp_full(pred_rows, cat):
    """pred_rows: iterable (acc,go,score). Returns full board-faithful BP metrics."""
    gt_file, known = G.GT[cat]
    d = tempfile.mkdtemp(prefix=f"bpstruct_{cat}_")
    pd.DataFrame(list(pred_rows), columns=["acc", "go", "score"]).to_csv(
        os.path.join(d, f"{cat}.tsv"), sep="\t", header=False, index=False,
        float_format="%.6f")
    df, dfs_best = cafa_eval(
        G.OBO, d, gt_file, ia=G.IA_PATH, no_orphans=True, norm="cafa", prop="fill",
        exclude=known, toi_file=G.TOI_FILE, th_step=0.01, n_cpu=8)
    shutil.rmtree(d, ignore_errors=True)
    full = df.reset_index()
    bp = full[full["ns"] == "biological_process"].copy()
    best = dfs_best["f_micro_w"].reset_index()
    bbp = best[best["ns"] == "biological_process"].iloc[0]
    btau = float(bbp["tau"])
    rc_ceiling = float(bp["rc_micro_w"].max())          # tau -> 0 propagated recall
    rc_realized = float(bp.loc[(bp["tau"] - btau).abs().idxmin(), "rc_micro_w"])
    return {"f_micro_w": round(float(bbp["f_micro_w"]), 5), "tau": round(btau, 3),
            "pr_micro_w": round(float(bbp.get("pr_micro_w", float("nan"))), 5),
            "rc_micro_w": round(float(bbp.get("rc_micro_w", float("nan"))), 5),
            "cov": round(float(bbp.get("cov", float("nan"))), 4),
            "rc_realized_at_best": round(rc_realized, 5),
            "rc_ceiling_tau0": round(rc_ceiling, 5)}


def grafted(cat, variant):
    rr = pd.read_parquet(os.path.join(HERE, f"rerank_bp_{cat}_{variant}.parquet"))
    u = G.union_naivemax(rr)
    return score_bp_full(list(zip(u.acc, u.go, u.s_max)), cat)


def main():
    out = {"transfew_bp": TRANSFEW, "champion_canonical": CHAMPION_CANON, "cells": {}}
    for cat in ["lk", "pk"]:
        champ = G.champion_bp(cat)              # existing champion tsv grafted (anchor)
        base = grafted(cat, "base")
        union = grafted(cat, "union")
        out["cells"][cat] = {
            "champion_reproduced": champ,
            "retrained_base": base,
            "union_pool": union,
            "transfew": TRANSFEW[cat],
            "delta_union_minus_base": round(union["f_micro_w"] - base["f_micro_w"], 5),
            "delta_base_minus_champion": round(base["f_micro_w"] - champ["f_micro_w"], 5),
            "gap_union_to_transfew": round(union["f_micro_w"] - TRANSFEW[cat], 5),
        }
        print(cat, "champ", champ["f_micro_w"], "base", base["f_micro_w"],
              "union", union["f_micro_w"],
              "| rc_ceiling base->union",
              base["rc_ceiling_tau0"], "->", union["rc_ceiling_tau0"], flush=True)
    json.dump(out, open(os.path.join(HERE, "score_variants.json"), "w"), indent=2)
    print("written score_variants.json", flush=True)


if __name__ == "__main__":
    main()
