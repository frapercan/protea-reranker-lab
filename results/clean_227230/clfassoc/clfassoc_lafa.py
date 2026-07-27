"""LAFA-exact 9-cell scoring of the clf+assoc per-category reranker.

Builds ONE prediction file over the 7401 LAFA query set: globalmm-normalised
per-category reranker_score for queries present in our eval frame, KNN fallback
for queries not covered. Scores with the EXACT LAFA cafaeval CLI per category
(NK/LK/PK) -prop fill -norm cafa -no_orphans -toi (PK adds -known), extracting
per-namespace f_micro_w. This mirrors the published board config exactly, so the
ONLY change vs the current board is the dataset (clf+assoc populated) + the 5
new features in the reranker.
"""
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

import lafa_harness as H

S = Path("/tmp/claude-1000/-home-frapercan-Thesis2/afd2c43a-ede7-46dc-94dd-9745808d2194/scratchpad")
SCORES = S / "clfassoc_out" / "eval_scores.parquet"
WORK = S / "clfassoc_out" / "lafa_work"
WORK.mkdir(parents=True, exist_ok=True)
OUTDIR = Path("/home/frapercan/Thesis2/repositories/protea-reranker-lab/results/clean_227230/clfassoc")
OUTDIR.mkdir(parents=True, exist_ok=True)

CATS = ["nk", "lk", "pk"]
ASPECTS = ["mfo", "bpo", "cco"]

CURRENT = {  # current PROTEA-reranked on the board (f_micro_w)
    "nk-mfo": 0.602, "nk-bpo": 0.309, "nk-cco": 0.431,
    "lk-mfo": 0.519, "lk-bpo": 0.348, "lk-cco": 0.419,
    "pk-mfo": 0.235, "pk-bpo": 0.117, "pk-cco": 0.254,
}
LEADER = {  # external leaders to beat (f_micro_w), where known
    "nk-cco": ("FunBind", 0.473), "nk-mfo": ("goa", 0.591),
    "lk-cco": ("TransFew/board", 0.434), "lk-bpo": ("TransFew", 0.512),
    "lk-mfo": ("goa", 0.510), "pk-bpo": ("TransFew", 0.294),
}


def globalmm(x):
    mn, mx = x.min(), x.max()
    return (x - mn) / (mx - mn)


def main():
    t = pq.read_table(SCORES)
    prot = np.asarray(t.column("protein_accession").to_pylist())
    go = np.asarray(t.column("go_term_id").to_pylist())
    rr = t.column("reranker_score").to_numpy(zero_copy_only=False).astype(np.float64)

    queries = set(H.query_ids())
    score = globalmm(rr)
    keep = np.array([p in queries for p in prot])
    rows = list(zip(prot[keep], go[keep], score[keep]))
    pred_file = WORK / "pred.tsv"
    ncov, nfb = H.build_pred_file(rows, pred_file)
    print(f"covered_queries_rows={keep.sum()} covered_proteins={ncov} fb_rows={nfb}", flush=True)

    res = H.score_all(pred_file, WORK)
    cells = {}
    for cat in ["NK", "LK", "PK"]:
        for a in ASPECTS:
            cells[f"{cat.lower()}-{a}"] = res[cat].get(a)
    print("RAW per-cat results:", json.dumps(res, indent=1), flush=True)

    cell_names = [f"{c}-{a}" for c in CATS for a in ASPECTS]
    verdict = {}
    for cell in cell_names:
        s = cells.get(cell)
        cur = CURRENT.get(cell)
        lead = LEADER.get(cell)
        verdict[cell] = {
            "clfassoc_f_micro_w": s,
            "current_board": cur,
            "delta_vs_current": round(s - cur, 4) if (s is not None and cur is not None) else None,
            "improves_vs_current": (s > cur + 1e-9) if (s is not None and cur is not None) else None,
            "leader": lead[0] if lead else None,
            "leader_f_micro_w": lead[1] if lead else None,
            "beats_leader": (s > lead[1]) if (s is not None and lead) else None,
        }

    cur_vals = [CURRENT[c] for c in cell_names]
    new_vals = [cells[c] for c in cell_names if cells[c] is not None]
    out = {
        "config": "clf+assoc per-category reranker (69 feat), LAFA-exact, globalmm + KNN fallback",
        "coverage": {"covered_proteins": ncov, "fallback_rows": nfb},
        "cells": cells,
        "verdict": verdict,
        "mean_current": round(float(np.mean(cur_vals)), 4),
        "mean_clfassoc": round(float(np.mean(new_vals)), 4) if new_vals else None,
        "n_improve": sum(1 for c in cell_names if cells[c] is not None and cells[c] > CURRENT[c] + 1e-9),
        "leaders_beaten": [c for c in cell_names
                           if c in LEADER and cells[c] is not None and cells[c] > LEADER[c][1]],
    }
    (OUTDIR / "comparison_9cell.json").write_text(json.dumps(out, indent=1))
    (WORK / "comparison_9cell.json").write_text(json.dumps(out, indent=1))

    print("\n9-CELL f_micro_w:")
    print(f"{'cell':9s} {'clf+assoc':>10s} {'current':>8s} {'delta':>8s} {'leader':>16s} {'beats?':>7s}")
    for cell in cell_names:
        v = verdict[cell]
        s = v["clfassoc_f_micro_w"]
        ld = f"{v['leader']}:{v['leader_f_micro_w']}" if v["leader"] else "-"
        bl = "" if v["beats_leader"] is None else ("YES" if v["beats_leader"] else "no")
        print(f"{cell:9s} {s if s is not None else 'NA':>10} {v['current_board']:>8} "
              f"{v['delta_vs_current'] if v['delta_vs_current'] is not None else 'NA':>8} {ld:>16s} {bl:>7s}")
    print(f"\nMEAN clf+assoc={out['mean_clfassoc']} vs current={out['mean_current']} "
          f"| improved cells={out['n_improve']}/9 | leaders beaten={out['leaders_beaten']}")


if __name__ == "__main__":
    main()
