"""DECISIVE control: is the board-faithful lift (LK +0.045, PK +0.032) real ORTHOGONAL
literature signal, or just the per-GO base-rate (test-label marginal / term-frequency
prior) that the OOF logistic learned through the shared GO embedding?

We build an OOF per-GO base-rate score (NO text at all) in the same [0,1] format and
run the IDENTICAL noisy-OR / top-k board blend as score_gate.py. If the base-rate-only
blend reproduces the text-trained lift, the lift is term-frequency recalibration
(a pool artifact, not a deployable literature lever). If the text-trained blend lifts
MORE, the excess is orthogonal literature signal.
"""
import collections
import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

import score_gate as sg  # reuse load_base, score_bpo, build_rows, build_topk, GT, prots

HERE = os.path.dirname(os.path.abspath(__file__))
SC = os.path.dirname(HERE)
HARD = os.path.join(SC, "phaseA_separability/hard_frame.parquet")


def main():
    hard = pd.read_parquet(HARD)
    hard = hard[(hard.aspect == "bpo") & (hard.category.isin(["lk", "pk"]))].copy()
    # restrict to the SAME text-covered pairs used by the retriever (fair comparison)
    oof = pd.read_parquet(os.path.join(HERE, "retriever_oof_scores.parquet"))
    covered = oof.rename(columns={"acc": "protein_accession", "go": "go_term_id"})[
        ["protein_accession", "go_term_id", "cat"]]
    hard = hard.merge(covered, left_on=["protein_accession", "go_term_id"],
                      right_on=["protein_accession", "go_term_id"], how="inner")

    rows = []
    for c in ["lk", "pk"]:
        d = hard[hard.cat == c].reset_index(drop=True)
        y = d.label.to_numpy().astype(int)
        groups = d.protein_accession.to_numpy()
        go = d.go_term_id.to_numpy()
        base = np.zeros(len(d), dtype=float)
        gkf = GroupKFold(n_splits=min(5, len(np.unique(groups))))
        for tr, te in gkf.split(d, y, groups):
            gp = pd.Series(y[tr]).groupby(go[tr]).mean()
            prior = y[tr].mean()
            base[te] = [gp.get(g, prior) for g in go[te]]
        lo, hi = base.min(), base.max()
        d["text_score"] = (base - lo) / (hi - lo + 1e-9)
        d["cat"] = c
        rows.append(d[["protein_accession", "go_term_id", "cat", "text_score"]]
                    .rename(columns={"protein_accession": "acc", "go_term_id": "go"}))
    br = pd.concat(rows, ignore_index=True)

    bmap = sg.load_base()
    text = {c: collections.defaultdict(dict) for c in ("lk", "pk")}
    for r in br.itertuples(index=False):
        text[r.cat][r.acc][r.go] = float(r.text_score)
    fails = {c: set() for c in ("lk", "pk")}
    for r in hard.itertuples(index=False):
        if r.clfonly or r.bottomhalf:
            fails[r.cat].add((r.protein_accession, r.go_term_id))

    grid = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
    out = {"note": "per-GO base-rate-only blend (NO text); compare lift to text-trained",
           "cells": {}}
    for c in ("lk", "pk"):
        ps = sg.prots_for(c) if hasattr(sg, "prots_for") else None
        if ps is None:
            route = json.load(open(os.path.join(SC, "goretriever_lite/bp_route.json")))
            ps = [p for p, cc in route.items() if cc == c]
        base_v = sg.score_bpo(sg.build_rows(bmap, text[c], ps, 0.0), c)
        blend = {}
        bestb = (0.0, base_v)
        for w in grid:
            v = sg.score_bpo(sg.build_rows(bmap, text[c], ps, w), c)
            blend[f"{w}"] = v
            if v > bestb[1]:
                bestb = (w, v)
        topk = {}
        bestk = (None, base_v)
        for k in (1, 3, 5):
            for w in (0.3, 0.5, 1.0):
                v = sg.score_bpo(sg.build_topk(bmap, text[c], ps, k, w, allow=fails[c]), c)
                topk[f"k{k}_w{w}"] = v
                if v > bestk[1]:
                    bestk = ((k, w), v)
        out["cells"][c] = {
            "graft_baseline": base_v,
            "baserate_blend": blend,
            "baserate_topk_pocket": topk,
            "best_blend": {"w": bestb[0], "bpo": bestb[1], "delta": round(bestb[1] - base_v, 5)},
            "best_topk": {"kw": bestk[0], "bpo": bestk[1], "delta": round(bestk[1] - base_v, 5)},
        }
        print(f"{c.upper()}-BP BASE-RATE-ONLY: graft {base_v} | best_blend {bestb[1]} "
              f"(+{bestb[1]-base_v:.5f}) | best_topk {bestk[1]} (+{bestk[1]-base_v:.5f})",
              flush=True)
    json.dump(out, open(os.path.join(HERE, "baserate_board.json"), "w"), indent=2)
    print("wrote baserate_board.json", flush=True)


if __name__ == "__main__":
    main()
