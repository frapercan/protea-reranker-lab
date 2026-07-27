"""Board-faithful complementarity: does the TRAINED retriever, stacked WITH the
incumbent, lift LK-BPO / PK-BPO f_micro_w? Same cafa_eval call as the graft harness
(IA, TOI, prop=fill, norm=cafa, no_orphans, PK excludes PK_known); w=0 reproduces the
graft. We test the CALIBRATED OOF retriever probability as evidence:
  A) noisy-OR blend over ALL hard-covered pairs, weight sweep
  B) noisy-OR restricted to the reranker-FAILS pocket (clf-only OR bottom-half) with
     text coverage -- the honest "help where the incumbent fails" test
  C) top-k targeted boost (the goretriever_lite variant that gave LK +0.003)
Reports bpo f_micro_w and delta vs graft, for lk and pk.
"""
import collections
import json
import os
import shutil
import tempfile

import pandas as pd
from cafaeval.evaluation import cafa_eval

HERE = os.path.dirname(os.path.abspath(__file__))
SC = os.path.dirname(HERE)
OBO = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
IA = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv"
GT_DIR = os.path.join(SC, "lafa_gt")
TOI = os.path.join(GT_DIR, "groundtruth_terms_of_interest.txt")
BASE = os.path.join(SC, "interpro2go_test/predictions_graft_interpro_naivemax.tsv")
HARD = os.path.join(SC, "phaseA_separability/hard_frame.parquet")
ROUTE = os.path.join(SC, "goretriever_lite/bp_route.json")
NS2ASP = {"biological_process": "bpo"}
GT = {"lk": (os.path.join(GT_DIR, "groundtruth_LK.tsv"), None),
      "pk": (os.path.join(GT_DIR, "groundtruth_PK.tsv"),
             os.path.join(GT_DIR, "groundtruth_PK_known.tsv"))}


def load_base():
    df = pd.read_csv(BASE, sep="\t", header=None, names=["acc", "go", "score"])
    bmap = collections.defaultdict(dict)
    for r in df.itertuples(index=False):
        d = bmap[r.acc]
        d[r.go] = max(d.get(r.go, 0.0), float(r.score))
    return bmap


def score_bpo(rows, cat):
    gt_file, known = GT[cat]
    d = tempfile.mkdtemp(prefix=f"gate_{cat}_")
    pd.DataFrame(rows, columns=["acc", "go", "score"]).to_csv(
        os.path.join(d, f"{cat}.tsv"), sep="\t", header=False, index=False,
        float_format="%.6f")
    try:
        _, best = cafa_eval(OBO, d, gt_file, ia=IA, no_orphans=True, norm="cafa",
                            prop="fill", exclude=known, toi_file=TOI, th_step=0.01,
                            n_cpu=8)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    b = best["f_micro_w"].reset_index()
    for _, r in b.iterrows():
        if NS2ASP.get(r["ns"]) == "bpo":
            return round(float(r["f_micro_w"]), 5)
    return None


def build_rows(bmap, text, proteins, w, allow=None):
    """noisy-OR blend base with text prob; allow = set of (acc,go) eligible for blend."""
    rows = []
    for p in proteins:
        bm = bmap.get(p, {})
        tm = text.get(p, {})
        for t in set(bm) | set(tm):
            b = bm.get(t, 0.0)
            ts = tm.get(t)
            elig = (allow is None) or ((p, t) in allow)
            if ts is not None and w > 0 and elig:
                comb = 1.0 - (1.0 - b) * (1.0 - w * ts)
            else:
                comb = b
            if comb > 0:
                rows.append((p, t, comb))
    return rows


def build_topk(bmap, text, proteins, k, w, allow=None):
    rows = []
    for p in proteins:
        bm = dict(bmap.get(p, {}))
        tm = text.get(p, {})
        items = [(t, s) for t, s in tm.items()
                 if (allow is None) or ((p, t) in allow)]
        for t, s in sorted(items, key=lambda x: -x[1])[:k]:
            b = bm.get(t, 0.0)
            bm[t] = 1.0 - (1.0 - b) * (1.0 - w * s)
        for t, sc in bm.items():
            if sc > 0:
                rows.append((p, t, sc))
    return rows


def main():
    bmap = load_base()
    route = json.load(open(ROUTE))
    ts = pd.read_parquet(os.path.join(HERE, "retriever_oof_scores.parquet"))
    text = {c: collections.defaultdict(dict) for c in ("lk", "pk")}
    for r in ts.itertuples(index=False):
        text[r.cat][r.acc][r.go] = float(r.text_score)

    hard = pd.read_parquet(HARD)
    hard = hard[(hard.aspect == "bpo")]
    fails = {c: set() for c in ("lk", "pk")}
    for r in hard.itertuples(index=False):
        if r.category in fails and (r.clfonly or r.bottomhalf):
            fails[r.category].add((r.protein_accession, r.go_term_id))

    prots = {c: [p for p, cc in route.items() if cc == c] for c in ("lk", "pk")}
    grid = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
    out = {"metric": "f_micro_w (board-faithful, bpo)", "cells": {}}

    for c in ("lk", "pk"):
        ps = prots[c]
        base = score_bpo(build_rows(bmap, text[c], ps, 0.0), c)
        res = {"graft_baseline": base, "blend_all": {}, "blend_pocket": {},
               "topk_pocket": {}}
        # A: noisy-OR all covered
        bestA = (0.0, base)
        for w in grid:
            v = score_bpo(build_rows(bmap, text[c], ps, w), c)
            res["blend_all"][f"{w}"] = v
            if v > bestA[1]:
                bestA = (w, v)
        # B: noisy-OR restricted to reranker-fails pocket
        bestB = (0.0, base)
        for w in grid:
            v = score_bpo(build_rows(bmap, text[c], ps, w, allow=fails[c]), c)
            res["blend_pocket"][f"{w}"] = v
            if v > bestB[1]:
                bestB = (w, v)
        # C: top-k targeted on pocket
        bestC = (None, base)
        for k in (1, 3, 5):
            for w in (0.3, 0.5, 1.0):
                v = score_bpo(build_topk(bmap, text[c], ps, k, w, allow=fails[c]), c)
                res["topk_pocket"][f"k{k}_w{w}"] = v
                if v > bestC[1]:
                    bestC = ((k, w), v)
        res["best_blend_all"] = {"w": bestA[0], "bpo": bestA[1],
                                 "delta": round(bestA[1] - base, 5)}
        res["best_blend_pocket"] = {"w": bestB[0], "bpo": bestB[1],
                                    "delta": round(bestB[1] - base, 5)}
        res["best_topk_pocket"] = {"kw": bestC[0], "bpo": bestC[1],
                                   "delta": round(bestC[1] - base, 5)}
        out["cells"][c] = res
        print(f"{c.upper()}-BP graft {base} | blendAll {bestA[1]} (+{bestA[1]-base:.5f}) "
              f"| blendPocket {bestB[1]} (+{bestB[1]-base:.5f}) "
              f"| topkPocket {bestC[1]} (+{bestC[1]-base:.5f})", flush=True)

    json.dump(out, open(os.path.join(HERE, "board_complementarity.json"), "w"), indent=2)
    print("wrote board_complementarity.json", flush=True)


if __name__ == "__main__":
    main()
