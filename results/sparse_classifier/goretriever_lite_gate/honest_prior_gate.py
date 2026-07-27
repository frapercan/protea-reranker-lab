"""HONEST t0/validation-estimated GO-frequency prior gate on LK-BPO / PK-BPO.

The literature gate (SUMMARY.txt) is RED: text adds ZERO orthogonal board lift; the
apparent +0.045 LK / +0.038 PK "lift" was a per-GO base rate that LEAKED the test
label marginal (fit OOF on the TEST hard candidates). The ONE honest follow-up it
named: does a per-GO frequency prior, estimated on HELD-OUT data (v225-v227
validation) and weight-selected on that SAME validation frame, still lift LK-BPO /
PK-BPO board-faithful f_micro_w on the v227-v230 TEST -- or does the leaky lift
vanish under honesty?

Protocol (nothing touches test labels until the final frozen score):
  1. Prior p(g) = per-GO positive rate on the v225-v227 VALIDATION BP pool
     (valid_base.parquet), smoothed toward the category global rate.
  2. Weight/variant selection on validation ONLY, via the SAME fixed-denominator
     IA-weighted micro-Fmax proxy the InterPro graft's own weight was tuned on
     (no validation GT file exists; this is the sanctioned precedent). To avoid
     in-sample optimism the prior used INSIDE the proxy is estimated OOF
     (GroupKFold-by-protein) within validation.
  3. Freeze (variant, w). Refit the prior on the FULL validation frame. Apply to
     TEST candidates, score board-faithful (cafa_eval, OBO/IA/TOI, prop=fill,
     norm=cafa, no_orphans, PK excludes PK_known, f_micro_w) via score_gate.
  4. Report frozen-honest test delta vs the graft baseline, plus a test ORACLE
     (best-w-on-test) as a diagnostic upper bound only (NOT a claim).

Baselines (graft, from board_complementarity.json): LK-BPO 0.42807, PK-BPO 0.21631.
"""
import collections
import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

import score_gate as sg  # load_base, build_rows, score_bpo, GT

HERE = os.path.dirname(os.path.abspath(__file__))
SC = os.path.dirname(HERE)
VALID_BASE = os.path.join(SC, "interpro2go_test/valid_base.parquet")
IA_TSV = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv"
ROUTE = os.path.join(SC, "goretriever_lite/bp_route.json")
GRAFT = {"lk": 0.42807, "pk": 0.21631}  # graft BP baseline (w=0), board-faithful
GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
ALPHA = 20.0  # Dirichlet smoothing count toward the category global rate


def load_ia():
    ia = {}
    with open(IA_TSV) as fh:
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2:
                try:
                    ia[p[0]] = float(p[1])
                except ValueError:
                    continue
    return ia


IA = load_ia()


def smoothed_prior(go, y, glob):
    """Per-GO smoothed positive rate from arrays go[], y[] with global fallback."""
    df = pd.DataFrame({"go": go, "y": y})
    g = df.groupby("go")["y"].agg(["sum", "count"])
    rate = (g["sum"] + ALPHA * glob) / (g["count"] + ALPHA)
    return rate.to_dict()


def norm01(d):
    if not d:
        return d
    vals = np.array(list(d.values()))
    lo, hi = vals.min(), vals.max()
    return {k: float((v - lo) / (hi - lo + 1e-9)) for k, v in d.items()}


def valid_proxy(base, score, y, ia_w):
    """Fixed-denominator IA-weighted micro-Fmax over the validation pool (no DAG
    propagation) -- the graft's own validation selection metric. Fixed denominator:
    recall denominator = total IA of all positives, threshold-independent."""
    order = np.argsort(-score)
    s, yy, w = score[order], y[order], ia_w[order]
    tot_pos_ia = float((w * yy).sum())
    if tot_pos_ia <= 0:
        return 0.0
    best = 0.0
    for th in np.arange(0.0, 1.0001, 0.01):
        pred = s >= th
        if not pred.any():
            continue
        tp = float((w * yy * pred).sum())
        fp = float((w * (1 - yy) * pred).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / tot_pos_ia
        f = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        if f > best:
            best = f
    return round(best, 5)


def blend_valid(base_score, prior_norm, go, w):
    """noisy-OR blend of the validation base with w*prior for a given w."""
    pr = np.array([prior_norm.get(g, 0.0) for g in go])
    return 1.0 - (1.0 - base_score) * (1.0 - w * pr)


def main():
    vb = pd.read_parquet(VALID_BASE)
    vb = vb[vb.aspect == "bpo"].copy()
    ia_all = {g: IA.get(g, 0.0) for g in vb.go_term_id.unique()}

    route = json.load(open(ROUTE))
    bmap = sg.load_base()  # TEST graft (predictions_graft_interpro_naivemax.tsv)
    hard = pd.read_parquet(os.path.join(SC, "phaseA_separability/hard_frame.parquet"))
    hard = hard[hard.aspect == "bpo"]
    hardset = {c: set() for c in ("lk", "pk")}
    for r in hard.itertuples(index=False):
        if r.category in hardset and (r.clfonly or r.bottomhalf):
            hardset[r.category].add((r.protein_accession, r.go_term_id))

    out = {"note": "honest validation-estimated per-GO prior; weight+variant selected "
                   "on v225-v227 validation proxy; frozen to v227-v230 test",
           "alpha": ALPHA, "graft_baseline": GRAFT, "cells": {}}

    for c in ("lk", "pk"):
        d = vb[vb.category == c].reset_index(drop=True)
        y = d.label.to_numpy().astype(int)
        go = d.go_term_id.to_numpy()
        base = d.base_score.to_numpy().astype(float)
        groups = d.protein_accession.to_numpy()
        ia_w = np.array([ia_all.get(g, 0.0) for g in go])
        glob = float(y.mean())

        # --- OOF prior within validation (for honest proxy weight selection) ---
        oof = np.zeros(len(d))
        gkf = GroupKFold(n_splits=min(5, len(np.unique(groups))))
        for tr, te in gkf.split(d, y, groups):
            pr = smoothed_prior(go[tr], y[tr], float(y[tr].mean()))
            oof[te] = [pr.get(g, glob) for g in go[te]]
        oof_norm = norm01({i: v for i, v in enumerate(oof)})
        oof_arr = np.array([oof_norm[i] for i in range(len(d))])

        # candidate populations: ALL cell candidates vs HARD-pocket only
        pocket = np.array([(p, g) in hardset[c]
                           for p, g in zip(d.protein_accession, go)])

        sel = {}
        for variant, mask in (("all", np.ones(len(d), bool)), ("hard", pocket)):
            base_proxy = valid_proxy(base, base, y, ia_w)
            best = (0.0, base_proxy)
            row = {"w0_proxy": base_proxy}
            for w in GRID:
                pr_use = np.where(mask, oof_arr, 0.0)
                blended = 1.0 - (1.0 - base) * (1.0 - w * pr_use)
                v = valid_proxy(base, blended, y, ia_w)
                row[f"w{w}"] = v
                if v > best[1]:
                    best = (w, v)
            row["best_w"] = best[0]
            row["best_proxy"] = best[1]
            sel[variant] = row

        # pick variant+w on validation proxy
        chosen_variant = max(sel, key=lambda v: sel[v]["best_proxy"])
        chosen_w = sel[chosen_variant]["best_w"]

        # --- refit prior on FULL validation, apply to TEST, board-faithful ---
        full = smoothed_prior(go, y, glob)
        full_norm = norm01(full)
        ps = [p for p, cc in route.items() if cc == c]

        def test_score(w, variant):
            text = collections.defaultdict(dict)
            for p in ps:
                for t in bmap.get(p, {}):
                    if variant == "hard" and (p, t) not in hardset[c]:
                        continue
                    text[p][t] = full_norm.get(t, 0.0)
            return sg.score_bpo(sg.build_rows(bmap, text, ps, w), c)

        graft = sg.score_bpo(sg.build_rows(bmap, {}, ps, 0.0), c)
        frozen = test_score(chosen_w, chosen_variant) if chosen_w > 0 else graft

        # diagnostic ONLY: test oracle over the chosen variant (upper bound)
        oracle_best = (0.0, graft)
        oracle_curve = {}
        for w in GRID:
            v = test_score(w, chosen_variant)
            oracle_curve[f"w{w}"] = v
            if v > oracle_best[1]:
                oracle_best = (w, v)

        out["cells"][c] = {
            "validation_selection": sel,
            "chosen_variant": chosen_variant,
            "chosen_w": chosen_w,
            "graft_test": graft,
            "frozen_honest_test": frozen,
            "frozen_delta": round(frozen - graft, 5),
            "test_oracle_curve": oracle_curve,
            "test_oracle_best": {"w": oracle_best[0], "bpo": oracle_best[1],
                                 "delta": round(oracle_best[1] - graft, 5)},
        }
        print(f"{c.upper()}-BP  graft {graft}  | FROZEN-HONEST {frozen} "
              f"({frozen-graft:+.5f}) [variant={chosen_variant} w={chosen_w}]  "
              f"| test-oracle {oracle_best[1]} ({oracle_best[1]-graft:+.5f})", flush=True)

    json.dump(out, open(os.path.join(HERE, "honest_prior_gate.json"), "w"), indent=2)
    print("wrote honest_prior_gate.json", flush=True)


if __name__ == "__main__":
    main()
