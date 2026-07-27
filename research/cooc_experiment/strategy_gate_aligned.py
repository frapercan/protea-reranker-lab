"""The per-cell strategy gate: does any cell want a different scorer than the deployed reranker?

Lab #113 recommended "rerank NK/LK, raw-KNN PK" and its TEST-frame export never happened, so the
recommendation has sat unmeasured on the frame the board actually uses. This measures it.

THE ARMS. Each is a complete scorer, submitting its own rows with its own score:

  reranker    the deployed booster's margin          (`eval_scores.parquet: reranker_score`)
  knn_vote    the raw neighbour vote fraction        (`eval.parquet: neighbor_vote_fraction`)
  classifier  the sparse two-tower logit alone       (`eval.parquet: classifier_score`)

**Each arm submits only where its own score is strictly positive.** That is not a tweak, it is
forced: `prop=fill` treats 0.0 as the sentinel for "not submitted" and the board reads us through
the vectorised parser, which has no positivity guard, so a submitted non-positive score is stored
and blocks the free-ancestor inheritance. Measured cost of getting this wrong: 0.0933. Every arm
carries the guard, so the arms differ in the SCORER and not in the guard.

THE DESIGN, the same discipline PKW.8 earned: 10 disjoint protein folds. For each fold, pick the
best arm on the other NINE pooled and apply that choice BLIND to the tenth. The selection never
sees the fold it is scored on. Scored per cell against that cell's own board ground truth, on the
board's real line (`-toi`, `prop=fill`, `norm=cafa`, `no_orphans`, **max_terms=None,
th_step=0.01**).

THIS IS A WITHIN-FRAME DELTA, which is why it is legitimate at all. We cannot reproduce the
board's absolutes: the host's t0 ontology and IA live on their cluster and our `lafa_t0_Sep_2025`
directory holds a 2025-07-22 ontology. Every arm here shares one obo and one IA, so the
reweighting is common to all of them and cancels in the comparison. **No number this script
prints is a board number.**

THE GATE, written before the number, with the quantity named (discipline #10):
  - the quantity is the held-out f_micro_w of the SELECTED arm minus that of `reranker`, on the
    same fold, averaged over the ten folds.
  - **SELECTION STABILITY:** if the ten folds do not agree on one arm, the choice is fitting noise
    and there is nothing to ship. Report the vote either way.
  - **SIGN:** the mean held-out delta must be positive in at least 8 of 10 folds AND exceed one
    fold-to-fold sd. A gain inside the sd is not a gain, and I will say so.

THE PRECONDITION, which is the thing that saved the last run: `reranker` at fold-level must
reproduce the whole-cell figure from `build_submission.py` (PK-BP 0.21269, LK-BP 0.30376,
NK-BP 0.3021) to within fold noise. If the anchor does not come back, the run is void and nothing
below it is reportable, however much I like it.

A NOTE ON WHAT THE ARMS ARE MADE OF, because a zero here is not a number (discipline #5):
`vote_count`, `neighbor_vote_fraction` and the KNN-path features are ZERO on 100% of the
classifier-only rows, where the producers never ran, and `classifier_score` is absent on the
knn-only rows. So `knn_vote` and `classifier` cannot see the whole pool by construction: each is
blind to the half the other proposed. That is a property of the arms, not a bug in this script,
and it is exactly why "raw-KNN on PK" is a real strategy rather than a degenerate one. It is
reported alongside the result.
"""
import json, subprocess, tempfile, time, collections, statistics
from pathlib import Path
import numpy as np, pyarrow.parquet as pq

W = Path("/home/frapercan/Thesis2/storage/cooc_experiment")
DS = Path("/home/frapercan/Thesis2/repositories/protea-reranker-lab/datasets/protst-global-train227-test230")
REL = Path("/home/frapercan/Thesis2/CAFA_forever/data/releases/Sep_2025_Mar_2026")
OBO = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"   # 2025-07-22, NOT the board's
IA = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv"
TOI = str(REL / "groundtruth_terms_of_interest.txt")
PY = "/home/frapercan/Thesis2/repositories/PROTEA/.venv/bin/python"
SEED, K = 42, 10
ANCHOR = {"pk": 0.21269, "lk": 0.30376, "nk": 0.3021}  # build_submission.py, guard_only, whole-cell (ALIGNED)    # build_submission.py, guard_only, BP

# --- the scored table (reranker) -------------------------------------------------
t = pq.read_table(W / "rerank_out" / "eval_scores.parquet")
# ALIGNED join, NOT a dict: the pool holds 616,223 pk-bpo rows over 576,031 distinct
# (protein,term) keys. 40,192 are duplicated (once from the KNN generator, once from the
# classifier) and their two scores differ by a MEDIAN of 3.60, straddling zero 9,510 times.
# A dict keyed on (protein,term) is last-wins and silently rewrites 8.3% of the rows.
# `resolve_contradiction.py` proved it is the ONLY thing that moved the prefilter result
# (+0.00924 -> -0.00032), while the flags and the row set moved nothing.
RRcol = t.column("reranker_score").to_numpy(zero_copy_only=False).astype(np.float64)
Ps = np.asarray(t.column("protein_accession").to_pylist())
Gs = np.asarray(t.column("go_term_id").to_pylist())

# --- the feature table (knn_vote, classifier) ------------------------------------
f = pq.read_table(DS / "eval.parquet", columns=["protein_accession", "go_term_id", "category",
                                                "aspect", "neighbor_vote_fraction",
                                                "classifier_score"])
P = np.asarray(f.column("protein_accession").to_pylist())
G = np.asarray(f.column("go_term_id").to_pylist())
C = np.asarray(f.column("category").to_pylist())
A = np.asarray(f.column("aspect").to_pylist())
KV = f.column("neighbor_vote_fraction").to_numpy(zero_copy_only=False).astype(np.float64)
CS = f.column("classifier_score").to_numpy(zero_copy_only=False).astype(np.float64)
# both tables are the same export in the same order; assert it rather than trusting it
assert len(Ps) == len(P) and (Ps == P).all() and (Gs == G).all(), (
    "eval_scores and eval.parquet are not row-aligned; an aligned join is invalid here")
RRv = RRcol
print(f"feature rows {len(P):,} | ALIGNED join verified row-for-row (no dict, no last-wins)", flush=True)

ARMS = {"reranker": RRv, "knn_vote": KV, "classifier": CS}
for name, v in ARMS.items():
    for cell in ("nk", "lk", "pk"):
        m = (C == cell) & (A == "bpo")
        x = v[m]
        print(f"  {name:11s} {cell}-bpo: nan={np.isnan(x).mean():5.1%} positive={np.nansum(x > 0) / max(len(x),1):5.1%}", flush=True)

DRIVER = '''
import json, signal
from cafaeval.evaluation import cafa_eval
signal.signal(signal.SIGTERM, signal.SIG_DFL)
df,dfs=cafa_eval("{obo}","{pd}","{gt}",ia="{ia}",prop="fill",norm="cafa",no_orphans=True,
    toi_file="{toi}",max_terms=None,th_step=0.01,n_cpu=4,weighted_only=False)
out={{}}
for k,v in dfs.items(): out[k]=v.reset_index().to_dict(orient="records")
json.dump(out,open("{o}","w"),default=str)
'''
GT = {}
for cell, fn in (("nk", "groundtruth_NK.tsv"), ("lk", "groundtruth_LK.tsv"), ("pk", "groundtruth_PK.tsv")):
    d = collections.defaultdict(set)
    with (REL / fn).open() as fh:
        next(fh)
        for line in fh:
            p_, t_, a_ = line.rstrip("\n").split("\t")[:3]
            if a_ == "P":
                d[p_].add(t_)
    GT[cell] = d
    print(f"  gt {cell}-BP: {len(d):,} proteins", flush=True)


def score(cell, arm, prots):
    v = ARMS[arm]
    keep = (C == cell) & np.isin(P, list(prots)) & np.isfinite(v) & (v > 0)
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "pd"
        d.mkdir()
        with (d / "p.tsv").open("w") as fh:
            for p_, g_, x in zip(P[keep], G[keep], v[keep]):
                fh.write(f"{p_}\t{g_}\t{x:.6f}\n")
        gt = Path(td) / "gt.tsv"
        with gt.open("w") as fh:
            for p_ in prots:
                for g_ in GT[cell].get(p_, ()):
                    fh.write(f"{p_}\t{g_}\n")
        raw = Path(td) / "r.json"
        drv = Path(td) / "d.py"
        drv.write_text(DRIVER.format(obo=OBO, pd=str(d), gt=str(gt), ia=IA, toi=TOI, o=str(raw)))
        r = subprocess.run([PY, str(drv)], capture_output=True, text=True, timeout=7200)
        if r.returncode != 0:
            return None
        best = None
        for rec in json.loads(raw.read_text()).get("f_micro_w", []):
            if (rec.get("ns") or "") == "biological_process" and rec.get("f_micro_w") is not None:
                best = rec
        return round(float(best["f_micro_w"]), 5) if best else None


t0 = time.time()
res = {"frame": "LAB frame: obo data-version 2025-07-22, NOT the board's Sep-2025 t0. Every arm "
                "shares one obo and one IA, so the reweighting cancels in the comparison. "
                "NO NUMBER HERE IS A BOARD NUMBER.",
       "guard": "every arm submits only where its own score is > 0; prop=fill treats 0.0 as the "
                "sentinel for 'not submitted' and the board's vectorised parser has no positivity "
                "guard (cost of getting it wrong: 0.0933). The arms differ in the SCORER, not the guard.",
       "gate": "selection must be unanimous across the ten folds AND the mean held-out delta must "
               "be positive in >=8/10 and exceed one fold-to-fold sd.",
       "precondition": "fold-level `reranker` must reproduce build_submission.py's whole-cell "
                       f"anchors {ANCHOR} within fold noise, else the run is void.",
       "cells": {}}

for cell in ("pk", "lk", "nk"):
    prots = sorted(GT[cell])
    rng = np.random.default_rng(SEED)
    rng.shuffle(prots)
    FOLDS = [set(prots[i::K]) for i in range(K)]
    res["cells"][cell] = {"n_gt_proteins": len(prots), "folds": []}
    print(f"\n=== {cell}-BP: {len(prots):,} gt proteins, {K} folds ===", flush=True)
    for i in range(K):
        rest = set().union(*[FOLDS[j] for j in range(K) if j != i])
        sel = {a: score(cell, a, rest) for a in ARMS}
        sel = {k: v for k, v in sel.items() if v is not None}
        if not sel:
            continue
        chosen = max(sel, key=sel.get)
        held = {a: score(cell, a, FOLDS[i]) for a in ARMS}
        rec = {"fold": i, "sweep_on_the_other_nine": sel, "chosen_arm": chosen,
               "held_out": held,
               "held_out_delta_vs_reranker": (round(held[chosen] - held["reranker"], 5)
                                              if held.get(chosen) is not None and held.get("reranker") is not None else None)}
        res["cells"][cell]["folds"].append(rec)
        print(f"  fold {i}: chose {chosen:10s} -> held-out {held.get(chosen)} vs reranker "
              f"{held.get('reranker')} = {rec['held_out_delta_vs_reranker']}  ({time.time()-t0:.0f}s)", flush=True)
        json.dump(res, open(W / "strategy_gate_aligned.json", "w"), indent=1)

    folds = res["cells"][cell]["folds"]
    ch = collections.Counter(x["chosen_arm"] for x in folds)
    rk = [x["held_out"]["reranker"] for x in folds if x["held_out"].get("reranker") is not None]
    d = [x["held_out_delta_vs_reranker"] for x in folds if x["held_out_delta_vs_reranker"] is not None]
    anchor_ok = abs(statistics.mean(rk) - ANCHOR[cell]) < 0.02 if rk else False
    v = {"selections": dict(ch), "unanimous": len(ch) == 1,
         "mean_reranker_held_out": round(statistics.mean(rk), 5) if rk else None,
         "whole_cell_anchor": ANCHOR[cell], "PRECONDITION_anchor_reproduces": anchor_ok,
         "mean_delta": round(statistics.mean(d), 5) if d else None,
         "sd_across_folds": round(statistics.stdev(d), 5) if len(d) > 1 else None,
         "n_positive": sum(1 for x in d if x > 0), "n_folds": len(d)}
    v["PASSES_GATE"] = bool(anchor_ok and len(ch) == 1 and v["n_positive"] >= 8
                            and v["mean_delta"] is not None and v["sd_across_folds"] is not None
                            and v["mean_delta"] > v["sd_across_folds"])
    res["cells"][cell]["verdict"] = v
    json.dump(res, open(W / "strategy_gate_aligned.json", "w"), indent=1)
    print(f"  {cell}-BP verdict: selections {dict(ch)} | reranker held-out mean "
          f"{v['mean_reranker_held_out']} vs anchor {ANCHOR[cell]} (precondition: {anchor_ok})", flush=True)
    print(f"    mean delta {v['mean_delta']} sd {v['sd_across_folds']} positive {v['n_positive']}/{v['n_folds']}"
          f" -> GATE: {'PASS' if v['PASSES_GATE'] else 'FAIL'}", flush=True)

print("\nDONE", flush=True)
