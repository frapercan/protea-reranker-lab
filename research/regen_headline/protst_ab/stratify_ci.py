"""Fast protein-level paired bootstrap CI on the BP-wall + NK-anchor cell deltas
(arm B - arm A). Parses the ontology ONCE (cafaeval primitives) and evaluates
in-process. Bootstrap uses th_step=0.01 (coarser tau grid) for tractability on
the large pk-bpo cell; the point-estimate deltas reported elsewhere use 0.001.
Same everything else: prop=fill, norm=cafa, no_orphans, max_terms=500, IA/OBO v227.

Bootstrap: resample the cell's proteins WITH REPLACEMENT, rename each instance so
repeats are distinct proteins; arm A and arm B share the resample -> paired delta.
"""
import json
import tempfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from cafaeval import evaluation as E

OBO = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
IA = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv"
NSFULL = {"bpo": "biological_process"}
Bn = 120
RNG = np.random.default_rng(42)
CELLS = [("nk", "bpo"), ("lk", "bpo"), ("pk", "bpo")]

print("parsing ontology once...", flush=True)
ONT = E.obo_parser(OBO, ("is_a", "part_of"), IA, True)
TAU = np.arange(0.01, 1, 0.01)   # th_step=0.01

A = pq.read_table("out_A/eval_scores.parquet")
Bt = pq.read_table("out_B/eval_scores.parquet")
prots = np.asarray(A.column("protein_accession").to_pylist())
gos = np.asarray(A.column("go_term_id").to_pylist())
lab = A.column("label").to_numpy(zero_copy_only=False).astype(np.int64)
cat = np.asarray(A.column("category").to_pylist())
asp = np.asarray(A.column("aspect").to_pylist())
sA = A.column("reranker_score").to_numpy(zero_copy_only=False).astype(np.float64)
sB = Bt.column("reranker_score").to_numpy(zero_copy_only=False).astype(np.float64)

ev = pq.read_table("sealed_eval.parquet", columns=["protein_accession", "length_query"])
lp = np.asarray(ev.column("protein_accession").to_pylist())
lq = ev.column("length_query").to_numpy(zero_copy_only=False).astype(np.float64)
plen = {}
for p, L in zip(lp, lq):
    plen.setdefault(p, L)


def fmw(pred_rows, gt_rows, aspect, td):
    d = Path(td)
    pdir = d / "pred"
    pdir.mkdir(parents=True, exist_ok=True)
    with (pdir / "p.tsv").open("w") as fh:
        for p, g, s in pred_rows:
            fh.write(f"{p}\t{g}\t{s:.6f}\n")
    gtf = d / "gt.tsv"
    with gtf.open("w") as fh:
        for p, g in gt_rows:
            fh.write(f"{p}\t{g}\n")
    gt = E.gt_parser(str(gtf), ONT)
    pred = E.pred_parser(str(pdir / "p.tsv"), ONT, gt, "fill", 500, 1)
    if not pred:
        return None
    df = E.evaluate_prediction(pred, gt, ONT, TAU, None, normalization="cafa", n_cpu=1, weighted_only=True)
    dd = df.reset_index()
    if "cov" in dd.columns:
        dd = dd[dd["cov"] > 0]
    nsc = "ns" if "ns" in dd.columns else "namespace"
    sub = dd[dd[nsc] == NSFULL[aspect]]
    if sub.empty or sub["f_micro_w"].isna().all():
        return None
    return float(sub["f_micro_w"].max())


def cell_arrays(c, a):
    m = (cat == c) & (asp == a)
    return prots[m], gos[m], sA[m], sB[m], lab[m]


def boot(c, a):
    P, G, SA, SB, L = cell_arrays(c, a)
    uniq = np.unique(P)
    rows_by = {u: np.where(P == u)[0] for u in uniq}
    deltas = []
    for b in range(Bn):
        samp = RNG.choice(uniq, size=len(uniq), replace=True)
        predA, predB, gt = [], [], []
        for k, u in enumerate(samp):
            nm = f"{u}#{k}"
            for i in rows_by[u]:
                predA.append((nm, G[i], SA[i]))
                predB.append((nm, G[i], SB[i]))
                if L[i] > 0:
                    gt.append((nm, G[i]))
        with tempfile.TemporaryDirectory() as td:
            ba = fmw(predA, gt, a, Path(td) / "A")
            bb = fmw(predB, gt, a, Path(td) / "B")
        if ba is not None and bb is not None:
            deltas.append(bb - ba)
    deltas = np.array(deltas)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    r = {"cell": f"{c}-{a}", "n_proteins": int(len(uniq)), "B": int(len(deltas)),
         "boot_mean_delta": round(float(deltas.mean()), 4),
         "ci95": [round(float(lo), 4), round(float(hi), 4)],
         "ci_excludes_zero": bool(lo > 0 or hi < 0),
         "frac_boot_positive": round(float((deltas > 0).mean()), 3)}
    print("CI", json.dumps(r), flush=True)
    return r


def length_strat():
    out = {}
    m = (asp == "bpo")
    P, G, SA, SB, L = prots[m], gos[m], sA[m], sB[m], lab[m]
    Ln = np.array([plen.get(p, np.nan) for p in P])
    for name, bm in [("short_<=300", Ln <= 300), ("mid_301_600", (Ln > 300) & (Ln <= 600)),
                     ("long_>600", Ln > 600)]:
        if bm.sum() == 0:
            continue
        with tempfile.TemporaryDirectory() as td:
            idx = np.where(bm)[0]
            fa = fmw([(P[i], G[i], SA[i]) for i in idx], [(P[i], G[i]) for i in idx if L[i] > 0], "bpo", Path(td) / "A")
            fb = fmw([(P[i], G[i], SB[i]) for i in idx], [(P[i], G[i]) for i in idx if L[i] > 0], "bpo", Path(td) / "B")
        out[name] = {"n_rows": int(bm.sum()), "n_pos": int((L[bm] > 0).sum()),
                     "armA": round(fa, 4) if fa else None, "armB": round(fb, 4) if fb else None,
                     "delta": round(fb - fa, 4) if (fa and fb) else None}
        print("LEN", name, json.dumps(out[name]), flush=True)
    return out


def main():
    res = {"note": "bootstrap CI uses th_step=0.01 for tractability; point deltas use 0.001",
           "bootstrap_ci": {}, "length_strat_bpo_thstep001": {}}
    for c, a in CELLS:
        res["bootstrap_ci"][f"{c}-{a}"] = boot(c, a)
    res["length_strat_bpo_thstep001"] = length_strat_001()
    Path("stratify_ci_result.json").write_text(json.dumps(res, indent=1))
    print("DONE -> stratify_ci_result.json", flush=True)


def length_strat_001():
    """Length strat at the real th_step=0.001 for reportable absolutes."""
    global TAU
    saved = TAU
    TAU = np.arange(0.001, 1, 0.001)
    try:
        return length_strat()
    finally:
        TAU = saved


if __name__ == "__main__":
    main()
