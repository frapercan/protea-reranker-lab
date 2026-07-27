"""MODE B cafa_eval + exact-parity paired bootstrap CI.

For each cell (pk/lk x mfo/bpo/cco): cafa_eval the baseline and +sse (and +sse_shuffled) retrained
prediction files in the TRUE frame (prop=fill norm=cafa no_orphans toi, PK exclude=PK_known), report
f_micro_w and the delta(+sse - baseline). Paired protein bootstrap on the delta reusing cafaeval's own
parser/DAG-propagation + per-protein weighted TP/FP/FN at each arm's cafa-best tau. baseline is also
cross-checked against the DEPLOYED anchor. Run under PROTEA/.venv.
"""
import sys, json, tempfile, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.sparse import issparse
from cafaeval.parser import obo_parser, gt_parser, pred_parser, gt_exclude_parser, update_toi
from cafaeval.evaluation import cafa_eval

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)
ROOT = Path("/home/frapercan/Thesis2")
PC = ROOT / "repositories/protea-reranker-lab/results/sparse_classifier/percut_rerank"
GTDIR = ROOT / "repositories/protea-reranker-lab/results/sparse_classifier/lafa_gt"
OBO = str(ROOT / "protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo")
IA = str(ROOT / "protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv")
TOI = str(GTDIR / "groundtruth_terms_of_interest.txt")
SSE = ROOT / "storage/sse_full"; OUTB = SSE / "modeB"
GTF = {"lk": (str(GTDIR / "groundtruth_LK.tsv"), None),
       "pk": (str(GTDIR / "groundtruth_PK.tsv"), str(GTDIR / "groundtruth_PK_known.tsv"))}
NSMAP = {"mfo": "molecular_function", "bpo": "biological_process", "cco": "cellular_component"}
DEPLOYED_ANCHOR = {("pk", "mfo"): 0.2416, ("pk", "cco"): 0.2655, ("pk", "bpo"): 0.14351,
                   ("lk", "mfo"): 0.41998, ("lk", "cco"): 0.35724, ("lk", "bpo"): 0.31323}
VARIANTS = ["baseline", "sse", "sse_shuffled"]
CELLS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["pk", "lk"]

def dense(M): return np.asarray(M.todense()) if issparse(M) else np.asarray(M)

def cell_f(df, gt_file, known, nsname):
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "pd"; d.mkdir()
        df[["prot", "term", "score"]].to_csv(d / "p.tsv", sep="\t", header=False, index=False,
                                             float_format="%.6f")
        _, dfs = cafa_eval(OBO, str(d), gt_file, ia=IA, no_orphans=True, norm="cafa", prop="fill",
                           exclude=known, toi_file=TOI, th_step=0.01, n_cpu=8)
    b = dfs["f_micro_w"].reset_index(); r = b[b.ns == nsname]
    if len(r) == 0: return None, None
    r = r.iloc[0]; return float(r["f_micro_w"]), float(r["tau"])

def contribs(df, ontologies, gt, gt_exclude, tau, nsname):
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "p.tsv"
        df[["prot", "term", "score"]].to_csv(f, sep="\t", header=False, index=False, float_format="%.6f")
        pred = pred_parser(str(f), ontologies, gt, "fill", None, 4)
    P = dense(pred[nsname].matrix).astype(float); G = dense(gt[nsname].matrix).astype(bool)
    ont = ontologies[nsname]; toi = np.asarray(ont.toi_ia); ia = np.asarray(ont.ia)
    Pt = P[:, toi]; Gt = G[:, toi]; iat = ia[toi]
    if gt_exclude is not None:
        keep = ~dense(gt_exclude[nsname].matrix).astype(bool)[:, toi]
    else:
        keep = np.ones_like(Gt, dtype=bool)
    ge = (Pt >= tau); w = iat[None, :]
    tp = ((ge & Gt & keep) * w).sum(1); fp = ((ge & (~Gt) & keep) * w).sum(1)
    fn = (((~ge) & Gt & keep) * w).sum(1); ngt = ((Gt & keep) * w).sum(1)
    return tp, fp, fn, ngt

def fmicro(tp, fp, fn):
    TP, FP, FN = tp.sum(), fp.sum(), fn.sum()
    pr = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    rc = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    return 2 * pr * rc / (pr + rc) if (pr + rc) > 0 else 0.0

def load_pred(variant, cat):
    p = OUTB / variant / "predictions" / cat / f"{cat}.tsv"
    return pd.read_csv(p, sep="\t", header=None, names=["prot", "term", "score"])

results = {}
for cat in CELLS:
    gt_file, known = GTF[cat]
    # deployed anchor cross-check (once per cat, per aspect)
    dep_df = pd.read_csv(PC / "predictions" / cat / f"{cat}.tsv", sep="\t", header=None,
                         names=["prot", "term", "score"])
    for asp, nsname in NSMAP.items():
        key = f"{cat}-{asp}"
        log(f"===== {key} =====")
        ontologies = obo_parser(OBO, ("is_a", "part_of"), IA, False)
        ontologies = update_toi(ontologies, TOI)
        gt = gt_parser(gt_file, ontologies)
        gt_exclude = gt_exclude_parser(known, gt, ontologies) if known else None

        cell = {"deployed_anchor_ref": DEPLOYED_ANCHOR.get((cat, asp))}
        f_dep, _ = cell_f(dep_df, gt_file, known, nsname)
        cell["deployed_reproduced"] = round(f_dep, 5) if f_dep is not None else None

        arm = {}
        for variant in VARIANTS:
            df = load_pred(variant, cat)
            fv, tv = cell_f(df, gt_file, known, nsname)
            arm[variant] = {"f_micro_w": round(fv, 5) if fv is not None else None, "tau": tv}
            log(f"  {variant}: {arm[variant]}")
        cell["arms"] = arm
        # deltas vs baseline
        fb = arm["baseline"]["f_micro_w"]
        for v in ["sse", "sse_shuffled"]:
            fv = arm[v]["f_micro_w"]
            cell[f"delta_{v}_vs_baseline"] = round(fv - fb, 5) if (fv is not None and fb is not None) else None
        cell["delta_sse_vs_deployed"] = round(arm["sse"]["f_micro_w"] - f_dep, 5) \
            if (arm["sse"]["f_micro_w"] is not None and f_dep is not None) else None

        # paired bootstrap: sse vs baseline
        base_df = load_pred("baseline", cat); sse_df = load_pred("sse", cat)
        tau_b = arm["baseline"]["tau"]; tau_s = arm["sse"]["tau"]
        if tau_b is not None and tau_s is not None:
            tpb, fpb, fnb, ngtb = contribs(base_df, ontologies, gt, gt_exclude, tau_b, nsname)
            tps, fps, fns, ngts = contribs(sse_df, ontologies, gt, gt_exclude, tau_s, nsname)
            vb, vs = fmicro(tpb, fpb, fnb), fmicro(tps, fps, fns)
            parity = abs(vb - fb) < 3e-3 and abs(vs - arm["sse"]["f_micro_w"]) < 3e-3
            has = (ngtb > 0) | (ngts > 0); idx = np.where(has)[0]
            rng = np.random.default_rng(7); B = 2000; deltas = np.empty(B)
            for b in range(B):
                s = rng.choice(idx, size=len(idx), replace=True)
                deltas[b] = fmicro(tps[s], fps[s], fns[s]) - fmicro(tpb[s], fpb[s], fnb[s])
            lo, hi = np.percentile(deltas, [2.5, 97.5])
            cell["bootstrap"] = {"n_proteins": int(len(idx)), "B": B, "parity_ok": bool(parity),
                "my_baseline": round(vb, 5), "my_sse": round(vs, 5),
                "delta_mean": round(float(deltas.mean()), 5),
                "ci95": [round(float(lo), 5), round(float(hi), 5)],
                "frac_positive": round(float((deltas > 0).mean()), 4)}
            log(f"  bootstrap delta {deltas.mean():+.5f} CI[{lo:+.5f},{hi:+.5f}] parity={parity}")
        results[key] = cell
        json.dump(results, open(SSE / "modeB_cafa.json", "w"), indent=1, default=float)

json.dump(results, open(SSE / "modeB_cafa.json", "w"), indent=1, default=float)
log("MODE B CAFA DONE"); print(json.dumps(results, indent=1, default=float))
