"""Exact-parity paired protein bootstrap on the true-frame f_micro_w delta (blend - incumbent).

Reuses cafaeval's own obo/gt/pred parsing + DAG propagation, then computes per-protein weighted
TP/FP/FN at each arm's best tau (mirroring compute_confusion_matrix[_exclude]). Validates the pooled
f_micro_w against cafaeval's own point estimate before trusting the CI. Bootstraps proteins with
replacement (>=1000 resamples), paired (same resample indexes both arms).

Usage: pass which arm to bootstrap via ARMS below.
"""
import sys, json, tempfile, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.sparse import issparse
from cafaeval.parser import obo_parser, gt_parser, pred_parser, gt_exclude_parser, update_toi
from cafaeval.evaluation import cafa_eval

t0 = time.time()
ROOT = Path("/home/frapercan/Thesis2")
PC = ROOT / "repositories/protea-reranker-lab/results/sparse_classifier/percut_rerank"
GTDIR = ROOT / "repositories/protea-reranker-lab/results/sparse_classifier/lafa_gt"
OBO = str(ROOT / "protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo")
IA = str(ROOT / "protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv")
TOI = str(GTDIR / "groundtruth_terms_of_interest.txt")
OUT = ROOT / "storage/regen_headline"
GTF = {"lk": (str(GTDIR / "groundtruth_LK.tsv"), None),
       "pk": (str(GTDIR / "groundtruth_PK.tsv"), str(GTDIR / "groundtruth_PK_known.tsv"))}
NS = "biological_process"

ns_bp = {}; cur = None
for line in open(OBO):
    line = line.rstrip("\n")
    if line == "[Term]": cur = None
    elif line.startswith("id: GO:"): cur = line[4:]
    elif cur and line.startswith("namespace: "): ns_bp[cur] = line[11:]
BP = {t for t, n in ns_bp.items() if n == "biological_process"}


def minmax(x):
    x = np.nan_to_num(np.asarray(x, float)); lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def dense(M):
    return np.asarray(M.todense()) if issparse(M) else np.asarray(M)


def per_protein_contribs(sub_df, cat, ontologies, gt, gt_exclude, tau):
    """Write submission, parse (prop=fill), return per-protein (tp,fp,fn,ngt) weighted at tau."""
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "p.tsv"
        sub_df[["prot", "term", "score"]].to_csv(f, sep="\t", header=False, index=False, float_format="%.6f")
        pred = pred_parser(str(f), ontologies, gt, "fill", None, 4)
    P = dense(pred[NS].matrix).astype(float)
    G = dense(gt[NS].matrix).astype(bool)
    ont = ontologies[NS]
    toi = np.asarray(ont.toi_ia)
    ia = np.asarray(ont.ia)
    Pt = P[:, toi]; Gt = G[:, toi]; iat = ia[toi]
    if gt_exclude is not None:
        Ex = dense(gt_exclude[NS].matrix).astype(bool)[:, toi]
        keep = ~Ex
    else:
        keep = np.ones_like(Gt, dtype=bool)
    ge = (Pt >= tau)
    w = iat[None, :]
    tp = ((ge & Gt & keep) * w).sum(1)
    fp = ((ge & (~Gt) & keep) * w).sum(1)
    fn = (((~ge) & Gt & keep) * w).sum(1)
    ngt = ((Gt & keep) * w).sum(1)
    return tp, fp, fn, ngt


def fmicro(tp, fp, fn):
    TP, FP, FN = tp.sum(), fp.sum(), fn.sum()
    pr = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    rc = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    return 2 * pr * rc / (pr + rc) if (pr + rc) > 0 else 0.0


# which arm to bootstrap per cell: (model_col, add_weight, best_tau_hint) — tau re-derived below
SPEC = json.load(open(sys.argv[1])) if len(sys.argv) > 1 else {
    "pk": {"model": "gbdt_known_transition", "w": 1.0, "source": "phase1"},
    "lk": {"model": "gbdt_known_transition", "w": 1.0, "source": "phase1"},
}

results = {}
for cat in ["pk", "lk"]:
    spec = SPEC[cat]
    print(f"\n[{time.time()-t0:.0f}s] ===== {cat.upper()} bootstrap ({spec}) =====", flush=True)
    gt_file, known = GTF[cat]
    ontologies = obo_parser(OBO, ("is_a", "part_of"), IA, False)   # orphans=False (no_orphans=True)
    ontologies = update_toi(ontologies, TOI)
    gt = gt_parser(gt_file, ontologies)
    gt_exclude = gt_exclude_parser(known, gt, ontologies) if known else None

    # build submissions over deployed candidate set
    src = pd.read_parquet(OUT / f"phase1_scores_{cat}.parquet").rename(
        columns={"protein_accession": "prot", "go_term_id": "term"}).drop_duplicates(["prot", "term"])
    full = pd.read_csv(PC / "predictions" / cat / f"{cat}.tsv", sep="\t", header=None, names=["prot", "term", "score"])
    full["isbp"] = full.term.isin(BP)
    full = full.merge(src[["prot", "term", spec["model"]]], on=["prot", "term"], how="left")
    bp = full.isbp.values; dep = full.score.values.astype(float)
    mvec = np.zeros(len(full)); mk = bp & full[spec["model"]].notna().values
    mvec[mk] = minmax(full.loc[mk, spec["model"]].values)
    blend = dep.copy(); blend[bp] = np.clip(dep[bp] + spec["w"] * mvec[bp], 0, 1)
    inc_df = full.assign(score=dep); bl_df = full.assign(score=blend)

    # best taus via cafaeval (authoritative)
    def cell_f(df):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / "pd"; d.mkdir()
            df[["prot", "term", "score"]].to_csv(d / "p.tsv", sep="\t", header=False, index=False, float_format="%.6f")
            _, dfs = cafa_eval(OBO, str(d), gt_file, ia=IA, no_orphans=True, norm="cafa", prop="fill",
                               exclude=known, toi_file=TOI, th_step=0.01, n_cpu=8)
        b = dfs["f_micro_w"].reset_index(); r = b[b.ns == NS].iloc[0]
        return float(r["f_micro_w"]), float(r["tau"])
    f_inc, tau_inc = cell_f(inc_df)
    f_bl, tau_bl = cell_f(bl_df)
    print(f"  cafaeval: incumbent {f_inc:.5f}@{tau_inc} | blend {f_bl:.5f}@{tau_bl} delta {f_bl-f_inc:+.5f}", flush=True)

    tpi, fpi, fni, ngti = per_protein_contribs(inc_df, cat, ontologies, gt, gt_exclude, tau_inc)
    tpb, fpb, fnb, ngtb = per_protein_contribs(bl_df, cat, ontologies, gt, gt_exclude, tau_bl)
    # validate parity
    vi, vb = fmicro(tpi, fpi, fni), fmicro(tpb, fpb, fnb)
    print(f"  parity check: my incumbent {vi:.5f} (cafa {f_inc:.5f}) | my blend {vb:.5f} (cafa {f_bl:.5f})", flush=True)
    ok = abs(vi - f_inc) < 2e-3 and abs(vb - f_bl) < 2e-3

    # bootstrap over proteins that have gt (ngt>0 in either)
    has = (ngti > 0) | (ngtb > 0)
    idx = np.where(has)[0]
    rng = np.random.default_rng(7)
    B = 2000
    deltas = np.empty(B)
    for b in range(B):
        s = rng.choice(idx, size=len(idx), replace=True)
        deltas[b] = fmicro(tpb[s], fpb[s], fnb[s]) - fmicro(tpi[s], fpi[s], fni[s])
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    fracpos = float((deltas > 0).mean())
    results[cat] = {"spec": spec, "cafa_incumbent": round(f_inc, 5), "cafa_blend": round(f_bl, 5),
                    "point_delta": round(f_bl - f_inc, 5), "tau_incumbent": tau_inc, "tau_blend": tau_bl,
                    "parity_ok": bool(ok), "my_incumbent": round(vi, 5), "my_blend": round(vb, 5),
                    "n_proteins": int(len(idx)), "bootstrap_B": B,
                    "delta_mean": round(float(deltas.mean()), 5),
                    "ci95": [round(float(lo), 5), round(float(hi), 5)],
                    "frac_positive": round(fracpos, 4)}
    print(f"  bootstrap delta mean {deltas.mean():+.5f} CI95 [{lo:+.5f},{hi:+.5f}] frac_pos {fracpos:.3f} parity_ok={ok}", flush=True)
    json.dump(results, open(OUT / "bootstrap_ci.json", "w"), indent=2)

print(f"\n[{time.time()-t0:.0f}s] DONE"); print(json.dumps(results, indent=2))
