"""ProtEx verification -- STAGE 4: the deciders.

(1) SEPARABILITY (diagnostic): per-protein AUC of true-vs-false BP-tail candidates for the
    exemplar verifier (with_neg / no_neg) vs the FLAT deployed baselines (reranker_score, kNN
    -distance). Overall and on the IA>=4 tail. This is the crux question: does pos-vs-neg exemplar
    conditioning convert the flat tail into a ranking signal?

(2) f_micro_w (DECIDER): true board frame -- lab obo+IA, prop=fill, norm=cafa, no_orphans, toi;
    PK adds -known (evaluation.nf:279). Anchor A = deployed pool reranker_score (reproduces the prior
    kwta anchors 0.11033 pk / 0.29896 lk). Arm B = the SAME candidate set re-ordered by the verifier
    (calibrated onto the pool score histogram, so only ORDER changes -- a pure separability->metric
    test). Paired protein bootstrap CI via cafaeval's own parser (exact-parity, mirrors bootstrap_ci.py).

Verdict rests on f_micro_w under the temporal gate with its CI; separability AUC is never the verdict.
"""
import json, time, tempfile, collections
from pathlib import Path
import numpy as np, pandas as pd
from scipy.sparse import issparse
from cafaeval.parser import obo_parser, gt_parser, pred_parser, gt_exclude_parser, update_toi
from cafaeval.evaluation import cafa_eval

t0 = time.time()
def log(m): print(f"[{time.time()-t0:.0f}s] {m}", flush=True)
W = Path("/home/frapercan/Thesis2/storage/protex")
REL = Path("/home/frapercan/Thesis2/CAFA_forever/data/releases/Sep_2025_Mar_2026")
T0D = Path("/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025")
OBO = str(T0D / "go-basic.obo"); IA_F = str(T0D / "IA.tsv")
TOI = str(REL / "groundtruth_terms_of_interest.txt")
NS = "biological_process"
GTF = {"pk": (str(REL / "groundtruth_PK.tsv"), str(REL / "groundtruth_PK_known.tsv")),
       "lk": (str(REL / "groundtruth_LK.tsv"), None)}
NOISE = 0.0034

# ---- ontology / IA / closures for separability labelling ----
par = collections.defaultdict(set); ns = {}; alt = {}; cur = None
for line in open(OBO):
    line = line.rstrip("\n")
    if line == "[Term]": cur = None
    elif line.startswith("id: GO:"): cur = line[4:]
    elif cur and line.startswith("namespace: "): ns[cur] = line[11:]
    elif cur and line.startswith("is_a: GO:"): par[cur].add(line[6:].split(" ! ")[0].strip())
    elif cur and line.startswith("relationship: part_of GO:"): par[cur].add(line[len("relationship: part_of "):].split(" ! ")[0].strip())
    elif cur and line.startswith("alt_id: GO:"): alt[line[8:]] = cur
BP = {t for t, n in ns.items() if n == "biological_process"}
AC = {}
def anc(t):
    if t in AC: return AC[t]
    o, st = set(), [t]
    while st:
        x = st.pop()
        for p in par.get(x, ()):
            if p not in o: o.add(p); st.append(p)
    AC[t] = o; return o
def closure(ts):
    o = set()
    for g in ts: o.add(g); o |= anc(g)
    return o
IA = {}
for line in open(IA_F):
    p = line.rstrip("\n").split("\t")
    if len(p) >= 2:
        try: IA[p[0]] = float(p[1])
        except ValueError: pass
def load_gt(fn):
    G = collections.defaultdict(set)
    with open(fn) as fh:
        next(fh)
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) >= 3 and f[2] == "P": G[f[0]].add(alt.get(f[1], f[1]))
    return {p: {g for g in closure(t) if g in BP} for p, t in G.items()}
gt_bp = {"pk": load_gt(GTF["pk"][0]), "lk": load_gt(GTF["lk"][0])}
known_bp = collections.defaultdict(set)
with open(GTF["pk"][1]) as fh:
    next(fh)
    for line in fh:
        f = line.rstrip("\n").split("\t")
        if f[2] == "P": known_bp[f[0]] |= {x for x in closure([alt.get(f[1], f[1])]) if x in BP}
log("ontology + GT loaded")

# ---- scores + exemplar meta (index-aligned; both from eval_scores order) ----
ts = np.load(W / "test_scores.npz", allow_pickle=True)
ex = np.load(W / "exemplars_test.npz", allow_pickle=True)
P = ts["protein"]; G = ts["term"]; CELL = ts["cell"]
assert np.array_equal(P, ex["protein"]) and np.array_equal(G, ex["term"]), "test_scores/exemplars misaligned"
COLS = {"reranker": ts["reranker_score"].astype(np.float64),
        "knn_negdist": -ex["distance"].astype(np.float64),
        "with_neg": ts["with_neg"].astype(np.float64),
        "no_neg": ts["no_neg"].astype(np.float64)}
n_pos = ts["n_pos"]

def pp_auc(y, s):
    o = np.argsort(s, kind="stable"); r = np.empty(len(s)); r[o] = np.arange(len(s))
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0: return None
    return float((r[y == 1].sum() - n1 * (n1 - 1) / 2) / (n1 * n0))

# ---- Part 1: separability ----
sep = {}
for cell in ("pk", "lk"):
    m = np.where(CELL == cell)[0]
    gtc = gt_bp[cell]
    by = collections.defaultdict(list)
    for i in m: by[P[i]].append(i)
    accs_all = {k: [] for k in COLS}; accs_tail = {k: [] for k in COLS}
    npos_exempl = []
    for prot, idxs in by.items():
        gset = gtc.get(prot, set()); kset = known_bp.get(prot, set()) if cell == "pk" else set()
        rows = [i for i in idxs if G[i] not in kset]        # novel-eligible only
        if len(rows) < 5: continue
        y = np.array([1.0 if G[i] in gset else 0.0 for i in rows])
        if y.sum() == 0 or y.sum() == len(y): continue
        tail = np.array([IA.get(G[i], 0.0) >= 4.0 for i in rows])
        for k, col in COLS.items():
            s = col[rows]
            a = pp_auc(y, s)
            if a is not None: accs_all[k].append(a)
            if tail.sum() >= 5 and y[tail].sum() not in (0, tail.sum()):
                at = pp_auc(y[tail], s[tail])
                if at is not None: accs_tail[k].append(at)
        npos_exempl.append(float((n_pos[rows] > 0).mean()))
    sep[cell] = {"n_prots_scored": len(accs_all["reranker"]),
                 "mean_auc_all": {k: round(float(np.mean(v)), 4) if v else None for k, v in accs_all.items()},
                 "mean_auc_IA>=4_tail": {k: round(float(np.mean(v)), 4) if v else None for k, v in accs_tail.items()},
                 "n_prots_tail": len(accs_tail["reranker"]),
                 "mean_frac_cands_with_pos_exemplar": round(float(np.mean(npos_exempl)), 3) if npos_exempl else None}
    log(f"SEP {cell}: all {sep[cell]['mean_auc_all']} | tail {sep[cell]['mean_auc_IA>=4_tail']}")

# ---- Part 2: f_micro_w true frame + paired bootstrap CI ----
def dense(M): return np.asarray(M.todense()) if issparse(M) else np.asarray(M)
def fmicro(tp, fp, fn):
    TP, FP, FN = tp.sum(), fp.sum(), fn.sum()
    pr = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    rc = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    return 2 * pr * rc / (pr + rc) if (pr + rc) > 0 else 0.0
def contribs(df, ontologies, gt, gt_excl, tau):
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "p.tsv"
        df[["prot", "term", "score"]].to_csv(f, sep="\t", header=False, index=False, float_format="%.6f")
        pred = pred_parser(str(f), ontologies, gt, "fill", None, 4)
    Pm = dense(pred[NS].matrix).astype(float); Gm = dense(gt[NS].matrix).astype(bool)
    ont = ontologies[NS]; toi = np.asarray(ont.toi_ia); ia = np.asarray(ont.ia)
    Pt = Pm[:, toi]; Gt = Gm[:, toi]; iat = ia[toi]
    keep = ~dense(gt_excl[NS].matrix).astype(bool)[:, toi] if gt_excl is not None else np.ones_like(Gt, bool)
    ge = Pt >= tau; w = iat[None, :]
    return (((ge & Gt & keep) * w).sum(1), ((ge & (~Gt) & keep) * w).sum(1),
            (((~ge) & Gt & keep) * w).sum(1), ((Gt & keep) * w).sum(1))
def cell_f(df, gt_file, known):
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "pd"; d.mkdir()
        df[["prot", "term", "score"]].to_csv(d / "p.tsv", sep="\t", header=False, index=False, float_format="%.6f")
        _, dfs = cafa_eval(OBO, str(d), gt_file, ia=IA_F, no_orphans=True, norm="cafa", prop="fill",
                           exclude=known, toi_file=TOI, th_step=0.01, n_cpu=8)
    b = dfs["f_micro_w"].reset_index(); r = b[b.ns == NS]
    if len(r) == 0: return None, None
    r = r.iloc[0]; return float(r["f_micro_w"]), float(r["tau"])
def calibrate(model, ref):
    order = np.argsort(np.argsort(model)); q = (order + 0.5) / len(model)
    return np.quantile(ref, q)

fres = {}
for cell in ("pk", "lk"):
    m = np.where(CELL == cell)[0]
    prot = P[m]; term = G[m]; rer = COLS["reranker"][m]
    gt_file, known = GTF[cell]
    ontologies = update_toi(obo_parser(OBO, ("is_a", "part_of"), IA_F, False), TOI)
    gt = gt_parser(gt_file, ontologies)
    gt_excl = gt_exclude_parser(known, gt, ontologies) if known else None
    rrank = np.argsort(np.argsort(rer)).astype(np.float64)
    vrank = np.argsort(np.argsort(COLS["with_neg"][m])).astype(np.float64)
    rng_ctrl = np.random.default_rng(123)
    arms = {"A_reranker": rer,
            "B_withneg": calibrate(COLS["with_neg"][m], rer),
            "B_noneg": calibrate(COLS["no_neg"][m], rer),
            "B_blend_rerank+withneg": calibrate(rrank + vrank, rer),
            "CONTROL_random_order": calibrate(rng_ctrl.random(len(m)), rer)}
    dfA = pd.DataFrame({"prot": prot, "term": term, "score": arms["A_reranker"]})
    fA, tauA = cell_f(dfA, gt_file, known)
    tpA, fpA, fnA, ngtA = contribs(dfA, ontologies, gt, gt_excl, tauA)
    log(f"F {cell}: A={fA:.5f}@{tauA} (parity {fmicro(tpA,fpA,fnA):.5f})")
    fres[cell] = {"anchor_A": round(fA, 5), "tau_A": tauA, "n_cands": int(len(m)),
                  "parity_A": round(fmicro(tpA, fpA, fnA), 5), "arms": {}}
    rng = np.random.default_rng(7)
    for name in ("B_withneg", "B_noneg", "B_blend_rerank+withneg", "CONTROL_random_order"):
        dfB = pd.DataFrame({"prot": prot, "term": term, "score": arms[name]})
        fB, tauB = cell_f(dfB, gt_file, known)
        tpB, fpB, fnB, ngtB = contribs(dfB, ontologies, gt, gt_excl, tauB)
        # paired protein bootstrap on (B - A); align per-protein arrays (same protein order in parser)
        has = (ngtA > 0) | (ngtB > 0); idx = np.where(has)[0]
        deltas = np.empty(2000)
        for b in range(2000):
            s = rng.choice(idx, len(idx), replace=True)
            deltas[b] = fmicro(tpB[s], fpB[s], fnB[s]) - fmicro(tpA[s], fpA[s], fnA[s])
        lo, hi = np.percentile(deltas, [2.5, 97.5])
        fres[cell]["arms"][name] = {"f": round(fB, 5), "tau": tauB, "delta": round(fB - fA, 5),
                                    "parity": round(fmicro(tpB, fpB, fnB), 5),
                                    "boot_delta_mean": round(float(deltas.mean()), 5),
                                    "ci95": [round(float(lo), 5), round(float(hi), 5)],
                                    "frac_positive": round(float((deltas > 0).mean()), 4),
                                    "clears_noise": bool((fB - fA) > NOISE)}
        r = fres[cell]["arms"][name]
        log(f"F {cell} {name}: f={fB:.5f} d={fB-fA:+.5f} CI[{lo:+.5f},{hi:+.5f}] fpos={r['frac_positive']} clears={r['clears_noise']}")
    json.dump({"separability": sep, "fmicrow": fres}, open(W / "protex_result.json", "w"), indent=1)

# ---- verdict ----
best = None
for cell in ("pk", "lk"):
    for name, a in fres[cell]["arms"].items():
        if best is None or a["delta"] > best[2]: best = (cell, name, a["delta"], a["ci95"], a["clears_noise"])
go = any(a["clears_noise"] and a["ci95"][0] > 0 for c in fres.values() for a in c["arms"].values())
verdict = {"GO": bool(go),
           "one_line": ("GO -- a verifier arm clears the noise floor with CI>0"
                        if go else "NO-GO -- no verifier arm clears the 0.0034 floor with CI lower bound > 0"),
           "best_arm": {"cell": best[0], "arm": best[1], "delta": best[2], "ci95": best[3]},
           "separability_crux": {c: {"reranker_tail_auc": sep[c]["mean_auc_IA>=4_tail"]["reranker"],
                                     "withneg_tail_auc": sep[c]["mean_auc_IA>=4_tail"]["with_neg"],
                                     "noneg_tail_auc": sep[c]["mean_auc_IA>=4_tail"]["no_neg"]} for c in ("pk", "lk")}}
out = {"frame": "TRUE board frame; prop=fill norm=cafa no_orphans toi; PK -known (evaluation.nf:279)",
       "space": "ProtT5 frozen v227 (084943c6), exemplars t0-clean", "noise_floor": NOISE,
       "separability": sep, "fmicrow": fres, "verdict": verdict}
json.dump(out, open(W / "protex_result.json", "w"), indent=1)
log(f"DONE. verdict={verdict['one_line']}")
