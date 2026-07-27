"""SELECTIVE GENERATION SUBMISSION -- the TEMPORAL GATE (the decisive test).

The protein-holdout run banked +0.015 (LK) / +0.032 (PK) deployable, but the leakage probe proved the
whole gain is a GBM memorizing which GO terms surge in THIS eval window (term-target-encoding alone
reproduces the GBM precision; on disjoint terms the GBM advantage collapses to the uniform level).
A protein-holdout within one window cannot catch that. This is the mandated train-on-past gate.

REAL FORWARD-TEMPORAL SPLIT, fully frozen (CAFA_forever release windows):
  FIT   window = Sep_2025 -> Nov_2025 gains  (fit the GBM precision policy + choose phi + tau here)
  APPLY window = Nov_2025 -> Mar_2026 gains  (apply the FROZEN policy blind; this is the deployable test)
The two windows are temporally DISJOINT, so a term that surges only in Sep-Nov cannot help Nov-Mar:
window-specific term memorization is forbidden by construction, exactly as a live deployment would be.

Everything the policy sees is <= Nov_2025 (FIT labels) + t0 strata (v227 embeddings/IA/freq/depth/taxon,
classifier is v227-trained). The eval labels are the Nov-Mar gains, strictly after the fit window.
Frame: lab obo+IA, prop=fill, norm=cafa, no_orphans, toi; PK adds that window's -known. f_micro_w via
cafaeval's own parser/propagation; per-protein weighted TP/FP/FN validated against cafaeval.
"""
import json, tempfile, time, collections
from pathlib import Path
import numpy as np, pandas as pd
from scipy.sparse import issparse
from sklearn.isotonic import IsotonicRegression
import lightgbm as lgb, torch, torch.nn as nn
import pyarrow.parquet as pq
from cafaeval.parser import obo_parser, gt_parser, pred_parser, gt_exclude_parser, update_toi
from cafaeval.evaluation import cafa_eval

t0 = time.time()
ROOT = Path("/home/frapercan/Thesis2")
REL = ROOT / "CAFA_forever/data/releases"
PC = ROOT / "repositories/protea-reranker-lab/results/sparse_classifier/percut_rerank"
LAB = ROOT / "repositories/protea-reranker-lab/results/sparse_classifier"
GF = ROOT / "storage/cooc_experiment/generator_frames"
OBO = str(ROOT / "protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo")
IA_F = str(ROOT / "protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv")
TOI = str(REL / "Sep_2025_Mar_2026/groundtruth_terms_of_interest.txt")
OUT = ROOT / "storage/regen_headline"
NS = "biological_process"
FIT_WIN, APPLY_WIN = "Sep_2025_Nov_2025", "Nov_2025_Mar_2026"
TAUS = np.round(np.arange(0.01, 1.0001, 0.01), 2)
PHIS = [0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0]
FEATS = ["logit", "ia", "tf", "depth", "taxden"]
K_CAND = 50

# ---- ontology / IA (for candidate strata) -------------------------------------
par = collections.defaultdict(set); nsd = {}; alt = {}
cur_ = None
for line in open(OBO):
    line = line.rstrip("\n")
    if line == "[Term]": cur_ = None
    elif line.startswith("id: GO:"): cur_ = line[4:]
    elif cur_ and line.startswith("namespace: "): nsd[cur_] = line[11:]
    elif cur_ and line.startswith("is_a: GO:"): par[cur_].add(line[6:].split(" ! ")[0].strip())
    elif cur_ and line.startswith("relationship: part_of GO:"):
        par[cur_].add(line[len("relationship: part_of "):].split(" ! ")[0].strip())
    elif cur_ and line.startswith("alt_id: GO:"): alt[line[8:]] = cur_
BP = {t for t, n in nsd.items() if n == "biological_process"}
_AC = {}
def anc(t):
    if t in _AC: return _AC[t]
    seen, stack = set(), [t]
    while stack:
        x = stack.pop()
        for p in par.get(x, ()):
            if p not in seen: seen.add(p); stack.append(p)
    _AC[t] = seen; return seen
_D = {}
def depth(t):
    if t in _D: return _D[t]
    ps = [p for p in par.get(t, ()) if p in BP]
    _D[t] = 0 if not ps else 1 + max(depth(p) for p in ps); return _D[t]
IA = {}
for line in open(IA_F):
    p = line.rstrip("\n").split("\t")
    if len(p) >= 2:
        try: IA[p[0]] = float(p[1])
        except ValueError: pass

# ---- classifier logits over all 88,212 frozen v227 proteins (DB-free) ---------
ck = torch.load(ROOT / "storage/fullgo_models/classifier_6plm_asl.pt", map_location="cpu", weights_only=False)
vocab = [alt.get(g, g) for g in ck["vocab"]]; mu, sd = ck["mu"].numpy(), ck["sd"].numpy()
net = nn.Sequential(nn.Linear(8320, 1024), nn.LayerNorm(1024), nn.GELU(), nn.Dropout(0.3),
                    nn.Linear(1024, 1024), nn.LayerNorm(1024), nn.GELU(), nn.Dropout(0.3),
                    nn.Linear(1024, len(vocab)))
net.load_state_dict({k[4:] if k.startswith("net.") else k: v for k, v in ck["state_dict"].items()}, strict=False)
net.eval()
vb = [i for i, g in enumerate(vocab) if g in BP]; vbg = [vocab[i] for i in vb]
accs = json.load(open(GF / "accs.json")); apos = {a: i for i, a in enumerate(accs)}
ORDER = ["raw_ankh_base", "raw_esm2_3b", "raw_ankh_large", "raw_esm2_650m", "raw_esmc_600m", "raw_prott5"]
X = np.hstack([np.load(GF / f"{n}.npy") for n in ORDER]).astype(np.float32)
Xn = torch.tensor((X - mu) / sd, dtype=torch.float32); del X
with torch.no_grad():
    LOGbp = np.vstack([net(Xn[i:i + 512]).cpu().numpy()[:, vb] for i in range(0, len(Xn), 512)])
del Xn
print(f"classifier BP logits {LOGbp.shape}  ({time.time()-t0:.0f}s)", flush=True)

# per-protein taxon-density, per-term freq
taxon = {}
for line in open(OUT / "eval_acc_taxon.tsv"):
    if line.startswith("DONE"): continue
    pp = line.rstrip("\n").split("\t")
    if len(pp) == 2: taxon[pp[0]] = pp[1]
taxden_map = collections.Counter(taxon.values())
te = pq.read_table(LAB / "percut_rerank/eval.parquet", columns=["go_term_id", "go_term_frequency"])
TF = {}
for g, f in zip(np.asarray(te.column("go_term_id").to_pylist()), te.column("go_term_frequency").to_numpy(zero_copy_only=False)):
    if f == f: TF[alt.get(g, g)] = float(f)

def load_gt(win, cell):
    gt = collections.defaultdict(set); known = collections.defaultdict(set)
    for line in open(REL / win / f"groundtruth_{cell}.tsv"):
        x = line.rstrip("\n").split("\t")
        if len(x) >= 2 and x[1].startswith("GO:"):
            t = alt.get(x[1], x[1]); gt[x[0]].add(t); gt[x[0]].update(anc(t))
    kf = REL / win / f"groundtruth_{cell}_known.tsv"
    if kf.exists():
        for line in open(kf):
            x = line.rstrip("\n").split("\t")
            if len(x) >= 2 and x[1].startswith("GO:"): known[x[0]].add(alt.get(x[1], x[1]))
    return gt, known

def load_pool(cell):
    pool = collections.defaultdict(set)
    short = cell.split("_")[0].lower()
    for line in open(LAB / "percut_rerank/predictions" / short / f"{short}.tsv"):
        x = line.rstrip("\n").split("\t")
        if len(x) >= 3: pool[x[0]].add(alt.get(x[1], x[1]))
    return pool

def build_cands(win, cell, pool):
    gt, known = load_gt(win, cell)
    rows = []
    for p in sorted(set(gt) & set(apos)):
        i = apos[p]; lg = LOGbp[i]
        top = np.argpartition(-lg, K_CAND)[:K_CAND]; top = top[np.argsort(-lg[top])]
        pl = pool.get(p, set()); gtp = gt[p]; kn = known.get(p, set()) if cell == "PK" else set()
        for rank, j in enumerate(top):
            g = vbg[j]
            if g in pl or (cell == "PK" and g in kn): continue
            rows.append((p, g, float(lg[j]), rank, 1 if g in gtp else 0,
                         IA.get(g, 0.0), TF.get(g, 0.0), float(depth(g)),
                         float(taxden_map.get(taxon.get(p, "?"), 0))))
    return pd.DataFrame(rows, columns=["prot", "term", "logit", "rank", "y", "ia", "tf", "depth", "taxden"])

# ---- cafaeval machinery -------------------------------------------------------
def dense(M): return np.asarray(M.todense()) if issparse(M) else np.asarray(M)
def parse_pred(df, ontologies, gt):
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "p.tsv"
        df[["prot", "term", "score"]].to_csv(f, sep="\t", header=False, index=False, float_format="%.6f")
        return pred_parser(str(f), ontologies, gt, "fill", None, 4)
def cmats(pred, gt, gtex, ont):
    P = dense(pred[NS].matrix).astype(float); G = dense(gt[NS].matrix).astype(bool)
    o = ont[NS]; toi = np.asarray(o.toi_ia); ia = np.asarray(o.ia)
    Pt = P[:, toi]; Gt = G[:, toi]; iat = ia[toi]
    keep = ~dense(gtex[NS].matrix).astype(bool)[:, toi] if gtex is not None else np.ones_like(Gt, bool)
    return Pt, Gt, iat, keep
def ppc(Pt, Gt, iat, keep, tau):
    ge = (Pt >= tau); w = iat[None, :]
    return (((ge & Gt & keep) * w).sum(1), ((ge & (~Gt) & keep) * w).sum(1),
            (((~ge) & Gt & keep) * w).sum(1), ((Gt & keep) * w).sum(1))
def fmicro(tp, fp, fn):
    TP, FP, FN = tp.sum(), fp.sum(), fn.sum()
    pr = TP / (TP + FP) if TP + FP > 0 else 0.0; rc = TP / (TP + FN) if TP + FN > 0 else 0.0
    return 2 * pr * rc / (pr + rc) if pr + rc > 0 else 0.0
def best_tau(Pt, Gt, iat, keep):
    bf, bt = -1.0, TAUS[0]
    for tau in TAUS:
        tp, fp, fn, _ = ppc(Pt, Gt, iat, keep, tau); f = fmicro(tp, fp, fn)
        if f > bf: bf, bt = f, tau
    return round(bf, 5), float(bt)

GTFILE = {("Sep_2025_Nov_2025", "LK"): (str(REL / "Sep_2025_Nov_2025/groundtruth_LK.tsv"), None),
          ("Sep_2025_Nov_2025", "PK"): (str(REL / "Sep_2025_Nov_2025/groundtruth_PK.tsv"), str(REL / "Sep_2025_Nov_2025/groundtruth_PK_known.tsv")),
          ("Nov_2025_Mar_2026", "LK"): (str(REL / "Nov_2025_Mar_2026/groundtruth_LK.tsv"), None),
          ("Nov_2025_Mar_2026", "PK"): (str(REL / "Nov_2025_Mar_2026/groundtruth_PK.tsv"), str(REL / "Nov_2025_Mar_2026/groundtruth_PK_known.tsv"))}

def setup(win, cell):
    ont = update_toi(obo_parser(OBO, ("is_a", "part_of"), IA_F, False), TOI)
    gtf, known = GTFILE[(win, cell)]
    gt = gt_parser(gtf, ont); gtex = gt_exclude_parser(known, gt, ont) if known else None
    return ont, gt, gtex, gtf, known

def strata_prec_table(df):
    """IA-band x logit-quintile -> precision, for drift diagnosis."""
    tab = {}
    ql = np.quantile(df.logit.values, [0.2, 0.4, 0.6, 0.8]) if len(df) else [0, 0, 0, 0]
    lb = np.digitize(df.logit.values, ql)
    for ib, (lo, hi) in enumerate([(0, 2), (2, 4), (4, 99)]):
        for q in range(5):
            m = (df.ia.values >= lo) & (df.ia.values < hi) & (lb == q)
            if m.sum() >= 30:
                tab[f"IA[{lo},{hi})_logitQ{q}"] = round(float(df.y.values[m].mean()), 4)
    return tab

report = {}
for cell in ("LK", "PK"):
    print(f"\n[{time.time()-t0:.0f}s] ===== {cell}_BPO temporal gate =====", flush=True)
    pool = load_pool(cell)
    fit = build_cands(FIT_WIN, cell, pool)
    app = build_cands(APPLY_WIN, cell, pool)
    print(f"  FIT {FIT_WIN}: {len(fit):,} cand {fit.y.mean():.3f} prec | APPLY {APPLY_WIN}: {len(app):,} cand {app.y.mean():.3f} prec", flush=True)

    # deployed submissions per window
    dep = pd.read_csv(LAB / "percut_rerank/predictions" / cell.lower() / f"{cell.lower()}.tsv",
                      sep="\t", header=None, names=["prot", "term", "score"])
    ont_f, gt_f, gtex_f, gtf_f, kn_f = setup(FIT_WIN, cell)
    ont_a, gt_a, gtex_a, gtf_a, kn_a = setup(APPLY_WIN, cell)
    dep_f = dep[dep.prot.isin(set(fit.prot))]; dep_a = dep[dep.prot.isin(set(app.prot))]

    def submit(base, sel, s=1.0):
        if len(sel):
            add = sel[["prot", "term"]].copy(); add["score"] = s
            return pd.concat([base, add], ignore_index=True)
        return base

    # ---- fit GBM + isotonic on FIT window; sweep phi + tau on FIT ------------------
    gbm = lgb.train({"objective": "binary", "learning_rate": 0.05, "num_leaves": 15, "min_data_in_leaf": 50,
                     "feature_fraction": 0.8, "seed": 13, "verbose": -1, "num_threads": 8},
                    lgb.Dataset(fit[FEATS].values, label=fit.y.values), num_boost_round=200)
    iso = IsotonicRegression(out_of_bounds="clip").fit(fit.logit.values, fit.y.values)
    phat_fit = {"gbm": gbm.predict(fit[FEATS].values), "iso": iso.predict(fit.logit.values)}
    phat_app = {"gbm": gbm.predict(app[FEATS].values), "iso": iso.predict(app.logit.values)}

    Pt0, Gt0, iat0, keep0 = cmats(parse_pred(dep_f, ont_f, gt_f), gt_f, gtex_f, ont_f)
    f_fit_dep, tau_fit_dep = best_tau(Pt0, Gt0, iat0, keep0)
    sweep = {}
    for model in ("gbm", "iso"):
        order = np.argsort(-phat_fit[model]); n = len(fit); rec = {}
        for phi in PHIS:
            k = int(round(n * phi)); sel = fit.iloc[order[:k]] if k else fit.iloc[[]]
            Pt, Gt, iat, keep = cmats(parse_pred(submit(dep_f, sel), ont_f, gt_f), gt_f, gtex_f, ont_f)
            f, tau = best_tau(Pt, Gt, iat, keep); rec[phi] = {"k": k, "f": f, "tau": tau, "delta": round(f - f_fit_dep, 5)}
        sweep[model] = rec
        print(f"  FIT {model}: dep {f_fit_dep} " + " ".join(f"{phi}:{rec[phi]['delta']:+.4f}" for phi in PHIS), flush=True)
    best = max(((m, phi, sweep[m][phi]["delta"], sweep[m][phi]["tau"], sweep[m][phi]["k"])
                for m in sweep for phi in PHIS if phi > 0), key=lambda z: z[2])
    bmodel, bphi, bdelta_fit, btau_fit, bk_fit = best
    no_aug = bool(bdelta_fit <= 0)
    print(f"  FIT best: {bmodel} phi={bphi} delta={bdelta_fit:+.5f} tau={btau_fit}  (no_aug_optimal={no_aug})", flush=True)

    # ---- apply FROZEN policy to APPLY window -------------------------------------
    order_a = np.argsort(-phat_app[bmodel]); na = len(app); ka = int(round(na * bphi))
    sel_a = app.iloc[order_a[:ka]]
    uni_a = app.iloc[np.argsort(-app.logit.values)[:ka]]  # matched-volume uniform (top logit)

    A_Pt, A_Gt, A_iat, A_keep = cmats(parse_pred(dep_a, ont_a, gt_a), gt_a, gtex_a, ont_a)
    fA, tauA = best_tau(A_Pt, A_Gt, A_iat, A_keep)
    tp, fp, fn, _ = ppc(A_Pt, A_Gt, A_iat, A_keep, tau_fit_dep); fA_dep = round(fmicro(tp, fp, fn), 5)
    S_Pt, S_Gt, S_iat, S_keep = cmats(parse_pred(submit(dep_a, sel_a), ont_a, gt_a), gt_a, gtex_a, ont_a)
    fS_own, tauS_own = best_tau(S_Pt, S_Gt, S_iat, S_keep)
    tp, fp, fn, _ = ppc(S_Pt, S_Gt, S_iat, S_keep, btau_fit); fS_dep = round(fmicro(tp, fp, fn), 5)
    U_Pt, U_Gt, U_iat, U_keep = cmats(parse_pred(submit(dep_a, uni_a), ont_a, gt_a), gt_a, gtex_a, ont_a)
    tp, fp, fn, _ = ppc(U_Pt, U_Gt, U_iat, U_keep, btau_fit); fU_dep = round(fmicro(tp, fp, fn), 5)

    # cafaeval parity on APPLY deployed + selective
    def cafa_point(sub):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / "pd"; d.mkdir()
            sub[["prot", "term", "score"]].to_csv(d / "p.tsv", sep="\t", header=False, index=False, float_format="%.6f")
            _, dfs = cafa_eval(OBO, str(d), gtf_a, ia=IA_F, no_orphans=True, norm="cafa", prop="fill",
                               exclude=kn_a, toi_file=TOI, th_step=0.01, n_cpu=8)
        b = dfs["f_micro_w"].reset_index(); r = b[b.ns == NS].iloc[0]; return round(float(r["f_micro_w"]), 5)
    cA = cafa_point(dep_a); cS = cafa_point(submit(dep_a, sel_a))
    parity = abs(cA - fA) < 3e-3 and abs(cS - fS_own) < 3e-3

    # paired bootstrap over APPLY proteins (rows gt_a-aligned)
    tpA, fpA, fnA, ngtA = ppc(A_Pt, A_Gt, A_iat, A_keep, tau_fit_dep)
    tpS, fpS, fnS, ngtS = ppc(S_Pt, S_Gt, S_iat, S_keep, btau_fit)
    tpU, fpU, fnU, _ = ppc(U_Pt, U_Gt, U_iat, U_keep, btau_fit)
    kr = np.where((ngtA > 0) | (ngtS > 0))[0]
    aA = np.stack([tpA, fpA, fnA], 1)[kr]; aS = np.stack([tpS, fpS, fnS], 1)[kr]; aU = np.stack([tpU, fpU, fnU], 1)[kr]
    rb = np.random.default_rng(7); B = 2000; dSA = np.empty(B); dSU = np.empty(B); idx = np.arange(len(kr))
    for b in range(B):
        s = rb.choice(idx, len(idx), replace=True)
        dSA[b] = fmicro(aS[s, 0], aS[s, 1], aS[s, 2]) - fmicro(aA[s, 0], aA[s, 1], aA[s, 2])
        dSU[b] = fmicro(aS[s, 0], aS[s, 1], aS[s, 2]) - fmicro(aU[s, 0], aU[s, 1], aU[s, 2])

    # stratum-precision stability FIT vs APPLY
    tab_fit = strata_prec_table(fit); tab_app = strata_prec_table(app)
    common = sorted(set(tab_fit) & set(tab_app))
    drift_corr = float(np.corrcoef([tab_fit[k] for k in common], [tab_app[k] for k in common])[0, 1]) if len(common) >= 3 else None

    report[cell + "_BPO"] = {
        "TEMPORAL_gate": {"fit_window": FIT_WIN, "apply_window": APPLY_WIN,
                          "note": "temporally disjoint; policy sees only <=Nov_2025 labels + t0 strata"},
        "fit_cand_precision": round(float(fit.y.mean()), 4), "apply_cand_precision": round(float(app.y.mean()), 4),
        "precision_model_chosen": bmodel, "phi_chosen": bphi, "fit_delta": bdelta_fit,
        "fit_optimal_is_no_augmentation": no_aug, "tau_from_fit": btau_fit, "n_extras_applied": int(ka),
        "selective_extra_precision_apply": round(float(sel_a.y.mean()), 4) if len(sel_a) else None,
        "uniform_extra_precision_apply": round(float(uni_a.y.mean()), 4) if len(uni_a) else None,
        "DEPLOYED_A_deployable": fA_dep, "SELECTIVE_deployable": fS_dep, "UNIFORM_deployable": fU_dep,
        "DELTA_selective_vs_deployed_DEPLOYABLE": round(fS_dep - fA_dep, 5),
        "DELTA_selective_vs_uniform_DEPLOYABLE": round(fS_dep - fU_dep, 5),
        "SELECTIVE_apply_own_tau_ceiling": fS_own,
        "bootstrap": {"B": B, "n_proteins": int(len(kr)),
                      "delta_sel_vs_dep_mean": round(float(dSA.mean()), 5),
                      "ci95_sel_vs_dep": [round(float(np.percentile(dSA, 2.5)), 5), round(float(np.percentile(dSA, 97.5)), 5)],
                      "frac_pos_sel_vs_dep": round(float((dSA > 0).mean()), 4),
                      "delta_sel_vs_uni_mean": round(float(dSU.mean()), 5),
                      "ci95_sel_vs_uni": [round(float(np.percentile(dSU, 2.5)), 5), round(float(np.percentile(dSU, 97.5)), 5)]},
        "cafaeval_parity": {"A_cafa": cA, "A_mine": fA, "Sel_cafa": cS, "Sel_mine": fS_own, "ok": bool(parity)},
        "stratum_precision_stability": {"corr_fit_vs_apply": round(drift_corr, 4) if drift_corr is not None else None,
                                        "fit_table": tab_fit, "apply_table": tab_app},
        "fit_sweep": sweep,
    }
    print(f"  APPLY {cell}: A_dep {fA_dep} | Sel(deployable) {fS_dep} ({fS_dep-fA_dep:+.5f}) | "
          f"Uni {fU_dep} ({fS_dep-fU_dep:+.5f}) | sel-extra-prec {report[cell+'_BPO']['selective_extra_precision_apply']} | "
          f"boot CI {report[cell+'_BPO']['bootstrap']['ci95_sel_vs_dep']} | strata-corr {report[cell+'_BPO']['stratum_precision_stability']['corr_fit_vs_apply']} | parity {parity}", flush=True)
    json.dump(report, open(OUT / "SELECTIVE_GENERATION_TEMPORAL.json", "w"), indent=1)

json.dump(report, open(OUT / "SELECTIVE_GENERATION_TEMPORAL.json", "w"), indent=1)
print(f"\n[{time.time()-t0:.0f}s] TEMPORAL GATE DONE", flush=True)
