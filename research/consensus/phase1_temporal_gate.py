"""CROSS-MODALITY CONSENSUS -- Phase 1: the deployable temporal gate (confirmatory).

Phase 0 showed cross-modality agreement compounds precision (~23x from 1->3 modalities) but the 3-way
ceiling is only ~2.5-3% (IA-weighted ~2-2.5%), below the network 5.8% bar and far below the ~12%+
breakeven where the 11.6%-precision classifier arm already lost -0.013 under -known. The precision gate
therefore FAILS. This phase confirms it in the decisive metric: does ANY consensus-selected augmentation
of the DEPLOYED submission beat the canonical deployed anchor (PK 0.14351 / LK 0.31323) under the
TEMPORAL gate, with matched-volume control and bootstrap CI? Expected: no. We measure it anyway so the
NO-GO is airtight, not asserted.

Harness reused verbatim from multiplm_pool.py (the mandated temporal-gate machinery). The ONLY change:
the candidate/agreement signal is CROSS-MODALITY consensus (orthogonal channels: seq-kNN family, STRING,
classifier), not within-PLM agreement.
  FIT   Sep_2025 -> Nov_2025   (fit GBM precision policy + choose phi/tau BLIND to APPLY)
  APPLY Nov_2025 -> Mar_2026   (apply FROZEN policy; the deployable delta)
Features: consensus level (1-3), per-modality flags, seq within-family count (1-6), ia, tf, depth, taxden.
Anchor for the paired delta = the DEPLOYED submission recomputed on the APPLY window (NOT the raw pool).
A separate full-window cafa_eval reproduces the canonical 0.14351/0.31323 as a sanity precondition.
Controls: matched-volume UNIFORM random, paired protein bootstrap, cafaeval parity.

Leakage: seq proposals = t0 (v227, self-neighbour dropped) homolog transfers; STRING v12.0 (2023);
classifier trained on v227 (t0) experimental labels; classifier logits from CACHED frames (no live DB).
Candidate LABEL is the only post-t0 thing. READ-ONLY; writes only under storage/consensus/.
"""
import json, tempfile, time, collections, pickle
from pathlib import Path
import numpy as np, pandas as pd
from scipy.sparse import issparse
import lightgbm as lgb
import pyarrow.parquet as pq
from cafaeval.parser import obo_parser, gt_parser, pred_parser, gt_exclude_parser, update_toi
from cafaeval.evaluation import cafa_eval

t0 = time.time()
ROOT = Path("/home/frapercan/Thesis2")
REL = ROOT / "CAFA_forever/data/releases"
LAB = ROOT / "repositories/protea-reranker-lab/results/sparse_classifier"
SC = ROOT / "storage/regen_headline/recall_scratch"
CONS = ROOT / "storage/consensus"
OBO = str(ROOT / "protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo")
IA_F = str(ROOT / "protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv")
TOI = str(REL / "Sep_2025_Mar_2026/groundtruth_terms_of_interest.txt")
OUT = CONS
NS = "biological_process"
FIT_WIN, APPLY_WIN = "Sep_2025_Nov_2025", "Nov_2025_Mar_2026"
TAUS = np.round(np.arange(0.01, 1.0001, 0.01), 2)
PHIS = [0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0]
FEATS = ["level", "in_seq", "in_net", "in_clf", "seq_count", "ia", "tf", "depth", "taxden"]
DIVERSE = ["ankh_base", "ankh_large", "esm2_150m", "esm2_650m", "esm2_3b", "esmc_600m"]
EXTRA_SCORE = 1.0
ANCHOR = {"LK": 0.31323, "PK": 0.14351}

def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)

# ---- ontology / IA -------------------------------------------------------------
par = collections.defaultdict(set); nsd = {}; alt = {}; cur = None
for line in open(OBO):
    line = line.rstrip("\n")
    if line == "[Term]": cur = None
    elif line.startswith("id: GO:"): cur = line[4:]
    elif cur and line.startswith("namespace: "): nsd[cur] = line[11:]
    elif cur and line.startswith("is_a: GO:"): par[cur].add(line[6:].split(" ! ")[0].strip())
    elif cur and line.startswith("relationship: part_of GO:"):
        par[cur].add(line[len("relationship: part_of "):].split(" ! ")[0].strip())
    elif cur and line.startswith("alt_id: GO:"): alt[line[8:]] = cur
BP = {t for t, n in nsd.items() if n == "biological_process"}
_AC = {}
def anc(t):
    if t in _AC: return _AC[t]
    o, st = set(), [t]
    while st:
        x = st.pop()
        for p in par.get(x, ()):
            if p not in o: o.add(p); st.append(p)
    _AC[t] = o; return o
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

# ---- proposal caches: seq (per-PLM), net, classifier ---------------------------
toi_go = [alt.get(t, t) for t in json.load(open(SC / "toi_vocab.json"))["toi_go"]]
caches = {n: {str(k): v for k, v in pickle.load(open(SC / f"knn_prop_{n}.pkl", "rb")).items()} for n in DIVERSE}
net_raw = {str(k): v for k, v in pickle.load(open(SC / "net_prop_clean.pkl", "rb")).items()}
clf_prop = {p: set(s) for p, s in pickle.load(open(CONS / "clf_prop_topk.pkl", "rb")).items()}
log(f"loaded seq({len(DIVERSE)} PLM) + STRING({len(net_raw)}) + classifier({len(clf_prop)}) caches")

def seq_count_map(prot):
    """term -> number of PLMs (1..6) proposing it (within-family sub-signal)."""
    c = collections.Counter()
    for n in DIVERSE:
        v = caches[n].get(prot)
        if v is not None:
            for i in set(v.tolist()): c[toi_go[i]] += 1
    return c
def net_set(prot):
    v = net_raw.get(prot)
    return {toi_go[i] for i in v.tolist()} if v is not None else set()

taxon = {}
for line in open(ROOT / "storage/regen_headline/eval_acc_taxon.tsv"):
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
            t = alt.get(x[1], x[1]); gt[x[0]].add(t); gt[x[0]].update(a for a in anc(t) if a in BP)
    kf = REL / win / f"groundtruth_{cell}_known.tsv"
    if kf.exists():
        for line in open(kf):
            x = line.rstrip("\n").split("\t")
            if len(x) >= 2 and x[1].startswith("GO:"): known[x[0]].add(alt.get(x[1], x[1]))
    return gt, known

def load_pool(cell):
    pool = collections.defaultdict(set)
    short = cell.lower()
    for line in open(LAB / "percut_rerank/predictions" / short / f"{short}.tsv"):
        x = line.rstrip("\n").split("\t")
        if len(x) >= 3: pool[x[0]].add(alt.get(x[1], x[1]))
    return pool

def build_cands(win, cell, pool, dep_prots):
    gt, known = load_gt(win, cell)
    ckeys = set(caches["ankh_base"])
    rows = []
    for p in sorted(set(gt) & ckeys & dep_prots):
        sc = seq_count_map(p); ss = set(sc)
        ns_ = net_set(p); cs = clf_prop.get(p, set())
        pl = pool.get(p, set()); gtp = gt[p]; kn = known.get(p, set()) if cell == "PK" else set()
        cand = (ss | ns_ | cs) - pl - kn
        for term in cand:
            if term not in BP: continue
            insq, inn, inc = term in ss, term in ns_, term in cs
            lvl = int(insq) + int(inn) + int(inc)
            rows.append((p, term, lvl, int(insq), int(inn), int(inc), int(sc.get(term, 0)),
                         1 if term in gtp else 0, IA.get(term, 0.0), TF.get(term, 0.0),
                         float(depth(term)), float(taxden_map.get(taxon.get(p, "?"), 0))))
    return pd.DataFrame(rows, columns=["prot", "term", "level", "in_seq", "in_net", "in_clf",
                                       "seq_count", "y", "ia", "tf", "depth", "taxden"])

# ---- cafaeval machinery (verbatim from multiplm_pool) --------------------------
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

report = {"note": "cross-modality consensus (3 orthogonal channels) via the mandated temporal-gate harness",
          "phylo": "PENDING -> 3 modalities", "anchor_canonical_fullwindow": ANCHOR, "cells": {}}
for cell in ("LK", "PK"):
    print(f"\n[{time.time()-t0:.0f}s] ===== {cell}_BPO cross-modality consensus temporal gate =====", flush=True)
    pool = load_pool(cell)
    dep = pd.read_csv(LAB / "percut_rerank/predictions" / cell.lower() / f"{cell.lower()}.tsv",
                      sep="\t", header=None, names=["prot", "term", "score"])
    dep_prots = set(dep.prot)
    fit = build_cands(FIT_WIN, cell, pool, dep_prots)
    app = build_cands(APPLY_WIN, cell, pool, dep_prots)
    print(f"  FIT {FIT_WIN}: {len(fit):,} cand {fit.y.mean():.4f} prec ({fit.prot.nunique()} prot) | "
          f"APPLY {APPLY_WIN}: {len(app):,} cand {app.y.mean():.4f} prec ({app.prot.nunique()} prot)", flush=True)
    # precision by consensus level on each window (for the report)
    def lvl_prec(df):
        return {f"agree{L}": {"n": int((df.level == L).sum()), "prec": round(float(df.y[df.level == L].mean()), 4) if (df.level == L).any() else None}
                for L in (1, 2, 3)}

    ont_f, gt_f, gtex_f, gtf_f, kn_f = setup(FIT_WIN, cell)
    ont_a, gt_a, gtex_a, gtf_a, kn_a = setup(APPLY_WIN, cell)
    dep_f = dep[dep.prot.isin(set(fit.prot))]; dep_a = dep[dep.prot.isin(set(app.prot))]

    def submit(base, sel, s=EXTRA_SCORE):
        if len(sel):
            add = sel[["prot", "term"]].copy(); add["score"] = s
            return pd.concat([base, add], ignore_index=True)
        return base

    gbm = lgb.train({"objective": "binary", "learning_rate": 0.05, "num_leaves": 15, "min_data_in_leaf": 50,
                     "feature_fraction": 0.8, "seed": 13, "verbose": -1, "num_threads": 8},
                    lgb.Dataset(fit[FEATS].values, label=fit.y.values), num_boost_round=200)
    phat_fit = gbm.predict(fit[FEATS].values)
    phat_app = gbm.predict(app[FEATS].values)

    Pt0, Gt0, iat0, keep0 = cmats(parse_pred(dep_f, ont_f, gt_f), gt_f, gtex_f, ont_f)
    f_fit_dep, tau_fit_dep = best_tau(Pt0, Gt0, iat0, keep0)
    order = np.argsort(-phat_fit); n = len(fit); rec = {}
    for phi in PHIS:
        k = int(round(n * phi)); sel = fit.iloc[order[:k]] if k else fit.iloc[[]]
        Pt, Gt, iat, keep = cmats(parse_pred(submit(dep_f, sel), ont_f, gt_f), gt_f, gtex_f, ont_f)
        f, tau = best_tau(Pt, Gt, iat, keep)
        rec[phi] = {"k": k, "f": f, "tau": tau, "delta": round(f - f_fit_dep, 5),
                    "extra_prec": round(float(sel.y.mean()), 4) if k else None,
                    "extra_mean_level": round(float(sel.level.mean()), 3) if k else None}
    print(f"  FIT gbm: dep {f_fit_dep} " + " ".join(f"{phi}:{rec[phi]['delta']:+.4f}" for phi in PHIS), flush=True)
    best = max(((phi, rec[phi]["delta"], rec[phi]["tau"], rec[phi]["k"]) for phi in PHIS if phi > 0), key=lambda z: z[1])
    bphi, bdelta_fit, btau_fit, bk_fit = best
    no_aug = bool(bdelta_fit <= 0)
    print(f"  FIT best: phi={bphi} delta={bdelta_fit:+.5f} tau={btau_fit}  (no_aug_optimal={no_aug})", flush=True)

    order_a = np.argsort(-phat_app); na = len(app); ka = int(round(na * bphi))
    sel_a = app.iloc[order_a[:ka]]
    rng_u = np.random.default_rng(11)
    uni_idx = rng_u.choice(na, size=ka, replace=False) if ka else np.array([], dtype=int)
    uni_a = app.iloc[uni_idx]

    A_Pt, A_Gt, A_iat, A_keep = cmats(parse_pred(dep_a, ont_a, gt_a), gt_a, gtex_a, ont_a)
    fA, tauA = best_tau(A_Pt, A_Gt, A_iat, A_keep)
    tp, fp, fn, _ = ppc(A_Pt, A_Gt, A_iat, A_keep, tau_fit_dep); fA_dep = round(fmicro(tp, fp, fn), 5)
    S_Pt, S_Gt, S_iat, S_keep = cmats(parse_pred(submit(dep_a, sel_a), ont_a, gt_a), gt_a, gtex_a, ont_a)
    fS_own, tauS_own = best_tau(S_Pt, S_Gt, S_iat, S_keep)
    tp, fp, fn, _ = ppc(S_Pt, S_Gt, S_iat, S_keep, btau_fit); fS_dep = round(fmicro(tp, fp, fn), 5)
    U_Pt, U_Gt, U_iat, U_keep = cmats(parse_pred(submit(dep_a, uni_a), ont_a, gt_a), gt_a, gtex_a, ont_a)
    tp, fp, fn, _ = ppc(U_Pt, U_Gt, U_iat, U_keep, btau_fit); fU_dep = round(fmicro(tp, fp, fn), 5)

    def cafa_point(sub, gtf_, kn_):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / "pd"; d.mkdir()
            sub[["prot", "term", "score"]].to_csv(d / "p.tsv", sep="\t", header=False, index=False, float_format="%.6f")
            _, dfs = cafa_eval(OBO, str(d), gtf_, ia=IA_F, no_orphans=True, norm="cafa", prop="fill",
                               exclude=kn_, toi_file=TOI, th_step=0.01, n_cpu=8)
        b = dfs["f_micro_w"].reset_index(); r = b[b.ns == NS].iloc[0]; return round(float(r["f_micro_w"]), 5)
    cA = cafa_point(dep_a, gtf_a, kn_a); cS = cafa_point(submit(dep_a, sel_a), gtf_a, kn_a)
    parity = abs(cA - fA) < 3e-3 and abs(cS - fS_own) < 3e-3

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

    report[cell + "_BPO"] = {
        "candidate_source": "cross-modality union (seq-kNN family / STRING / classifier); consensus level 1-3",
        "TEMPORAL_gate": {"fit_window": FIT_WIN, "apply_window": APPLY_WIN},
        "n_cand_fit": int(len(fit)), "n_cand_apply": int(len(app)),
        "fit_cand_precision": round(float(fit.y.mean()), 4), "apply_cand_precision": round(float(app.y.mean()), 4),
        "fit_precision_by_level": lvl_prec(fit), "apply_precision_by_level": lvl_prec(app),
        "phi_chosen": bphi, "fit_delta": bdelta_fit, "fit_optimal_is_no_augmentation": no_aug,
        "tau_from_fit": btau_fit, "n_extras_applied": int(ka),
        "selective_extra_precision_apply": round(float(sel_a.y.mean()), 4) if len(sel_a) else None,
        "uniform_extra_precision_apply": round(float(uni_a.y.mean()), 4) if len(uni_a) else None,
        "selective_extra_mean_level": round(float(sel_a.level.mean()), 3) if len(sel_a) else None,
        "APPLYWIN_DEPLOYED_anchor": fA_dep, "APPLYWIN_SELECTIVE": fS_dep, "APPLYWIN_UNIFORM": fU_dep,
        "DELTA_selective_vs_deployed": round(fS_dep - fA_dep, 5),
        "DELTA_selective_vs_uniform": round(fS_dep - fU_dep, 5),
        "SELECTIVE_apply_own_tau_ceiling": fS_own,
        "bootstrap": {"B": B, "n_proteins": int(len(kr)),
                      "delta_sel_vs_dep_mean": round(float(dSA.mean()), 5),
                      "ci95_sel_vs_dep": [round(float(np.percentile(dSA, 2.5)), 5), round(float(np.percentile(dSA, 97.5)), 5)],
                      "frac_pos_sel_vs_dep": round(float((dSA > 0).mean()), 4),
                      "delta_sel_vs_uni_mean": round(float(dSU.mean()), 5),
                      "ci95_sel_vs_uni": [round(float(np.percentile(dSU, 2.5)), 5), round(float(np.percentile(dSU, 97.5)), 5)]},
        "cafaeval_parity": {"A_cafa": cA, "A_mine": fA, "Sel_cafa": cS, "Sel_mine": fS_own, "ok": bool(parity)},
        "fit_sweep": rec,
    }
    print(f"  APPLY {cell}: A_dep {fA_dep} | Sel {fS_dep} ({fS_dep-fA_dep:+.5f}) | Uni {fU_dep} ({fS_dep-fU_dep:+.5f}) | "
          f"sel-prec {report[cell+'_BPO']['selective_extra_precision_apply']} lvl {report[cell+'_BPO']['selective_extra_mean_level']} | "
          f"boot CI {report[cell+'_BPO']['bootstrap']['ci95_sel_vs_dep']} | parity {parity}", flush=True)
    json.dump(report, open(OUT / "phase1_temporal_gate.json", "w"), indent=1)

# ---- full-window deployed anchor reproduction (sanity vs canonical 0.14351/0.31323) ----
anchors = {}
for cell in ("LK", "PK"):
    gtf = str(REL / "Sep_2025_Mar_2026" / f"groundtruth_{cell}.tsv")
    kn = str(REL / "Sep_2025_Mar_2026" / f"groundtruth_{cell}_known.tsv") if cell == "PK" else None
    _, dfs = cafa_eval(OBO, str(LAB / "percut_rerank/predictions" / cell.lower()), gtf, ia=IA_F,
                       no_orphans=True, norm="cafa", prop="fill", exclude=kn, toi_file=TOI, th_step=0.01, n_cpu=8)
    b = dfs["f_micro_w"].reset_index(); r = b[b.ns == NS].iloc[0]
    anchors[cell] = {"reproduced": round(float(r["f_micro_w"]), 5), "canonical": ANCHOR[cell]}
    log(f"full-window deployed anchor {cell}: reproduced {anchors[cell]['reproduced']} vs canonical {ANCHOR[cell]}")
report["fullwindow_anchor_reproduction"] = anchors
json.dump(report, open(OUT / "phase1_temporal_gate.json", "w"), indent=1)
log("PHASE 1 DONE")
