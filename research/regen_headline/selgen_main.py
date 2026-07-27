"""SELECTIVE GENERATION SUBMISSION -- the deployable test.

QUESTION. We have always submitted generated candidates top-k UNIFORMLY. Precision is concentrated
(recon: PK top-1% logit ~20%, low-IA terms higher precision, human/mouse/zebrafish highest taxa).
Can we bank a real deployable BP gain on LK-BPO / PK-BPO by submitting generated candidates ONLY in
high-precision strata, where predicted precision clears the fill-tax break-even, instead of top-k
uniformly?

PRIMARY SOURCE. The classifier extras (aggregate 11.6%, the best generator), scored DB-FREE from the
cached v227 frames (recon: LK 6.14% / PK 9.73% aggregate over the top-50 BP-logit extras per protein,
because we include a deeper candidate pool than the top5 the 11.6% was measured at).

POLICY. Predict each candidate's precision p_hat from its t0 strata (source logit, term IA, DAG depth,
term freq, taxon-density). Include a candidate iff p_hat is in the top fraction phi -> submit the
selected extras at score 1.0 (a commitment to submit). Sweep phi on a HELD-OUT tune split to maximize
true-frame f_micro_w; apply the chosen phi (and tau) BLIND. Compare against UNIFORM top-logit of the
SAME volume, so any win is attributable to SELECTIVITY not volume.

TEMPORAL / LEAKAGE DISCIPLINE.
  * Generation is train-on-past by construction: classifier is v227(t0)-trained, its input frame is
    the cached v227 embeddings, and the eval window v227->v230 is strictly after (cache_generator_frames
    certified 0 leaked pairs). All strata features are t0-derived (asserted below).
  * The candidate LABEL is the post-t0 board GT (the target). The precision model + inclusion threshold
    are fit on a DISJOINT protein TUNE split and applied to a BLIND split -> a policy that wins only
    in-window (fit==eval) but loses held-out is a calibration artefact and is reported as such.
  * NOTE (stated honestly): a genuine v225-227 GT for GENERATED candidates over these same proteins
    does not exist (the board GT is the single v227->v230 window). The protein-level held-out split is
    therefore the generalization gate; the train-on-past guarantee sits at the generation level.

FRAME. Lab obo+IA, prop=fill, norm=cafa, no_orphans, toi; PK adds -known (evaluation.nf:279). All
f_micro_w via cafaeval's own parser/propagation; per-protein weighted TP/FP/FN at a fixed tau mirror
compute_confusion_matrix, validated against cafaeval's point estimate before any CI is trusted.
"""
import json, tempfile, time, collections
from pathlib import Path
import numpy as np, pandas as pd
from scipy.sparse import issparse
from sklearn.isotonic import IsotonicRegression
import lightgbm as lgb
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
TAUS = np.round(np.arange(0.01, 1.0001, 0.01), 2)
PHIS = [0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0]
SEED = 13


def dense(M):
    return np.asarray(M.todense()) if issparse(M) else np.asarray(M)


def parse_pred(df, ontologies, gt):
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "p.tsv"
        df[["prot", "term", "score"]].to_csv(f, sep="\t", header=False, index=False, float_format="%.6f")
        return pred_parser(str(f), ontologies, gt, "fill", None, 4)


def contrib_matrices(pred, gt, gt_exclude, ontologies):
    """Return propagated P (float, toi cols), G (bool), ia weights, keep mask.

    pred[NS].matrix is allocated to gt[NS].matrix.shape, so rows are the SAME protein order as gt for
    every prediction file parsed against this gt -> arms are row-aligned by construction (no protein
    name mapping needed for the paired bootstrap)."""
    P = dense(pred[NS].matrix).astype(float)
    G = dense(gt[NS].matrix).astype(bool)
    ont = ontologies[NS]
    toi = np.asarray(ont.toi_ia); ia = np.asarray(ont.ia)
    Pt = P[:, toi]; Gt = G[:, toi]; iat = ia[toi]
    if gt_exclude is not None:
        Ex = dense(gt_exclude[NS].matrix).astype(bool)[:, toi]
        keep = ~Ex
    else:
        keep = np.ones_like(Gt, dtype=bool)
    return Pt, Gt, iat, keep


def pp_contribs(Pt, Gt, iat, keep, tau):
    ge = (Pt >= tau); w = iat[None, :]
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


def best_tau(Pt, Gt, iat, keep, prot_mask=None):
    """f_micro_w over the (optionally masked) proteins, swept over TAUS; return (f, tau)."""
    bestf, bestt = -1.0, TAUS[0]
    for tau in TAUS:
        tp, fp, fn, _ = pp_contribs(Pt, Gt, iat, keep, tau)
        if prot_mask is not None:
            tp, fp, fn = tp[prot_mask], fp[prot_mask], fn[prot_mask]
        f = fmicro(tp, fp, fn)
        if f > bestf:
            bestf, bestt = f, tau
    return round(bestf, 5), float(bestt)


results = {}
for cat in ["lk", "pk"]:
    print(f"\n[{time.time()-t0:.0f}s] ===== {cat.upper()}_BPO =====", flush=True)
    gt_file, known = GTF[cat]
    ontologies = obo_parser(OBO, ("is_a", "part_of"), IA, False)
    ontologies = update_toi(ontologies, TOI)

    # deployed submission (all aspects; BP is scored per-ns)
    dep = pd.read_csv(PC / "predictions" / cat / f"{cat}.tsv", sep="\t", header=None,
                      names=["prot", "term", "score"])

    # candidate extras + t0 strata + label (from recon)
    R = np.load(OUT / f"selgen_cand_{cat}.npy")
    cand = pd.DataFrame({"prot": R["p"], "term": R["g"], "logit": R["logit"].astype(float),
                         "rank": R["rank"], "y": R["y"].astype(int), "ia": R["ia"].astype(float),
                         "tf": R["tf"].astype(float), "depth": R["depth"].astype(float),
                         "taxden": R["taxden"].astype(float)})
    # LEAKAGE ASSERT: every strata column is t0-derived (no board-GT-derived column except y)
    assert set(cand.columns) >= {"logit", "ia", "tf", "depth", "taxden"}, "strata missing"

    # protein-level held-out split of GT proteins (disjoint tune/blind)
    gtprots = sorted(set(dep.prot) | set(cand.prot))
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(gtprots)
    half = len(perm) // 2
    tune_prots = set(perm[:half]); blind_prots = set(perm[half:])

    # Split-specific GT: the deployed/blind evaluation must NOT see tune proteins (they would count as
    # all-FN and deflate recall). Write filtered gt (+ known) per split and parse each independently.
    SCR = OUT / "selgen_tmp"; SCR.mkdir(exist_ok=True)
    def build_split(tag, prots):
        gf = SCR / f"gt_{cat}_{tag}.tsv"
        with open(gt_file) as fi, open(gf, "w") as fo:
            for line in fi:
                if line.split(None, 1) and line.split(None, 1)[0] in prots:
                    fo.write(line)
        g = gt_parser(str(gf), ontologies)
        kf, gex = None, None
        if known:
            kf = SCR / f"known_{cat}_{tag}.tsv"
            with open(known) as fi, open(kf, "w") as fo:
                for line in fi:
                    if line.split(None, 1) and line.split(None, 1)[0] in prots:
                        fo.write(line)
            gex = gt_exclude_parser(str(kf), g, ontologies)
        return g, gex, str(gf), (str(kf) if kf else None)
    gt_tune, gtex_tune, gtf_tune, knf_tune = build_split("tune", tune_prots)
    gt_blind, gtex_blind, gtf_blind, knf_blind = build_split("blind", blind_prots)

    def submission(base, extras_df, extra_score=1.0):
        if len(extras_df):
            add = extras_df[["prot", "term"]].copy(); add["score"] = extra_score
            return pd.concat([base, add], ignore_index=True)
        return base

    # ---- fit precision model on TUNE candidates -----------------------------------
    ct = cand[cand.prot.isin(tune_prots)].reset_index(drop=True)
    cb = cand[cand.prot.isin(blind_prots)].reset_index(drop=True)
    feats = ["logit", "ia", "tf", "depth", "taxden"]
    # (a) isotonic on logit (robust 1-feature calibrated precision)
    iso = IsotonicRegression(out_of_bounds="clip").fit(ct.logit.values, ct.y.values)
    # (b) small GBM over strata
    gbm = lgb.train({"objective": "binary", "learning_rate": 0.05, "num_leaves": 15,
                     "min_data_in_leaf": 50, "feature_fraction": 0.8, "seed": SEED,
                     "verbose": -1, "num_threads": 8},
                    lgb.Dataset(ct[feats].values, label=ct.y.values), num_boost_round=200)
    phat_tune = {"iso": iso.predict(ct.logit.values), "gbm": gbm.predict(ct[feats].values)}
    phat_blind = {"iso": iso.predict(cb.logit.values), "gbm": gbm.predict(cb[feats].values)}

    # ---- sweep phi on TUNE for each predictor, pick best ---------------------------
    dep_tune = dep[dep.prot.isin(tune_prots)]
    dep_blind = dep[dep.prot.isin(blind_prots)]
    Pt0, Gt0, iat0, keep0 = contrib_matrices(parse_pred(dep_tune, ontologies, gt_tune), gt_tune, gtex_tune, ontologies)
    f_tune_dep, tau_tune_dep = best_tau(Pt0, Gt0, iat0, keep0)

    sweep = {}
    for model in ("iso", "gbm"):
        pv = phat_tune[model]
        order = np.argsort(-pv)
        n = len(ct)
        rec = {}
        for phi in PHIS:
            k = int(round(n * phi))
            sel = ct.iloc[order[:k]] if k > 0 else ct.iloc[[]]
            sub = submission(dep_tune, sel)
            Pt, Gt, iat, keep = contrib_matrices(parse_pred(sub, ontologies, gt_tune), gt_tune, gtex_tune, ontologies)
            f, tau = best_tau(Pt, Gt, iat, keep)
            rec[phi] = {"k": k, "f": f, "tau": tau, "delta": round(f - f_tune_dep, 5)}
        sweep[model] = rec
        print(f"  TUNE {model}: dep {f_tune_dep}  " +
              " ".join(f"phi{phi}:{rec[phi]['delta']:+.4f}(k{rec[phi]['k']})" for phi in PHIS), flush=True)

    # pick (model, phi) maximizing TUNE delta
    best = max(((m, phi, sweep[m][phi]["delta"], sweep[m][phi]["tau"], sweep[m][phi]["k"])
                for m in sweep for phi in PHIS if phi > 0), key=lambda z: z[2])
    bmodel, bphi, bdelta_tune, btau_tune, bk_tune = best
    # The honest deployable policy would pick phi=0 (submit no extras) whenever no augmentation arm
    # beats the pool on TUNE. Record that; a NO-GO is exactly this being True.
    tune_optimal_is_no_augmentation = bool(bdelta_tune <= 0)
    print(f"  TUNE best: model={bmodel} phi={bphi} delta={bdelta_tune:+.5f} tau={btau_tune} k={bk_tune}", flush=True)

    # ---- apply BLIND -------------------------------------------------------------
    pvb = phat_blind[bmodel]
    orderb = np.argsort(-pvb)
    nb = len(cb)
    kb = int(round(nb * bphi))
    sel_blind = cb.iloc[orderb[:kb]]
    # uniform matched-volume control: same kb, chosen by top logit uniformly (not stratified)
    uni_order = np.argsort(-cb.logit.values)
    uni_blind = cb.iloc[uni_order[:kb]]

    # BLIND arms (all row-aligned to gt_blind)
    A_Pt, A_Gt, A_iat, A_keep = contrib_matrices(parse_pred(dep_blind, ontologies, gt_blind), gt_blind, gtex_blind, ontologies)
    fA, tauA = best_tau(A_Pt, A_Gt, A_iat, A_keep)  # in-window ceiling for the deployed arm
    tp, fp, fn, _ = pp_contribs(A_Pt, A_Gt, A_iat, A_keep, tau_tune_dep)  # DEPLOYABLE deployed (TUNE tau)
    fA_dep = round(fmicro(tp, fp, fn), 5)

    sub_sel = submission(dep_blind, sel_blind)
    S_Pt, S_Gt, S_iat, S_keep = contrib_matrices(parse_pred(sub_sel, ontologies, gt_blind), gt_blind, gtex_blind, ontologies)
    fSel_own, tauSel_own = best_tau(S_Pt, S_Gt, S_iat, S_keep)
    tp, fp, fn, _ = pp_contribs(S_Pt, S_Gt, S_iat, S_keep, btau_tune)  # deployable: tau fixed from TUNE
    fSel_fixed = round(fmicro(tp, fp, fn), 5)

    sub_uni = submission(dep_blind, uni_blind)
    U_Pt, U_Gt, U_iat, U_keep = contrib_matrices(parse_pred(sub_uni, ontologies, gt_blind), gt_blind, gtex_blind, ontologies)
    fUni_own, tauUni_own = best_tau(U_Pt, U_Gt, U_iat, U_keep)
    tp, fp, fn, _ = pp_contribs(U_Pt, U_Gt, U_iat, U_keep, btau_tune)
    fUni_fixed = round(fmicro(tp, fp, fn), 5)

    # ---- cafaeval parity on the deployed BLIND anchor + selective arm --------------
    def cafa_point(sub):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / "pd"; d.mkdir()
            sub[["prot", "term", "score"]].to_csv(d / "p.tsv", sep="\t", header=False, index=False, float_format="%.6f")
            _, dfs = cafa_eval(OBO, str(d), gtf_blind, ia=IA, no_orphans=True, norm="cafa", prop="fill",
                               exclude=knf_blind, toi_file=TOI, th_step=0.01, n_cpu=8)
        b = dfs["f_micro_w"].reset_index(); r = b[b.ns == NS].iloc[0]
        return round(float(r["f_micro_w"]), 5), float(r["tau"])
    cA, ctauA = cafa_point(dep_blind)
    cSel, ctauSel = cafa_point(sub_sel)
    parity_ok = abs(cA - fA) < 2e-3 and abs(cSel - fSel_own) < 2e-3
    print(f"  BLIND parity: A mine {fA}@{tauA} cafa {cA}@{ctauA} | Sel mine {fSel_own}@{tauSel_own} cafa {cSel}@{ctauSel} ok={parity_ok}", flush=True)

    # ---- paired protein bootstrap on BLIND (rows are gt-aligned across all arms) ----
    # Deployable framing: all three arms scored at the SAME tau fixed from TUNE, per-protein weighted.
    tpA, fpA, fnA, ngtA = pp_contribs(A_Pt, A_Gt, A_iat, A_keep, tau_tune_dep)  # deployed at its TUNE tau
    tpS, fpS, fnS, ngtS = pp_contribs(S_Pt, S_Gt, S_iat, S_keep, btau_tune)
    tpU, fpU, fnU, ngtU = pp_contribs(U_Pt, U_Gt, U_iat, U_keep, btau_tune)
    keep_rows = np.where((ngtA > 0) | (ngtS > 0))[0]
    aA = np.stack([tpA, fpA, fnA], 1)[keep_rows]
    aS = np.stack([tpS, fpS, fnS], 1)[keep_rows]
    aU = np.stack([tpU, fpU, fnU], 1)[keep_rows]
    rngb = np.random.default_rng(7); B = 2000
    dSA = np.empty(B); dSU = np.empty(B)
    idx = np.arange(len(keep_rows))
    for b in range(B):
        s = rngb.choice(idx, size=len(idx), replace=True)
        fS = fmicro(aS[s, 0], aS[s, 1], aS[s, 2])
        fAA = fmicro(aA[s, 0], aA[s, 1], aA[s, 2])
        fUU = fmicro(aU[s, 0], aU[s, 1], aU[s, 2])
        dSA[b] = fS - fAA; dSU[b] = fS - fUU
    ci_sa = [round(float(np.percentile(dSA, 2.5)), 5), round(float(np.percentile(dSA, 97.5)), 5)]
    ci_su = [round(float(np.percentile(dSU, 2.5)), 5), round(float(np.percentile(dSU, 97.5)), 5)]

    # selective included-strata precision (which strata paid)
    sel_prec = round(float(sel_blind.y.mean()), 4) if len(sel_blind) else None
    uni_prec = round(float(uni_blind.y.mean()), 4) if len(uni_blind) else None
    strata_included = {}
    if len(sel_blind):
        for lo, hi in [(0, 2), (2, 4), (4, 99)]:
            m = (sel_blind.ia >= lo) & (sel_blind.ia < hi)
            strata_included[f"IA[{lo},{hi})"] = {"n": int(m.sum()),
                                                 "prec": round(float(sel_blind.y[m].mean()), 4) if m.sum() else None}

    results[cat.upper() + "_BPO"] = {
        "primary_source": "classifier_6plm_asl extras (top-50 BP logits per protein not in pool), DB-free from v227 frames",
        "tune_blind_split": {"tune_proteins": len(tune_prots), "blind_proteins": len(blind_prots), "seed": SEED},
        "precision_model_chosen": bmodel, "phi_chosen": bphi, "tune_delta": bdelta_tune,
        "tune_optimal_is_no_augmentation": tune_optimal_is_no_augmentation,
        "tau_from_tune": btau_tune,
        "n_extras_submitted_blind": int(kb), "selective_extra_precision_blind": sel_prec,
        "uniform_extra_precision_blind": uni_prec,
        "selective_included_strata": strata_included,
        "DEPLOYED_A_blind_inwindow": fA, "tau_A_inwindow": tauA,
        "DEPLOYED_A_blind_deployable": fA_dep, "tau_A_deployable": tau_tune_dep,
        "SELECTIVE_blind_own_tau": fSel_own, "tau_sel_own": tauSel_own,
        "SELECTIVE_blind_deployable_tuneTau": fSel_fixed,
        "UNIFORM_blind_own_tau": fUni_own, "tau_uni_own": tauUni_own,
        "UNIFORM_blind_deployable_tuneTau": fUni_fixed,
        "delta_selective_vs_deployed_deployable": round(fSel_fixed - fA_dep, 5),
        "delta_selective_vs_uniform_deployable": round(fSel_fixed - fUni_fixed, 5),
        "delta_selective_vs_deployed_inwindow_ceiling": round(fSel_own - fA, 5),
        "bootstrap": {"B": B, "n_proteins": int(len(keep_rows)),
                      "delta_sel_vs_dep_mean": round(float(dSA.mean()), 5), "ci95_sel_vs_dep": ci_sa,
                      "frac_pos_sel_vs_dep": round(float((dSA > 0).mean()), 4),
                      "delta_sel_vs_uni_mean": round(float(dSU.mean()), 5), "ci95_sel_vs_uni": ci_su,
                      "frac_pos_sel_vs_uni": round(float((dSU > 0).mean()), 4)},
        "cafaeval_parity": {"A_cafa": cA, "A_mine": fA, "Sel_cafa": cSel, "Sel_mine": fSel_own, "ok": bool(parity_ok)},
        "full_tune_sweep": sweep,
    }
    print(f"  RESULT {cat.upper()}: A_dep {fA_dep} | Sel(deployable) {fSel_fixed} ({fSel_fixed-fA_dep:+.5f}) | "
          f"Uni(deployable) {fUni_fixed} ({fSel_fixed-fUni_fixed:+.5f} vs uni) | "
          f"boot CI(sel-dep) {ci_sa} fracpos {(dSA>0).mean():.2f}", flush=True)
    json.dump(results, open(OUT / "selgen_main.json", "w"), indent=1)

json.dump(results, open(OUT / "selgen_main.json", "w"), indent=1)
print(f"\n[{time.time()-t0:.0f}s] SELGEN DONE", flush=True)
