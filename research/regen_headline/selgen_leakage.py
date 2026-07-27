"""LEAKAGE PROBE for the selective-generation gain (fast, NO cafaeval).

The main run banks +0.015 (LK) / +0.032 (PK) deployable, and the mechanism is a GBM precision model
that selects extras at 37-48% blind precision (vs 17% uniform, 6-10% aggregate). The GBM beats an
isotonic-on-logit model by ~0.04 f_micro_w; that gap is entirely from the term-level features
(IA, term-freq, DAG-depth) which jointly IDENTIFY the GO term. Because the tune/blind split is
PROTEIN-level within the SAME eval window (v227->v230), the GBM can learn which terms are being gained
THIS window from tune proteins and boost them on blind proteins -- a within-window term-popularity
memorization that a true train-on-past split (the mandated temporal gate) would forbid.

DECISIVE TEST: does the GBM's precision advantage survive on UNSEEN terms?
  * PROTEIN-split  (main): fit on tune proteins, evaluate on blind proteins  -> terms are SHARED.
  * TERM-split           : fit on term-set A (all proteins), evaluate on term-set B (disjoint)
                           -> the model has NEVER seen term B's label rate. Term memorization is
                           impossible; only generalizable per-candidate precision can transfer.
If GBM precision collapses toward the isotonic/uniform level on the TERM-split, the gain is term
memorization = the leakage the temporal gate guards against, and the deployable verdict is NO-GO.
We also target-encode each term by its TUNE precision to quantify the memorization directly.
"""
import json
from pathlib import Path
import numpy as np
from sklearn.isotonic import IsotonicRegression
import lightgbm as lgb

OUT = Path("/home/frapercan/Thesis2/storage/regen_headline")
FEATS = ["logit", "ia", "tf", "depth", "taxden"]
PHI = {"lk": 0.05, "pk": 0.1}
SEED = 13


def fit_gbm(Xtr, ytr):
    return lgb.train({"objective": "binary", "learning_rate": 0.05, "num_leaves": 15,
                      "min_data_in_leaf": 50, "feature_fraction": 0.8, "seed": SEED,
                      "verbose": -1, "num_threads": 8},
                     lgb.Dataset(Xtr, label=ytr), num_boost_round=200)


def prec_at(scores, y, phi):
    k = max(1, int(round(len(scores) * phi)))
    idx = np.argsort(-scores)[:k]
    return round(float(y[idx].mean()), 4), int(k)


report = {}
for cell in ("lk", "pk"):
    R = np.load(OUT / f"selgen_cand_{cell}.npy")
    P = R["p"]; G = R["g"]; y = R["y"].astype(int)
    X = np.stack([R[f].astype(float) for f in FEATS], 1)
    logit = R["logit"].astype(float)
    phi = PHI[cell]
    rng = np.random.default_rng(SEED)

    # -- PROTEIN split (reproduces the main mechanism) --
    prots = np.array(sorted(set(P.tolist())))
    pp = rng.permutation(prots); tune_p = set(pp[:len(pp) // 2].tolist())
    mtr = np.array([p in tune_p for p in P]); mbl = ~mtr
    gbm = fit_gbm(X[mtr], y[mtr])
    iso = IsotonicRegression(out_of_bounds="clip").fit(logit[mtr], y[mtr])
    g_ps, _ = prec_at(gbm.predict(X[mbl]), y[mbl], phi)
    i_ps, _ = prec_at(iso.predict(logit[mbl]), y[mbl], phi)
    l_ps, kk = prec_at(logit[mbl], y[mbl], phi)  # raw logit ranking

    # -- TERM split (disjoint term vocab; memorization impossible) --
    terms = np.array(sorted(set(G.tolist())))
    tp = rng.permutation(terms); termA = set(tp[:len(tp) // 2].tolist())
    mA = np.array([g in termA for g in G]); mB = ~mA
    gbmT = fit_gbm(X[mA], y[mA])
    isoT = IsotonicRegression(out_of_bounds="clip").fit(logit[mA], y[mA])
    g_ts, _ = prec_at(gbmT.predict(X[mB]), y[mB], phi)
    i_ts, _ = prec_at(isoT.predict(logit[mB]), y[mB], phi)

    # -- direct term-memorization: target-encode each term by its TUNE-protein precision --
    tune_prec = {}
    cnt = {}
    for gg, yy, mm in zip(G, y, mtr):
        if mm:
            tune_prec[gg] = tune_prec.get(gg, 0) + yy; cnt[gg] = cnt.get(gg, 0) + 1
    te = np.array([(tune_prec.get(g, 0) / cnt[g]) if cnt.get(g, 0) else -1.0 for g in G])
    te_ps, _ = prec_at(te[mbl], y[mbl], phi)  # blind proteins, ranked by term's tune precision alone

    # feature importances (gain): how much does the GBM lean on term-identity features?
    imp = dict(zip(FEATS, gbm.feature_importance(importance_type="gain").tolist()))
    tot = sum(imp.values()) or 1
    imp_frac = {k: round(v / tot, 3) for k, v in imp.items()}

    report[cell.upper() + "_BPO"] = {
        "phi": phi, "aggregate_precision": round(float(y.mean()), 4),
        "PROTEIN_split_seen_terms": {"gbm": g_ps, "iso_logit": i_ps, "raw_logit": l_ps,
                                     "term_target_encoding_only": te_ps, "n_selected": kk},
        "TERM_split_unseen_terms": {"gbm": g_ts, "iso_logit": i_ts},
        "gbm_minus_iso_PROTEIN": round(g_ps - i_ps, 4),
        "gbm_minus_iso_TERM": round(g_ts - i_ts, 4),
        "gbm_gain_feature_importance_frac": imp_frac,
        "term_features_frac": round(imp_frac["ia"] + imp_frac["tf"] + imp_frac["depth"], 3),
    }
    print(f"{cell.upper()}: PROTEIN-split gbm {g_ps} iso {i_ps} (te-only {te_ps}) | "
          f"TERM-split gbm {g_ts} iso {i_ts} | gbm-iso protein {g_ps-i_ps:+.3f} term {g_ts-i_ts:+.3f} | "
          f"term-feat-imp {report[cell.upper()+'_BPO']['term_features_frac']}", flush=True)

json.dump(report, open(OUT / "selgen_leakage.json", "w"), indent=1)
print("LEAKAGE PROBE DONE", flush=True)
