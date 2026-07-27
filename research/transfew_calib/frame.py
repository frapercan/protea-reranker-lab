"""Shared true-board-frame scoring + per-protein confusion for the TransFew-calibration graft.

Frame (matches the deployed cell, repositories/.../percut_rerank/score_cafaeval.py):
  obo = lafa_t0 go-basic.obo ; ia = lafa_t0 IA.tsv
  prop=fill, norm=cafa, no_orphans=True, toi_file=terms_of_interest
  LK: exclude=None ; PK: exclude=groundtruth_PK_known.tsv (-known)
  metric = f_micro_w on biological_process, max over tau in arange(0.01,1,0.01)

This module runs under the PROTEA venv (has cafaeval + lightgbm + sklearn).
"""
import os
import numpy as np

OBO = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
IA_F = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv"
GT_DIR = "/home/frapercan/Thesis2/repositories/protea-reranker-lab/results/sparse_classifier/lafa_gt"
TOI = os.path.join(GT_DIR, "groundtruth_terms_of_interest.txt")
GT = {
    "lk": (os.path.join(GT_DIR, "groundtruth_LK.tsv"), None),
    "pk": (os.path.join(GT_DIR, "groundtruth_PK.tsv"),
           os.path.join(GT_DIR, "groundtruth_PK_known.tsv")),
}
NS_BP = "biological_process"
TH_STEP = 0.01


def score_dir(pred_dir, cat, n_cpu=8):
    """Authoritative BP f_micro_w for a prediction directory (single cafa_eval call)."""
    from cafaeval.evaluation import cafa_eval
    gt_file, known = GT[cat]
    df, dfs_best = cafa_eval(OBO, pred_dir, gt_file, ia=IA_F, no_orphans=True, norm="cafa",
                             prop="fill", exclude=known, toi_file=TOI, th_step=TH_STEP,
                             n_cpu=n_cpu, weighted_only=False)
    best = dfs_best["f_micro_w"].reset_index()
    row = best[best["ns"] == NS_BP]
    if len(row) == 0:
        return None
    row = row.iloc[0]
    return {"f_micro_w": float(row["f_micro_w"]), "tau": float(row["tau"]),
            "pr_micro_w": float(row.get("pr_micro_w", np.nan)),
            "rc_micro_w": float(row.get("rc_micro_w", np.nan)),
            "cov": float(row.get("cov", np.nan))}


def build_bp_matrices(pred_file, cat, n_cpu=8):
    """Parse obo/gt/pred exactly like cafa_eval and return per-protein BP arrays for bootstrap.

    Returns dict with, for the BP namespace:
      prot_ids : (P,) protein accessions with GT in TOI (post-exclude), order fixed
      pred     : (P, T) propagated prediction score matrix restricted to these proteins
      gt       : (P, T) propagated groundtruth (bool) restricted to these proteins
      ia       : (T,) information accretion per term index
      toi_ia   : (K,) term indices used for the weighted metric (ia>0 and in toi)
      excl     : (P, T) bool exclude mask (PK) or None
    """
    from cafaeval.parser import (obo_parser, gt_parser, pred_parser,
                                  gt_exclude_parser, update_toi)
    gt_file, known = GT[cat]
    onts = obo_parser(OBO, ("is_a", "part_of"), IA_F, False)  # no_orphans=True -> orphans=False
    onts = update_toi(onts, TOI)
    gt = gt_parser(gt_file, onts)
    gt_excl = gt_exclude_parser(known, gt, onts) if known else None
    pred = pred_parser(pred_file, onts, gt, "fill", None, n_cpu)

    ns = NS_BP
    ont = onts[ns]
    g = gt[ns]
    p = pred[ns]
    # gt[ns].ids is a dict {protein: row_index}; pred_parser builds p.matrix over the SAME gt row
    # indexing (matrix = zeros(gt.matrix.shape)). Recover accessions in matrix-row order.
    inv = [None] * len(g.ids)
    for prot, i in g.ids.items():
        inv[i] = prot
    prot_ids = np.asarray(inv, dtype=object)

    toi_ia = ont.toi_ia  # term indices with ia>0 (weighted-metric TOI)
    ia = ont.ia
    gmat = g.matrix
    pmat = p.matrix
    excl = gt_excl[ns].matrix if gt_excl is not None else None

    # Restrict to proteins with a surviving GT annotation in toi_ia (post-exclude): these are the
    # proteins that define ne / the weighted-metric denominators (see _count_proteins_in_toi).
    toi_mask = np.zeros(gmat.shape[1], dtype=bool)
    toi_mask[toi_ia] = True
    if excl is not None:
        valid_gt = (gmat != 0) & toi_mask[None, :] & (excl == 0)
    else:
        valid_gt = (gmat != 0) & toi_mask[None, :]
    keep = valid_gt.any(axis=1)
    idx = np.flatnonzero(keep)
    return {
        "prot_ids": prot_ids[idx],
        "pred": pmat[idx][:, toi_ia].astype(np.float64),
        "gt": (gmat[idx][:, toi_ia] != 0),
        "ia": ia[toi_ia].astype(np.float64),
        "excl": (excl[idx][:, toi_ia] != 0) if excl is not None else None,
    }


def per_protein_confusion(M, tau):
    """Weighted TP/FP/FN per protein at threshold tau, matching compute_confusion_matrix.

    pred solid = pred >= tau (cafaeval: solidify uses >=). For PK, excluded (protein,term) cells are
    removed from both pred and gt.  Returns (wtp, wfp, wfn) each shape (P,).
    """
    pred = M["pred"]
    gt = M["gt"]
    ia = M["ia"]
    excl = M["excl"]
    sel = pred >= tau
    truth = gt
    if excl is not None:
        keep = ~excl
        sel = sel & keep
        truth = truth & keep
    tp = sel & truth
    fp = sel & (~truth)
    fn = (~sel) & truth
    wtp = (tp * ia[None, :]).sum(axis=1)
    wfp = (fp * ia[None, :]).sum(axis=1)
    wfn = (fn * ia[None, :]).sum(axis=1)
    return wtp, wfp, wfn


def micro_f_from_confusion(wtp, wfp, wfn):
    TP = wtp.sum(); FP = wfp.sum(); FN = wfn.sum()
    if TP <= 0:
        return 0.0
    pr = TP / (TP + FP)
    rc = TP / (TP + FN)
    return float(2 * pr * rc / (pr + rc)) if (pr + rc) > 0 else 0.0


def best_tau_micro_f(M, taus=None):
    """Sweep taus, return (best_f, best_tau) using the per-protein confusion (self-check vs cafaeval)."""
    if taus is None:
        taus = np.arange(TH_STEP, 1.0, TH_STEP)
    best_f, best_t = -1.0, None
    # precompute weighted contribution efficiently: for each tau recompute (vectorized but simple)
    for t in taus:
        wtp, wfp, wfn = per_protein_confusion(M, t)
        f = micro_f_from_confusion(wtp, wfp, wfn)
        if f > best_f:
            best_f, best_t = f, float(t)
    return best_f, best_t
