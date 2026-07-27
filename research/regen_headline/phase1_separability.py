"""PHASE 1 of the transition-separability probe.

For LK-BPO and PK-BPO: measure per-protein ranking quality (per-protein AUC) of
  (a) the DEPLOYED reranker score  [incumbent]
  (b) first-order association_cross alone
  (c) a LEARNED GBDT over ONLY known-conditioned features, grouped-CV OOF
  (d) a LEARNED logistic over the same, grouped-CV OOF
  (e) a higher-order PPMI diffusion transition score s(t|K_p) alone
  (f) the GBDT with the transition score added

Everything frozen, on disk. Grouped CV by protein (no protein in two folds).
"""
import json, collections, time
from pathlib import Path
import numpy as np, pandas as pd
import pyarrow.parquet as pq
import scipy.sparse as sp
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

t0 = time.time()
ROOT = Path("/home/frapercan/Thesis2")
EVAL = ROOT / "repositories/protea-reranker-lab/results/sparse_classifier/percut_rerank/eval.parquet"
PREDDIR = ROOT / "repositories/protea-reranker-lab/results/sparse_classifier/percut_rerank/predictions"
GTDIR = ROOT / "repositories/protea-reranker-lab/results/sparse_classifier/lafa_gt"
FROZ = ROOT / "storage/protea-frozen-v227-2025-09-04"
OUT = ROOT / "storage/regen_headline"

KNOWN_FEATS = ["association_total", "association_cross", "association_present",
               "anc2vec_query_known_cos", "anc2vec_query_known_maxcos", "anc2vec_query_known_count",
               "self_prior_score", "lineage_is_ancestor_of_known", "lineage_is_descendant_of_known",
               "lineage_ancestor_of_count", "lineage_descendant_of_count", "go_term_frequency"]

print(f"[{time.time()-t0:.0f}s] loading metadata + reference annotations", flush=True)
meta = pq.read_table(FROZ / "go_term_metadata.parquet").to_pandas()
id2go = dict(zip(meta.go_term_id, meta.go_id))
id2asp = dict(zip(meta.go_term_id, meta.aspect))
ref = pq.read_table(FROZ / "reference_annotations.parquet", columns=["accession", "go_term_id"]).to_pandas()
ref["go"] = ref.go_term_id.map(id2go)
ref["asp"] = ref.go_term_id.map(id2asp)
ref = ref.dropna(subset=["go"])

# ------- co-occurrence graph over reference proteins (built once, sliced per cell) -------
prot_codes, prot_uniq = pd.factorize(ref.accession, sort=False)
term_codes, term_uniq = pd.factorize(ref.go, sort=False)
n_prot, n_term = len(prot_uniq), len(term_uniq)
A = sp.csr_matrix((np.ones(len(ref), np.float32), (prot_codes, term_codes)), shape=(n_prot, n_term))
A.data[:] = 1.0  # binary
termpos = {g: i for i, g in enumerate(term_uniq)}
term_marg = np.asarray(A.sum(0)).ravel()  # doc frequency per term
Ntot = n_prot
print(f"[{time.time()-t0:.0f}s] cooc matrix {A.shape}, nnz {A.nnz:,}", flush=True)

# PK known set (BP=P) from the canonical known file; LK known = reference F/C
pkk = pd.read_csv(GTDIR / "groundtruth_PK_known.tsv", sep="\t")
pkk_asp = pkk.copy()
pk_known_bp = pkk_asp[pkk_asp.aspect == "P"].groupby("EntryID").term.apply(set).to_dict()


def transition_scores(cell, cand_prot, cand_go, known_of):
    """PPMI first-order and 2-hop diffusion s(t|K_p). Returns dict term->none; computed per row.

    cand_prot, cand_go: arrays aligned with candidate rows. known_of: prot -> set(GO seeds).
    """
    cand_terms = sorted(set(cand_go))
    Tidx = np.array([termpos.get(g, -1) for g in cand_terms])
    valid_T = Tidx >= 0
    cand_terms = [g for g, v in zip(cand_terms, valid_T) if v]
    Tidx = Tidx[valid_T]
    tpos = {g: j for j, g in enumerate(cand_terms)}  # local candidate index
    # seed universe
    seeds = set()
    for p in cand_prot:
        seeds |= known_of.get(p, set())
    seeds = sorted(seeds)
    Sidx = np.array([termpos.get(g, -1) for g in seeds])
    vS = Sidx >= 0
    seeds = [g for g, v in zip(seeds, vS) if v]
    Sidx = Sidx[vS]
    spos = {g: j for j, g in enumerate(seeds)}
    print(f"  [{cell}] seeds {len(seeds)} cand-terms {len(cand_terms)}", flush=True)
    # co-occurrence counts: seed x cand  and  cand x cand
    As = A[:, Sidx]          # n_prot x |S|
    At = A[:, Tidx]          # n_prot x |T|
    C_st = (As.T @ At).toarray().astype(np.float64)      # |S| x |T|
    C_tt = (At.T @ At).toarray().astype(np.float64)      # |T| x |T|
    mS = term_marg[Sidx]; mT = term_marg[Tidx]
    # PPMI(seed, cand)
    with np.errstate(divide="ignore", invalid="ignore"):
        Pst = C_st / Ntot
        pmi = np.log((Pst) / ((mS[:, None] / Ntot) * (mT[None, :] / Ntot) + 1e-12) + 1e-12)
    ppmi_st = np.maximum(0.0, pmi)
    # cand-cand transition (row-normalized cooc, for the 2nd hop)
    Wtt = C_tt.copy()
    np.fill_diagonal(Wtt, 0.0)
    rs = Wtt.sum(1, keepdims=True); rs[rs == 0] = 1.0
    Wtt_n = Wtt / rs
    # per protein, aggregate seed->cand
    s1 = np.zeros(len(cand_prot))   # first-order PPMI diffusion
    s2 = np.zeros(len(cand_prot))   # 2-hop diffusion
    # precompute per-protein seed local indices
    protseed = {p: [spos[g] for g in known_of.get(p, set()) if g in spos] for p in set(cand_prot)}
    # per-protein cand-vector of first order
    cache1 = {}
    for i, (p, g) in enumerate(zip(cand_prot, cand_go)):
        if g not in tpos:
            continue
        j = tpos[g]
        if p not in cache1:
            si = protseed.get(p, [])
            if si:
                vec = ppmi_st[si, :].sum(0)   # |T|
            else:
                vec = np.zeros(len(cand_terms))
            cache1[p] = (vec, vec @ Wtt_n)    # (first order over cand, its 2-hop spread)
        v1, v2 = cache1[p]
        s1[i] = v1[j]; s2[i] = v2[j]
    return s1, s2


def per_protein_auc(prot, y, score):
    aucs = []
    df = pd.DataFrame({"p": prot, "y": y, "s": score})
    for p, g in df.groupby("p"):
        if g.y.nunique() == 2:
            aucs.append(roc_auc_score(g.y, g.s))
    return float(np.mean(aucs)), len(aucs)


def grouped_oof_gbdt(X, y, groups):
    oof = np.zeros(len(y))
    gkf = GroupKFold(n_splits=5)
    for tr, va in gkf.split(X, y, groups):
        d = lgb.Dataset(X[tr], label=y[tr])
        params = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=50,
                      feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=5, seed=42,
                      verbose=-1, num_threads=8, metric="average_precision",
                      scale_pos_weight=(y[tr] == 0).sum() / max(1, (y[tr] == 1).sum()))
        m = lgb.train(params, d, num_boost_round=400)
        oof[va] = m.predict(X[va])
    return oof


def grouped_oof_logit(X, y, groups):
    oof = np.zeros(len(y))
    gkf = GroupKFold(n_splits=5)
    Xi = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    for tr, va in gkf.split(Xi, y, groups):
        sc = StandardScaler().fit(Xi[tr])
        m = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)
        m.fit(sc.transform(Xi[tr]), y[tr])
        oof[va] = m.predict_proba(sc.transform(Xi[va]))[:, 1]
    return oof


results = {}
for cell, cat in [("LK-BPO", "lk"), ("PK-BPO", "pk")]:
    print(f"\n[{time.time()-t0:.0f}s] ===== {cell} =====", flush=True)
    cols = ["protein_accession", "go_term_id", "label", "category", "aspect"] + KNOWN_FEATS
    tb = pq.read_table(EVAL, columns=cols).to_pandas()
    d = tb[(tb.category == cat) & (tb.aspect == "bpo")].reset_index(drop=True)
    # deployed reranker score
    pred = pd.read_csv(PREDDIR / cat / f"{cat}.tsv", sep="\t", header=None,
                       names=["protein_accession", "go_term_id", "reranker_score"])
    d = d.merge(pred, on=["protein_accession", "go_term_id"], how="left")
    miss = d.reranker_score.isna().mean()
    d.reranker_score = d.reranker_score.fillna(0.0)
    y = d.label.values.astype(int)
    prot = d.protein_accession.values
    print(f"  rows {len(d):,} prot {d.protein_accession.nunique()} true% {100*y.mean():.2f} "
          f"reranker-join-miss {100*miss:.2f}%", flush=True)

    # known set per protein
    if cat == "lk":
        rr = ref[ref.accession.isin(set(prot)) & ref.asp.isin(["F", "C"])]
        known_of = rr.groupby("accession").go.apply(set).to_dict()
    else:
        known_of = {p: pk_known_bp.get(p, set()) for p in set(prot)}
    kc = np.array([len(known_of.get(p, set())) for p in prot])
    print(f"  mean known-set size {kc.mean():.1f} (median {np.median(kc):.0f})", flush=True)

    s1, s2 = transition_scores(cell, prot, d.go_term_id.values, known_of)
    d["transition_ppmi_1hop"] = s1
    d["transition_diffusion_2hop"] = s2

    X = d[KNOWN_FEATS].values.astype(np.float32)
    Xt = np.hstack([X, s1[:, None], s2[:, None]]).astype(np.float32)
    oof_gbdt = grouped_oof_gbdt(X, y, prot)
    oof_gbdt_t = grouped_oof_gbdt(Xt, y, prot)
    oof_logit = grouped_oof_logit(X, y, prot)

    scores = {
        "deployed_reranker": d.reranker_score.values,
        "association_cross_1storder": d.association_cross.values,
        "gbdt_known_conditioned": oof_gbdt,
        "logit_known_conditioned": oof_logit,
        "transition_ppmi_1hop_alone": s1,
        "transition_diffusion_2hop_alone": s2,
        "gbdt_known_plus_transition": oof_gbdt_t,
    }
    cell_res = {"rows": int(len(d)), "proteins": int(d.protein_accession.nunique()),
                "true_pct": round(100 * float(y.mean()), 3),
                "mean_known_set": round(float(kc.mean()), 2), "per_protein_auc": {}}
    for name, sc in scores.items():
        auc, n = per_protein_auc(prot, y, np.nan_to_num(sc))
        cell_res["per_protein_auc"][name] = {"auc": round(auc, 4), "n_prot": n}
        print(f"    {name:34s} per-prot-AUC {auc:.4f} (n={n})", flush=True)

    # stratify by known-count bucket for the best learned vs incumbent vs 1st order
    buckets = [(0, 5), (5, 15), (15, 40), (40, 10**9)]
    strat = {}
    for lo, hi in buckets:
        m = (kc >= lo) & (kc < hi)
        if m.sum() < 50:
            continue
        row = {}
        for name in ["deployed_reranker", "association_cross_1storder", "gbdt_known_conditioned",
                     "gbdt_known_plus_transition", "transition_diffusion_2hop_alone"]:
            a, n = per_protein_auc(prot[m], y[m], np.nan_to_num(scores[name][m]))
            row[name] = round(a, 4)
        row["n_prot"] = int(pd.Series(prot[m]).nunique())
        strat[f"known_{lo}_{hi if hi < 10**8 else 'inf'}"] = row
    cell_res["stratified_by_known_count"] = strat
    results[cell] = cell_res
    # persist the OOF + transition scores for Phase 2
    d[["protein_accession", "go_term_id", "label", "reranker_score",
       "transition_ppmi_1hop", "transition_diffusion_2hop"]].assign(
        gbdt_known=oof_gbdt, gbdt_known_transition=oof_gbdt_t, logit_known=oof_logit
    ).to_parquet(OUT / f"phase1_scores_{cat}.parquet", index=False)

json.dump(results, open(OUT / "phase1_separability.json", "w"), indent=2)
print(f"\n[{time.time()-t0:.0f}s] wrote phase1_separability.json", flush=True)
print(json.dumps(results, indent=2))
