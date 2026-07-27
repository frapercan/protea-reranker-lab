"""Treatment-2: LEAN + IA + sp_present + literal DAG-PROPAGATED clf_p / sp_p / knn_p.

This implements the brief's LITERAL interpretation of `_p` = propagate a per-(protein,
term) score UP the GO is_a/part_of ancestor closure within the protein's candidate set:

    score_p(protein, t) = max over { s in candidates(protein) : t in ancestors(s) or s==t }
                          of base_score(protein, s)

i.e. a term's propagated score = max of its own score and the scores of its
DESCENDANTS that are present in that protein's candidate set. (max-over-descendants;
documented assumption, mirrors how cafaeval prop=fill pushes leaf evidence up.)

  knn_p  propagates vote_count        (the knn vote signal)
  clf_p  propagates classifier_score
  sp_p   propagates self_prior_score

Memory plan: process ONE category at a time. Pass A streams the category's
(protein, term, vote_count, classifier_score, self_prior_score) into per-protein
dicts (small: candidates/protein ~ hundreds). Compute the 3 propagated scores per
(protein,term) into a lookup dict. Pass B re-streams to build the full feature matrix
(LEAN + IA + sp_present + knn_p/clf_p/sp_p), train, eval. Frees between categories.
Aborts if free RAM < 5G.
"""
from __future__ import annotations

import gc
import json
import math
import os
import sys
from collections import defaultdict

import lightgbm as lgb
import numpy as np
import pyarrow.parquet as pq

D = "/home/frapercan/Thesis2/storage/fullgo_models/lean_ia_p_experiment"
DATA = f"{D}/data"
IA_TSV = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv"
OBO = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
CATS = ("nk", "lk", "pk")

LEAN = [
    "distance", "identity_nw", "similarity_nw", "alignment_score_nw", "gaps_pct_nw",
    "alignment_length_nw", "identity_sw", "similarity_sw", "alignment_score_sw",
    "gaps_pct_sw", "alignment_length_sw", "length_query", "length_ref",
    "taxonomic_distance", "taxonomic_common_ancestors", "vote_count", "k_position",
    "go_term_frequency", "ref_annotation_density", "neighbor_distance_std",
    "neighbor_vote_fraction", "neighbor_min_distance", "neighbor_mean_distance",
    "knn_present", "classifier_score", "classifier_present", "self_prior_score",
    "association_total", "association_cross", "association_present",
]
DERIVED = ["IA", "sp_present", "knn_p", "clf_p", "sp_p"]
FEATS = LEAN + DERIVED

PARAMS = {
    "objective": "binary", "metric": "auc", "learning_rate": 0.05,
    "num_leaves": 63, "min_data_in_leaf": 100, "feature_fraction": 0.9,
    "bagging_fraction": 0.9, "bagging_freq": 1, "verbosity": -1, "num_threads": 8,
    "seed": 42,
}
NUM_BOOST = 1500
EARLY = 50


def free_g():
    with open("/proc/meminfo") as fh:
        for ln in fh:
            if ln.startswith("MemAvailable"):
                return int(ln.split()[1]) / 1024 / 1024
    return 99.0


def load_ia():
    ia = {}
    for line in open(IA_TSV):
        p = line.rstrip("\n").split("\t")
        if len(p) >= 2:
            try:
                ia[p[0]] = float(p[1])
            except ValueError:
                pass
    return ia


def parents_map():
    par = defaultdict(set)
    cur = None
    for line in open(OBO):
        line = line.strip()
        if line == "[Term]":
            cur = None
        elif line.startswith("id: GO:"):
            cur = line[4:]
        elif line.startswith("is_a:") and cur:
            par[cur].add(line.split()[1])
        elif line.startswith("relationship: part_of") and cur:
            p = line.split()
            if len(p) >= 3:
                par[cur].add(p[2])
    return par


def ancestors(t, par, cache):
    if t in cache:
        return cache[t]
    o = set()
    st = list(par.get(t, ()))
    while st:
        a = st.pop()
        if a in o:
            continue
        o.add(a)
        st.extend(par.get(a, ()))
    cache[t] = o
    return o


def build_prop_lookup(path, cat, par, cache):
    """Return dict (protein,term) -> (knn_p, clf_p, sp_p) by max-over-descendants."""
    pf = pq.ParquetFile(path)
    # per protein: term -> [vote, clf, sp]
    by = defaultdict(dict)
    cols = ["protein_accession", "go_term_id", "vote_count",
            "classifier_score", "self_prior_score", "category"]
    for b in pf.iter_batches(batch_size=1_000_000, columns=cols):
        d = b.to_pydict()
        c = np.asarray([str(x) for x in d["category"]])
        idx = np.nonzero(c == cat)[0]
        prot = d["protein_accession"]; term = d["go_term_id"]
        vc = d["vote_count"]; cl = d["classifier_score"]; sp = d["self_prior_score"]
        for k in idx:
            p = prot[k]; t = term[k]
            v = float(vc[k] or 0.0); cs = float(cl[k] or 0.0); ss = float(sp[k] or 0.0)
            dd = by[p]
            cur = dd.get(t)
            if cur is None:
                dd[t] = [v, cs, ss]
            else:  # collapse duplicate (prot,term) rows by max, matching seal
                if v > cur[0]: cur[0] = v
                if cs > cur[1]: cur[1] = cs
                if ss > cur[2]: cur[2] = ss
        del d
    # propagate: each source term s pushes its score up to its ancestors present in protein
    out = {}
    for p, dd in by.items():
        cand = set(dd)
        # init each candidate with its own score
        acc = {t: list(sc) for t, sc in dd.items()}
        for s, sc in dd.items():
            for a in ancestors(s, par, cache):
                if a in cand:
                    aa = acc[a]
                    if sc[0] > aa[0]: aa[0] = sc[0]
                    if sc[1] > aa[1]: aa[1] = sc[1]
                    if sc[2] > aa[2]: aa[2] = sc[2]
        for t, sc in acc.items():
            out[(p, t)] = (sc[0], sc[1], sc[2])
    by.clear()
    return out


def build_cat(path, cat, ia, prop):
    pf = pq.ParquetFile(path)
    read_cols = list(dict.fromkeys(
        LEAN + ["protein_accession", "go_term_id", "self_prior_score",
                "category", "label"]))
    xs, ys = [], []
    for b in pf.iter_batches(batch_size=1_000_000, columns=read_cols):
        d = b.to_pydict()
        c = np.asarray([str(x) for x in d["category"]])
        idx = np.nonzero(c == cat)[0]
        if idx.size == 0:
            continue
        n = idx.size
        X = np.empty((n, len(FEATS)), dtype=np.float32)
        prot = d["protein_accession"]; term = d["go_term_id"]
        for j, f in enumerate(FEATS):
            if f == "IA":
                X[:, j] = np.asarray([ia.get(term[k], 0.0) for k in idx], dtype=np.float32)
            elif f == "sp_present":
                sp = np.asarray(d["self_prior_score"], dtype=np.float32)[idx]
                X[:, j] = (sp > 0).astype(np.float32)
            elif f in ("knn_p", "clf_p", "sp_p"):
                pos = {"knn_p": 0, "clf_p": 1, "sp_p": 2}[f]
                X[:, j] = np.asarray(
                    [prop.get((prot[k], term[k]), (0.0, 0.0, 0.0))[pos] for k in idx],
                    dtype=np.float32)
            else:
                arr = np.asarray(d[f])
                if arr.dtype == object or arr.dtype == bool:
                    X[:, j] = np.asarray(
                        [float(v) if v is not None else np.nan for v in arr[idx]],
                        dtype=np.float32)
                else:
                    X[:, j] = arr[idx].astype(np.float32)
        xs.append(X)
        ys.append(np.asarray(d["label"], dtype=np.int8)[idx])
        del d
    X = np.concatenate(xs); del xs; gc.collect()
    y = np.concatenate(ys)
    return X, y


def main():
    outdir = f"{D}/lean_ia_prop"
    os.makedirs(outdir, exist_ok=True)
    ia = load_ia()
    par = parents_map(); cache = {}
    print(f"features={len(FEATS)} IA={len(ia)} par={len(par)}", flush=True)
    summary = {"variant": "lean_ia_prop", "features": FEATS, "boosters": {}}
    for cat in CATS:
        if free_g() < 5.0:
            print(f"ABORT free {free_g():.1f}G < 5G before {cat}", flush=True)
            break
        print(f"[{cat}] prop lookup (train)... free={free_g():.1f}G", flush=True)
        prop_tr = build_prop_lookup(f"{DATA}/train.parquet", cat, par, cache)
        print(f"[{cat}] prop_tr pairs={len(prop_tr)} free={free_g():.1f}G", flush=True)
        Xtr, ytr = build_cat(f"{DATA}/train.parquet", cat, ia, prop_tr)
        prop_tr.clear(); del prop_tr; gc.collect()
        print(f"[{cat}] train {Xtr.shape} pos={int(ytr.sum())} free={free_g():.1f}G", flush=True)
        prop_ev = build_prop_lookup(f"{DATA}/eval.parquet", cat, par, cache)
        Xev, yev = build_cat(f"{DATA}/eval.parquet", cat, ia, prop_ev)
        prop_ev.clear(); del prop_ev; gc.collect()
        print(f"[{cat}] eval  {Xev.shape} pos={int(yev.sum())} free={free_g():.1f}G", flush=True)
        dtr = lgb.Dataset(Xtr, label=ytr, free_raw_data=True)
        dev = lgb.Dataset(Xev, label=yev, reference=dtr, free_raw_data=True)
        evals = {}
        bst = lgb.train(
            PARAMS, dtr, num_boost_round=NUM_BOOST, valid_sets=[dev],
            valid_names=["eval"],
            callbacks=[lgb.early_stopping(EARLY, verbose=False),
                       lgb.record_evaluation(evals),
                       lgb.log_evaluation(period=100)],
        )
        auc = float(bst.best_score["eval"]["auc"])
        bst.save_model(f"{outdir}/ensemble_gbm_{cat.upper()}.txt")
        gains = bst.feature_importance(importance_type="gain")
        imp = sorted(zip(FEATS, gains.tolist()), key=lambda x: -x[1])
        order = [x[0] for x in imp]
        summary["boosters"][cat] = {
            "train_rows": int(Xtr.shape[0]), "pos": int(ytr.sum()),
            "best_iter": int(bst.best_iteration), "eval_auc": auc,
            "top_importance_gain": [[f, round(g, 1)] for f, g in imp[:14]],
            "derived_importance": {
                f: {"gain": round(float(dict(imp)[f]), 1),
                    "rank": order.index(f) + 1} for f in DERIVED},
        }
        print(f"[{cat}] AUC={auc:.4f} best_iter={bst.best_iteration}", flush=True)
        for f in DERIVED:
            print(f"    {f} gain={dict(imp)[f]:.1f} rank={order.index(f)+1}/{len(FEATS)}", flush=True)
        del Xtr, ytr, Xev, yev, dtr, dev, bst
        gc.collect()
    with open(f"{outdir}/summary.json", "w") as w:
        json.dump(summary, w, indent=2)
    print(f"WROTE {outdir}/summary.json", flush=True)


if __name__ == "__main__":
    main()
