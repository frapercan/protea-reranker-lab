"""Apply a variant's boosters to eval.parquet and score with cafaeval f_micro_w,
mirroring native_boosters_v5/validation_score (no-TOI primary recipe). Emits
per-category pred/gt TSVs and a result json. One variant per invocation.

Usage: apply_and_score.py <variant>   where variant in {lean, lean_ia, lean_ia_prop}
"""
from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pyarrow.parquet as pq
from cafaeval.evaluation import cafa_eval

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
FEATS = {
    "lean": LEAN,
    "lean_ia": LEAN + ["IA", "sp_present"],
    "lean_ia_prop": LEAN + ["IA", "sp_present", "knn_p", "clf_p", "sp_p"],
}


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
    par = defaultdict(set); cur = None
    for line in open(OBO):
        line = line.strip()
        if line == "[Term]": cur = None
        elif line.startswith("id: GO:"): cur = line[4:]
        elif line.startswith("is_a:") and cur: par[cur].add(line.split()[1])
        elif line.startswith("relationship: part_of") and cur:
            p = line.split()
            if len(p) >= 3: par[cur].add(p[2])
    return par


def ancestors(t, par, cache):
    if t in cache: return cache[t]
    o = set(); st = list(par.get(t, ()))
    while st:
        a = st.pop()
        if a in o: continue
        o.add(a); st.extend(par.get(a, ()))
    cache[t] = o; return o


def build_prop_lookup(cat, par, cache):
    pf = pq.ParquetFile(f"{DATA}/eval.parquet")
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
            dd = by[p]; cur = dd.get(t)
            if cur is None: dd[t] = [v, cs, ss]
            else:
                if v > cur[0]: cur[0] = v
                if cs > cur[1]: cur[1] = cs
                if ss > cur[2]: cur[2] = ss
    out = {}
    for p, dd in by.items():
        cand = set(dd); acc = {t: list(sc) for t, sc in dd.items()}
        for s, sc in dd.items():
            for a in ancestors(s, par, cache):
                if a in cand:
                    aa = acc[a]
                    if sc[0] > aa[0]: aa[0] = sc[0]
                    if sc[1] > aa[1]: aa[1] = sc[1]
                    if sc[2] > aa[2]: aa[2] = sc[2]
        for t, sc in acc.items():
            out[(p, t)] = (sc[0], sc[1], sc[2])
    return out


def main():
    variant = sys.argv[1]
    feats = FEATS[variant]
    vdir = f"{D}/{variant}"
    sdir = f"{vdir}/validation_score"
    os.makedirs(sdir, exist_ok=True)
    ia = load_ia()
    need_prop = variant == "lean_ia_prop"
    par = parents_map() if need_prop else None
    cache = {}
    boosters = {c: lgb.Booster(model_file=f"{vdir}/ensemble_gbm_{c.upper()}.txt") for c in CATS}
    for c in CATS:
        assert boosters[c].num_feature() == len(feats), (c, boosters[c].num_feature(), len(feats))

    prop = {c: build_prop_lookup(c, par, cache) for c in CATS} if need_prop else None

    pf = pq.ParquetFile(f"{DATA}/eval.parquet")
    pred = {c: {} for c in CATS}; gt = {c: set() for c in CATS}
    read_cols = list(dict.fromkeys(
        LEAN + ["protein_accession", "go_term_id", "self_prior_score", "label", "category"]))
    n = 0
    for b in pf.iter_batches(batch_size=500_000, columns=read_cols):
        d = b.to_pydict()
        cat = np.asarray([str(x) for x in d["category"]])
        prot = d["protein_accession"]; term = d["go_term_id"]
        lab = np.asarray(d["label"], dtype=np.int8)
        X = np.empty((len(cat), len(feats)), dtype=np.float32)
        for j, f in enumerate(feats):
            if f == "IA":
                X[:, j] = np.asarray([ia.get(t, 0.0) for t in term], dtype=np.float32)
            elif f == "sp_present":
                sp = np.asarray(d["self_prior_score"], dtype=np.float32)
                X[:, j] = (sp > 0).astype(np.float32)
            elif f in ("knn_p", "clf_p", "sp_p"):
                pos = {"knn_p": 0, "clf_p": 1, "sp_p": 2}[f]
                # filled per-row below per category (prop is per-cat); default 0
                X[:, j] = 0.0
            else:
                arr = np.asarray(d[f])
                if arr.dtype == object or arr.dtype == bool:
                    X[:, j] = np.asarray([float(v) if v is not None else np.nan for v in arr],
                                         dtype=np.float32)
                else:
                    X[:, j] = arr.astype(np.float32)
        for c in CATS:
            m = cat == c
            if not m.any():
                continue
            idx = np.nonzero(m)[0]
            Xc = X[idx]
            if need_prop:
                pc = prop[c]
                for jj, f in enumerate(feats):
                    if f in ("knn_p", "clf_p", "sp_p"):
                        pos = {"knn_p": 0, "clf_p": 1, "sp_p": 2}[f]
                        Xc[:, jj] = np.asarray(
                            [pc.get((prot[k], term[k]), (0.0, 0.0, 0.0))[pos] for k in idx],
                            dtype=np.float32)
            scores = boosters[c].predict(Xc, num_threads=4)
            pcd = pred[c]; gcd = gt[c]
            for k, s in zip(idx.tolist(), scores.tolist()):
                key = (prot[k], term[k])
                if key not in pcd or s > pcd[key]:
                    pcd[key] = s
                if lab[k] == 1:
                    gcd.add(key)
        n += len(cat)
        print(f"  ...{n} rows", flush=True)
        del d, X
    for c in CATS:
        with open(f"{sdir}/pred_{c}.tsv", "w") as w:
            for (p, t), s in pred[c].items():
                w.write(f"{p}\t{t}\t{s:.6f}\n")
        with open(f"{sdir}/gt_{c}.tsv", "w") as w:
            for (p, t) in sorted(gt[c]):
                w.write(f"{p}\t{t}\n")
        print(f"{c}: pred={len(pred[c])} gt={len(gt[c])}", flush=True)

    # score with cafaeval (no-TOI primary recipe)
    res = {}; perns = {}
    for c in CATS:
        pdir = Path(sdir) / f"_pred_{c}"; pdir.mkdir(exist_ok=True)
        link = pdir / "m.tsv"
        if link.exists() or link.is_symlink(): link.unlink()
        link.symlink_to(f"{sdir}/pred_{c}.tsv")
        df, _ = cafa_eval(OBO, str(pdir), f"{sdir}/gt_{c}.tsv", ia=IA_TSV, prop="fill",
                          norm="cafa", no_orphans=True, max_terms=None, th_step=0.01, n_cpu=1)
        sub = df.reset_index()
        col = "f_micro_w" if "f_micro_w" in sub.columns else "f_w"
        pn = sub.groupby("ns")[col].max()
        res[c] = float(pn.mean()); perns[c] = {k: round(float(v), 4) for k, v in pn.items()}
        print(f"{c.upper()}: f_micro_w={res[c]:.4f} per_ns={perns[c]}", flush=True)
    mean = sum(res.values()) / 3
    out = {"variant": variant, "f_micro_w": {c: round(res[c], 6) for c in CATS},
           "per_ns": perns, "mean": round(mean, 6)}
    with open(f"{sdir}/result.json", "w") as w:
        json.dump(out, w, indent=2)
    print(f"MEAN={mean:.4f}  variant={variant}", flush=True)


if __name__ == "__main__":
    main()
