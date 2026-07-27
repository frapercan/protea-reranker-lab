"""GORetriever-lite KILL-GATE (offline lab). One process: encode -> build learnable
Hadamard features -> OOF-train a text->GO retriever head -> honest AUROC vs the naive
max-cosine ceiling -> emit OOF retriever scores for the board-faithful blend harness.

Design (apples-to-apples with GATE 0):
  naive cosine(protein sentence, GO def) = UNIFORM-weight sum of the Hadamard product.
  A trained retriever = a LEARNED per-dimension reweighting of that same Hadamard.
  We give the lever its BEST shot: features = [ hadamard(argmax-sentence, GO) |
  hadamard(mean-sentence, GO) ] plus the naive scalars; a logistic head (learned
  diagonal metric) with GroupKFold-by-protein OOF (no per-protein label leakage).
  This is an OPTIMISTIC in-distribution ceiling for the learned head: if even this
  cannot beat naive cosine or add board lift, a temporally-honest retriever cannot.
Temporally honest text: only pre-cut (<= 2025-09) sentences.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
R = HERE.parent
CACHE = R / "literature_infame" / "uniprot_cache.json"
ABSTR = R / "phaseA_separability" / "abstracts.json"
HARD = R / "phaseA_separability" / "hard_frame.parquet"
OBO = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
ENCODER = "pritamdeka/S-PubMedBert-MS-MARCO"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
EMB_CACHE = HERE / "emb_cache.npz"
T0_YEAR, T0_MONTH = 2025, 9
NAIVE_CEIL = {"lk": 0.617, "pk": 0.602}  # GATE-0 abstract-subset naive max-cosine


def is_precut(r):
    y, m = r.get("year"), r.get("month")
    if y is None or y < T0_YEAR:
        return True
    if y > T0_YEAR:
        return False
    return (m is None) or (m <= T0_MONTH)


def parse_obo_defs(terms):
    name, defn, cur = {}, {}, {}
    with open(OBO) as f:
        for line in f:
            line = line.rstrip("\n")
            if line == "[Term]":
                cur = {}
            elif line.startswith("id: GO:"):
                cur["id"] = line[4:].strip()
            elif line.startswith("name:"):
                cur["name"] = line[5:].strip()
            elif line.startswith("def:"):
                mm = re.search(r'"(.*?)"', line)
                cur["def"] = mm.group(1) if mm else ""
            elif line == "":
                gid = cur.get("id")
                if gid and gid in terms:
                    name[gid] = cur.get("name", "")
                    defn[gid] = cur.get("def", "")
    return name, defn


def dedupe(sents):
    seen, out = set(), []
    for s in sents:
        k = s.lower()
        if k not in seen:
            seen.add(k)
            out.append(s)
    return out


def base_sentences(rec):
    sents = []
    if rec.get("desc"):
        sents.append(rec["desc"])
    for ftxt in rec.get("funcs", []):
        for s in re.split(r"(?<=[.;])\s+", ftxt):
            s = s.strip()
            if len(s) > 3:
                sents.append(s)
    for r in rec.get("refs", []):
        t = (r.get("title") or "").strip()
        if t and is_precut(r):
            sents.append(t)
    return dedupe(sents)


def abstract_sentences(rec, abstracts):
    out = []
    for r in rec.get("refs", []):
        pmid = r.get("pmid")
        if pmid and is_precut(r) and str(pmid) in abstracts:
            for s in re.split(r"(?<=[.;])\s+", abstracts[str(pmid)]):
                s = s.strip()
                if len(s) > 3:
                    out.append(s)
    return out


def build_substrate():
    cache = json.load(open(CACHE))
    abstracts = json.load(open(ABSTR))
    hard = pd.read_parquet(HARD)
    hard = hard[(hard.aspect == "bpo") & (hard.category.isin(["lk", "pk"]))].copy()

    prot_sents, has_abs = {}, {}
    for p in hard.protein_accession.unique():
        rec = cache.get(p) or {}
        b = base_sentences(rec)
        a = abstract_sentences(rec, abstracts)
        s = dedupe(b + a)[:160]
        if s:
            prot_sents[p] = s
            has_abs[p] = bool(a)
    covered = set(prot_sents)
    sub = hard[hard.protein_accession.isin(covered)].copy()

    terms = set(sub.go_term_id.unique())
    name, defn = parse_obo_defs(terms)
    go_terms = [g for g in terms if g in defn]
    go_idx = {g: i for i, g in enumerate(go_terms)}
    sub = sub[sub.go_term_id.isin(go_idx)].reset_index(drop=True)

    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(ENCODER, device=DEV)

    def enc(texts, bs=256):
        return st.encode(texts, batch_size=bs, normalize_embeddings=True,
                         show_progress_bar=False).astype(np.float32)

    go_texts = [(name.get(g, "") + ". " + defn.get(g, "")).strip() for g in go_terms]
    go_emb = enc(go_texts)
    allsents = sorted({s for ss in prot_sents.values() for s in ss})
    sidx = {s: i for i, s in enumerate(allsents)}
    semb = enc(allsents)
    return sub, go_emb, go_idx, semb, sidx, prot_sents, has_abs


def build_features(sub, go_emb, go_idx, semb, sidx, prot_sents):
    N = len(sub)
    Xarg = np.zeros((N, 768), dtype=np.float32)
    Xmean = np.zeros((N, 768), dtype=np.float32)
    max_cos = np.zeros(N, dtype=np.float32)
    mean_cos = np.zeros(N, dtype=np.float32)
    nsent = np.zeros(N, dtype=np.float32)
    semb_t = torch.from_numpy(semb).to(DEV)
    goemb_t = torch.from_numpy(go_emb).to(DEV)
    prot_mean = {}
    for p, ss in prot_sents.items():
        ids = [sidx[s] for s in ss]
        m = semb[ids].mean(0)
        n = np.linalg.norm(m)
        prot_mean[p] = (m / n if n > 0 else m).astype(np.float32)
    done = 0
    for p, g in sub.groupby("protein_accession", sort=False):
        ids = [sidx[s] for s in prot_sents[p]]
        S = semb_t[ids]
        cand = g.go_term_id.map(go_idx).to_numpy()
        rows = g.index.to_numpy()
        C = goemb_t[cand]
        sims = S @ C.T
        amax = sims.argmax(0)
        Xarg[rows] = (S[amax] * C).cpu().numpy()
        pm = torch.from_numpy(prot_mean[p]).to(DEV)
        Xmean[rows] = (pm.unsqueeze(0) * C).cpu().numpy()
        max_cos[rows] = sims.max(0).values.cpu().numpy()
        mean_cos[rows] = (pm @ C.T).cpu().numpy()
        nsent[rows] = len(ids)
        done += 1
        if done % 1000 == 0:
            print(f"  feat {done} proteins", flush=True)
    return Xarg, Xmean, max_cos, mean_cos, nsent


def auroc(y, s):
    m = np.isfinite(s)
    if y[m].sum() == 0 or y[m].sum() == m.sum():
        return None
    return round(float(roc_auc_score(y[m], s[m])), 4)


def main():
    sub, go_emb, go_idx, semb, sidx, prot_sents, has_abs = build_substrate()
    print(f"covered pairs {len(sub)} | lk {int((sub.category=='lk').sum())} "
          f"pk {int((sub.category=='pk').sum())}", flush=True)
    Xarg, Xmean, max_cos, mean_cos, nsent = build_features(
        sub, go_emb, go_idx, semb, sidx, prot_sents)

    y = sub.label.to_numpy().astype(np.int8)
    cat = sub.category.to_numpy()
    acc = sub.protein_accession.to_numpy()
    clf = sub.clfonly.to_numpy().astype(bool)
    hab = np.array([has_abs[p] for p in acc], dtype=bool)
    scal = np.column_stack([max_cos, mean_cos, np.log1p(nsent)]).astype(np.float32)

    report = {"encoder": ENCODER, "note": "OOF (GroupKFold-by-protein) trained "
              "logistic retriever vs naive max-cosine; optimistic in-distribution "
              "ceiling for the learned head.", "cells": {}}
    oof_all = np.full(len(sub), np.nan, dtype=np.float32)

    for c in ["lk", "pk"]:
        m = np.where(cat == c)[0]
        ym = y[m]
        groups = acc[m]
        X = np.hstack([Xarg[m], Xmean[m], scal[m]]).astype(np.float32)
        oof = np.zeros(len(m), dtype=np.float32)
        n_splits = min(5, len(np.unique(groups)))
        gkf = GroupKFold(n_splits=n_splits)
        for tr, te in gkf.split(X, ym, groups):
            sc = StandardScaler().fit(X[tr])
            clf_lr = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
            clf_lr.fit(sc.transform(X[tr]), ym[tr])
            oof[te] = clf_lr.predict_proba(sc.transform(X[te]))[:, 1]
        oof_all[m] = oof

        def strat(mask, name):
            if mask.sum() == 0:
                return None
            yy = ym[mask]
            if yy.sum() == 0 or yy.sum() == mask.sum():
                return {"n": int(mask.sum()), "pos": int(yy.sum()), "note": "degenerate"}
            return {"n": int(mask.sum()), "pos": int(yy.sum()),
                    "auroc_naive": auroc(yy, max_cos[m][mask]),
                    "auroc_trained": auroc(yy, oof[mask]),
                    "ap_naive": round(float(average_precision_score(yy, max_cos[m][mask])), 4),
                    "ap_trained": round(float(average_precision_score(yy, oof[mask])), 4)}

        habm = hab[m]
        clfm = clf[m]
        cell = {
            "naive_ceiling_gate0": NAIVE_CEIL[c],
            "all_covered": strat(np.ones(len(m), bool), "all"),
            "abstract_subset": strat(habm, "abs"),
            "title_only_subset": strat(~habm, "noabs"),
            "clfonly_remote": strat(clfm, "clfonly"),
            "reachable": strat(~clfm, "reach"),
        }
        report["cells"][c] = cell
        a = cell["all_covered"]
        print(f"\n{c.upper()}-BP  naive {a['auroc_naive']}  trained {a['auroc_trained']} "
              f"(ceiling {NAIVE_CEIL[c]})  n={a['n']} pos={a['pos']}", flush=True)

    # emit OOF retriever scores for the board-faithful blend harness (min-max to [0,1])
    ts = pd.DataFrame({"acc": acc, "go": sub.go_term_id.to_numpy(),
                       "cat": cat, "text_score_raw": oof_all})
    for c in ["lk", "pk"]:
        mm = ts.cat == c
        v = ts.loc[mm, "text_score_raw"]
        lo, hi = float(v.min()), float(v.max())
        ts.loc[mm, "text_score"] = (v - lo) / (hi - lo + 1e-9)
    ts[["acc", "go", "cat", "text_score", "text_score_raw"]].to_parquet(
        HERE / "retriever_oof_scores.parquet")
    json.dump(report, open(HERE / "retriever_auroc.json", "w"), indent=2)
    print("\nwrote retriever_oof_scores.parquet + retriever_auroc.json", flush=True)


if __name__ == "__main__":
    main()
