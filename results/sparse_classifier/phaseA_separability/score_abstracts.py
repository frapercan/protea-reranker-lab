"""Titles-vs-abstracts literature separability on the HARD BP pairs.

For the hard PK/LK-BP proteins that gained >=1 pre-cut PubMed abstract, recompute
the literature max-cosine(protein text, GO def) TWICE on the SAME pairs:
  cos_title    = desc + FUNCTION comments + pre-cut reference TITLES   (shipped baseline)
  cos_abstract = the above + pre-cut ABSTRACT bodies                   (the upgrade)
Same encoder (S-PubMedBert-MS-MARCO), same GO-def embeddings, same temporal cut.
Output: lit_abstract_scores.parquet (acc, go, cos_title, cos_abstract).
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
CACHE = HERE.parent / "literature_infame" / "uniprot_cache.json"
ABSTR = HERE / "abstracts.json"
HARD = HERE / "hard_frame.parquet"
OBO = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
ENCODER = "pritamdeka/S-PubMedBert-MS-MARCO"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
T0_YEAR, T0_MONTH = 2025, 9


def is_precut(r):
    y, m = r.get("year"), r.get("month")
    if y is None:
        return True
    if y < T0_YEAR:
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


def base_sentences(rec):
    """desc + function comments + pre-cut reference titles."""
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
    """pre-cut abstract bodies, split into sentences."""
    out = []
    for r in rec.get("refs", []):
        pmid = r.get("pmid")
        if pmid and is_precut(r) and str(pmid) in abstracts:
            body = abstracts[str(pmid)]
            for s in re.split(r"(?<=[.;])\s+", body):
                s = s.strip()
                if len(s) > 3:
                    out.append(s)
    return out


def dedupe(sents):
    seen, out = set(), []
    for s in sents:
        k = s.lower()
        if k not in seen:
            seen.add(k)
            out.append(s)
    return out


def main():
    cache = json.load(open(CACHE))
    abstracts = json.load(open(ABSTR))
    hard = pd.read_parquet(HARD, columns=["protein_accession", "go_term_id"])
    # restrict to proteins that gained an abstract
    prot_with_abs = set()
    for p in hard.protein_accession.unique():
        rec = cache.get(p) or {}
        if abstract_sentences(rec, abstracts):
            prot_with_abs.add(p)
    sub = hard[hard.protein_accession.isin(prot_with_abs)].drop_duplicates()
    terms = set(sub.go_term_id.unique())
    print(f"proteins with abstracts: {len(prot_with_abs)}; "
          f"hard pairs to score: {len(sub)}; unique GO: {len(terms)}", flush=True)

    name, defn = parse_obo_defs(terms)
    go_terms = [g for g in terms if g in defn]
    go_texts = [(name.get(g, "") + ". " + defn.get(g, "")).strip() for g in go_terms]
    go_idx = {g: i for i, g in enumerate(go_terms)}

    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(ENCODER, device=DEV)

    def enc(texts, bs=128):
        return st.encode(texts, batch_size=bs, normalize_embeddings=True,
                         show_progress_bar=False).astype(np.float32)

    go_emb = torch.from_numpy(enc(go_texts)).to(DEV)

    # build both sentence sets per protein
    base_s, abs_s = {}, {}
    allsents = set()
    for p in prot_with_abs:
        rec = cache.get(p) or {}
        b = base_sentences(rec)[:64]
        a = dedupe(b + abstract_sentences(rec, abstracts))[:128]
        base_s[p] = b
        abs_s[p] = a
        allsents.update(b)
        allsents.update(a)
    uniq = sorted(allsents)
    print(f"unique sentences (title+abstract corpus): {len(uniq)}", flush=True)
    semb = torch.from_numpy(enc(uniq)).to(DEV)
    sidx = {s: i for i, s in enumerate(uniq)}

    rows = []
    grp = sub.groupby("protein_accession")
    done = 0
    for p, g in grp:
        cand = [x for x in g.go_term_id.tolist() if x in go_idx]
        if not cand:
            continue
        C = go_emb[[go_idx[x] for x in cand]]
        bs = base_s[p]
        ab = abs_s[p]
        ct = (semb[[sidx[s] for s in bs]] @ C.T).max(0).values.cpu().numpy() if bs else np.zeros(len(cand), np.float32)
        ca = (semb[[sidx[s] for s in ab]] @ C.T).max(0).values.cpu().numpy() if ab else ct
        for go, t, a in zip(cand, ct, ca):
            rows.append((p, go, float(t), float(a)))
        done += 1
        if done % 100 == 0:
            print(f"  scored {done}/{len(prot_with_abs)}", flush=True)

    out = pd.DataFrame(rows, columns=["acc", "go", "cos_title", "cos_abstract"])
    out.to_parquet(HERE / "lit_abstract_scores.parquet")
    json.dump({"proteins_with_abstracts": len(prot_with_abs),
               "hard_pairs_scored": len(out),
               "mean_cos_title": round(float(out.cos_title.mean()), 4),
               "mean_cos_abstract": round(float(out.cos_abstract.mean()), 4)},
              open(HERE / "abstract_score_meta.json", "w"), indent=2)
    print("wrote lit_abstract_scores.parquet", len(out), flush=True)


if __name__ == "__main__":
    main()
