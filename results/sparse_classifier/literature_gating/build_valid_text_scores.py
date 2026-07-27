"""Validation (225->227) literature text->GO max-cosine scores, EARLIER cut (<=2025-03).

Mirror of goretriever_lite/build_text_scores.py: score = max cosine(protein pre-cut
text sentences, GO definition) with S-PubMedBERT-MS-MARCO. Candidates = validation BP
(protein, GO) from valid_base. Writes text_scores_valid_bpo.parquet (acc, go, category,
cos, text_score[0,1]) and a coverage json.
"""
import os, re, json
import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "valid_uniprot_cache.json")
OBO = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
VALID_BASE = "/home/frapercan/Thesis2/repositories/protea-reranker-lab/results/sparse_classifier/interpro2go_test/valid_base.parquet"
ENCODER = "pritamdeka/S-PubMedBert-MS-MARCO"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def parse_obo_defs(terms):
    name, defn = {}, {}
    cur = {}
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
                m = re.search(r'"(.*?)"', line)
                cur["def"] = m.group(1) if m else ""
            elif line == "":
                gid = cur.get("id")
                if gid and gid in terms:
                    name[gid] = cur.get("name", "")
                    defn[gid] = cur.get("def", "")
    return name, defn


def protein_sentences(rec):
    """precut mode: desc + function comments + ONLY precut-bucket reference titles."""
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
        if t and r.get("bucket") == "precut":
            sents.append(t)
    seen, out = set(), []
    for s in sents:
        k = s.lower()
        if k not in seen:
            seen.add(k)
            out.append(s)
    return out[:64]


def main():
    cache = json.load(open(CACHE))
    vb = pd.read_parquet(VALID_BASE)
    bp = vb[(vb.aspect == "bpo") & (vb.category.isin(["lk", "pk"]))][
        ["protein_accession", "go_term_id", "category"]].drop_duplicates()
    bp = bp[bp.protein_accession.isin(cache)].copy()
    print("valid BP candidate rows:", len(bp), "proteins:", bp.protein_accession.nunique())

    # coverage report
    cov = {}
    for c in ["lk", "pk"]:
        prots = vb[(vb.aspect == "bpo") & (vb.category == c)].protein_accession.unique()
        have_func = sum(1 for p in prots if cache.get(p, {}).get("funcs"))
        have_precut = 0
        for p in prots:
            rec = cache.get(p, {})
            if rec.get("desc") or rec.get("funcs") or any(
                    r.get("bucket") == "precut" and r.get("title") for r in rec.get("refs", [])):
                have_precut += 1
        cov[c] = {"n_bp_proteins": int(len(prots)),
                  "function_comment_pct": round(100 * have_func / max(1, len(prots)), 1),
                  "any_precut_text_pct": round(100 * have_precut / max(1, len(prots)), 1)}
    # leakage audit
    n_precut = n_future = 0
    for p in bp.protein_accession.unique():
        for r in cache.get(p, {}).get("refs", []):
            if r.get("bucket") == "precut":
                n_precut += 1
            elif r.get("bucket") == "future":
                n_future += 1
    cov["precut_refs"] = n_precut
    cov["future_refs_excluded"] = n_future
    cov["cut"] = "<=2025-03 (v225 t0)"
    print("coverage:", json.dumps(cov, indent=2))
    json.dump(cov, open(os.path.join(HERE, "valid_text_coverage.json"), "w"), indent=2)

    terms = set(bp.go_term_id.unique())
    name, defn = parse_obo_defs(terms)
    go_terms = [g for g in terms if g in defn]
    go_texts = [(name.get(g, "") + ". " + defn.get(g, "")).strip() for g in go_terms]
    print("GO terms with defs:", len(go_terms), "/", len(terms))

    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(ENCODER, device=DEV)

    def enc(texts, bs=128):
        return st.encode(texts, batch_size=bs, normalize_embeddings=True,
                         show_progress_bar=False).astype(np.float32)

    print("embedding GO defs...", flush=True)
    go_emb = enc(go_texts)
    go_idx = {g: i for i, g in enumerate(go_terms)}

    prot_sents = {p: protein_sentences(cache[p]) for p in bp.protein_accession.unique()}
    uniq = sorted({s for ss in prot_sents.values() for s in ss})
    print("unique sentences:", len(uniq), flush=True)
    sent_emb = enc(uniq)
    sidx = {s: i for i, s in enumerate(uniq)}

    go_emb_t = torch.from_numpy(go_emb).to(DEV)
    rows = []
    for p, grp in bp.groupby("protein_accession"):
        sents = prot_sents.get(p, [])
        cand = [g for g in grp.go_term_id.tolist() if g in go_idx]
        if not cand:
            continue
        cat = grp.category.iloc[0]
        if not sents:
            for g in cand:
                rows.append((p, g, cat, 0.0))
            continue
        S = torch.from_numpy(sent_emb[[sidx[s] for s in sents]]).to(DEV)
        C = go_emb_t[[go_idx[g] for g in cand]]
        sim = S @ C.T
        mx = sim.max(0).values.cpu().numpy()
        for g, v in zip(cand, mx):
            rows.append((p, g, cat, float(v)))

    out = pd.DataFrame(rows, columns=["acc", "go", "category", "cos"])
    lo, hi = out.cos.min(), out.cos.max()
    out["text_score"] = (out.cos - lo) / (hi - lo + 1e-9)
    fn = os.path.join(HERE, "text_scores_valid_bpo.parquet")
    out.to_parquet(fn)
    print("wrote", fn, "rows", len(out), "prots", out.acc.nunique(),
          "cos[min,med,max]=", round(lo, 3), round(float(out.cos.median()), 3), round(hi, 3))


if __name__ == "__main__":
    main()
