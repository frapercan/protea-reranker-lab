"""PK TEST literature text->GO max-cosine scores (cut <=2025-09), S-PubMedBERT.
Candidates = PK test BP (protein, GO) from the PK base pred pool. Writes
text_scores_test_pk_bpo.parquet (acc, go, category=pk, cos, text_score[0,1]).
"""
import os, re, json
import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "test_pk_uniprot_cache.json")
OBO = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
NS = "/home/frapercan/Thesis2/repositories/protea-reranker-lab/results/sparse_classifier/interpro2go_test/go_namespace.json"
PK_PRED = "/home/frapercan/Thesis2/repositories/protea-reranker-lab/results/sparse_classifier/percut_rerank/predictions/pk/pk.tsv"
ENCODER = "pritamdeka/S-PubMedBert-MS-MARCO"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


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
                m = re.search(r'"(.*?)"', line)
                cur["def"] = m.group(1) if m else ""
            elif line == "":
                gid = cur.get("id")
                if gid and gid in terms:
                    name[gid] = cur.get("name", "")
                    defn[gid] = cur.get("def", "")
    return name, defn


def protein_sentences(rec):
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
    nm = json.load(open(NS))
    pk = pd.read_csv(PK_PRED, sep="\t", header=None, names=["acc", "go", "score"])
    pk = pk[pk.go.map(nm) == "bpo"][["acc", "go"]].drop_duplicates()
    pk = pk[pk.acc.isin(cache)].copy()
    print("PK test BP candidate rows:", len(pk), "proteins:", pk.acc.nunique())

    terms = set(pk.go.unique())
    name, defn = parse_obo_defs(terms)
    go_terms = [g for g in terms if g in defn]
    go_texts = [(name.get(g, "") + ". " + defn.get(g, "")).strip() for g in go_terms]
    go_idx = {g: i for i, g in enumerate(go_terms)}

    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(ENCODER, device=DEV)

    def enc(texts, bs=128):
        return st.encode(texts, batch_size=bs, normalize_embeddings=True,
                         show_progress_bar=False).astype(np.float32)

    print("embedding GO defs...", flush=True)
    go_emb = enc(go_texts)
    prot_sents = {p: protein_sentences(cache[p]) for p in pk.acc.unique()}
    uniq = sorted({s for ss in prot_sents.values() for s in ss})
    print("unique sentences:", len(uniq), flush=True)
    sent_emb = enc(uniq)
    sidx = {s: i for i, s in enumerate(uniq)}

    go_emb_t = torch.from_numpy(go_emb).to(DEV)
    rows = []
    for p, grp in pk.groupby("acc"):
        sents = prot_sents.get(p, [])
        cand = [g for g in grp.go.tolist() if g in go_idx]
        if not cand:
            continue
        if not sents:
            for g in cand:
                rows.append((p, g, "pk", 0.0))
            continue
        S = torch.from_numpy(sent_emb[[sidx[s] for s in sents]]).to(DEV)
        C = go_emb_t[[go_idx[g] for g in cand]]
        mx = (S @ C.T).max(0).values.cpu().numpy()
        for g, v in zip(cand, mx):
            rows.append((p, g, "pk", float(v)))

    out = pd.DataFrame(rows, columns=["acc", "go", "category", "cos"])
    lo, hi = out.cos.min(), out.cos.max()
    out["text_score"] = (out.cos - lo) / (hi - lo + 1e-9)
    fn = os.path.join(HERE, "text_scores_test_pk_bpo.parquet")
    out.to_parquet(fn)
    print("wrote", fn, "rows", len(out), "prots", out.acc.nunique())


if __name__ == "__main__":
    main()
