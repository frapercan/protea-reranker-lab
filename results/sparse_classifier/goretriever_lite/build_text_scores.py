"""Stage 2: GORetriever-lite text->GO scoring (BioBERT, offline).

For each BP candidate (protein, GO) in the clean_227230 pool, score =
max cosine( protein pre-cut text sentences , GO definition ).
Writes text_scores_bpo.parquet (acc, go, category, cos, text_score[0,1]).
"""
import os
import re
import json
import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = "/tmp/claude-1000/-home-frapercan-Thesis2/afd2c43a-ede7-46dc-94dd-9745808d2194/scratchpad/uniprot_cache.json"
OBO = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
# ENCODER: 'biobert' (offline mean-pool) or a sentence-transformers model id.
ENCODER = os.environ.get("ENCODER", "pritamdeka/S-PubMedBert-MS-MARCO")
TEXT_MODE = os.environ.get("TEXT_MODE", "precut")  # precut | precut_refs_only | all
TAG = os.environ.get("TAG", "spubmed")
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def parse_obo_defs(terms):
    name, defn, ns = {}, {}, {}
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
            elif line.startswith("namespace:"):
                cur["ns"] = line[10:].strip()
            elif line.startswith("def:"):
                m = re.search(r'"(.*?)"', line)
                cur["def"] = m.group(1) if m else ""
            elif line == "":
                gid = cur.get("id")
                if gid and gid in terms:
                    name[gid] = cur.get("name", "")
                    defn[gid] = cur.get("def", "")
                    ns[gid] = cur.get("ns", "")
    return name, defn, ns


def protein_sentences(rec, mode):
    sents = []
    if rec.get("desc"):
        sents.append(rec["desc"])
    if mode != "precut_refs_only":
        for ftxt in rec.get("funcs", []):
            for s in re.split(r"(?<=[.;])\s+", ftxt):
                s = s.strip()
                if len(s) > 3:
                    sents.append(s)
    for r in rec.get("refs", []):
        t = (r.get("title") or "").strip()
        if not t:
            continue
        if mode == "all" or r.get("bucket") == "precut":
            sents.append(t)
    # de-dup, cap
    seen, out = set(), []
    for s in sents:
        k = s.lower()
        if k not in seen:
            seen.add(k)
            out.append(s)
    return out[:64]


def build_encoder():
    if ENCODER == "biobert":
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from transformers import AutoTokenizer, AutoModel
        mid = "dmis-lab/biobert-base-cased-v1.1"
        tok = AutoTokenizer.from_pretrained(mid)
        model = AutoModel.from_pretrained(mid).to(DEV).eval()

        @torch.no_grad()
        def enc(texts, bs=64):
            embs = []
            for i in range(0, len(texts), bs):
                e = tok(texts[i:i + bs], padding=True, truncation=True,
                        max_length=256, return_tensors="pt").to(DEV)
                o = model(**e).last_hidden_state
                m = e["attention_mask"].unsqueeze(-1).float()
                v = (o * m).sum(1) / m.sum(1).clamp(min=1e-9)
                v = torch.nn.functional.normalize(v, dim=1)
                embs.append(v.cpu())
            return torch.cat(embs, 0).numpy().astype(np.float32)
        return enc
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(ENCODER, device=DEV)

    def enc(texts, bs=64):
        return st.encode(texts, batch_size=bs, normalize_embeddings=True,
                         show_progress_bar=False).astype(np.float32)
    return enc


def main():
    cache = json.load(open(CACHE))
    bp = pd.read_parquet(os.path.join(HERE, "bpo_candidates.parquet"))
    # restrict to proteins we have text for (NK+LK fetched). PK not fetched.
    bp = bp[bp.protein_accession.isin(cache)].copy()
    print("BP candidate rows (NK+LK):", len(bp),
          "proteins:", bp.protein_accession.nunique())

    terms = set(bp.go_term_id.unique())
    name, defn, ns = parse_obo_defs(terms)
    go_terms = [g for g in terms if g in defn]
    go_texts = [(name.get(g, "") + ". " + defn.get(g, "")).strip()
                for g in go_terms]
    print("GO terms with defs:", len(go_terms), "/", len(terms))

    enc = build_encoder()
    print("encoder:", ENCODER)
    print("embedding GO defs...")
    go_emb = enc(go_texts)
    go_idx = {g: i for i, g in enumerate(go_terms)}

    # build per-protein sentences; embed unique sentences globally
    prot_sents = {p: protein_sentences(cache[p], TEXT_MODE)
                  for p in bp.protein_accession.unique()}
    uniq = sorted({s for ss in prot_sents.values() for s in ss})
    print("unique sentences:", len(uniq))
    sent_emb = enc(uniq)
    sidx = {s: i for i, s in enumerate(uniq)}

    go_emb_t = torch.from_numpy(go_emb).to(DEV)  # C x d
    rows = []
    cand_by_prot = bp.groupby("protein_accession")
    for p, grp in cand_by_prot:
        sents = prot_sents.get(p, [])
        cand = [g for g in grp.go_term_id.tolist() if g in go_idx]
        if not cand:
            continue
        cat = grp.category.iloc[0]
        if not sents:
            for g in cand:
                rows.append((p, g, cat, 0.0))
            continue
        S = torch.from_numpy(sent_emb[[sidx[s] for s in sents]]).to(DEV)  # SxD
        C = go_emb_t[[go_idx[g] for g in cand]]  # CxD
        sim = S @ C.T  # S x C
        mx = sim.max(0).values.cpu().numpy()  # C
        for g, v in zip(cand, mx):
            rows.append((p, g, cat, float(v)))

    out = pd.DataFrame(rows, columns=["acc", "go", "category", "cos"])
    # normalize cosine -> [0,1] min-max over all BP pairs
    lo, hi = out.cos.min(), out.cos.max()
    out["text_score"] = (out.cos - lo) / (hi - lo + 1e-9)
    fn = os.path.join(HERE, f"text_scores_bpo_{TAG}_{TEXT_MODE}.parquet")
    out.to_parquet(fn)
    print("wrote", fn, "rows", len(out),
          "cos[min,med,max]=", round(lo, 3),
          round(float(out.cos.median()), 3), round(hi, 3))


if __name__ == "__main__":
    main()
