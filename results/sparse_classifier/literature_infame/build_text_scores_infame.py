"""Text->GO max-cosine scores for ALL BP (protein,go) training+eval pairs.

Single temporal cut: publications <= 2025-09 (v227 training-frame t0). Since the
cut is snapshot-independent, text_score(protein,go) is computed ONCE per unique
(protein,go) BP pair and later joined to every training/eval row.

Encoder = S-PubMedBERT-MS-MARCO (same as the prior literature arm).
Output: text_scores_infame.parquet (acc, go, cos, text_score[0,1]).
"""
import os, re, json
import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "uniprot_cache.json")
PAIRS = os.path.join(HERE, "bp_pairs.parquet")
OBO = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
ENCODER = "pritamdeka/S-PubMedBert-MS-MARCO"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
T0_YEAR, T0_MONTH = 2025, 9


def is_precut(r):
    y, m = r.get("year"), r.get("month")
    if y is None:
        return True   # undated: keep (text only, low leak risk)
    if y < T0_YEAR:
        return True
    if y > T0_YEAR:
        return False
    return (m is None) or (m <= T0_MONTH)   # 2025: keep <=Sep and undated-month


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
                mm = re.search(r'"(.*?)"', line)
                cur["def"] = mm.group(1) if mm else ""
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
        if t and is_precut(r):
            sents.append(t)
    seen, out = set(), []
    for s in sents:
        k = s.lower()
        if k not in seen:
            seen.add(k)
            out.append(s)
    return out[:64]


def build_encoder():
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(ENCODER, device=DEV)

    def enc(texts, bs=128):
        return st.encode(texts, batch_size=bs, normalize_embeddings=True,
                         show_progress_bar=False).astype(np.float32)
    return enc


def main():
    cache = json.load(open(CACHE))
    pairs = pd.read_parquet(PAIRS)
    proteins = pairs.acc.unique()
    terms = set(pairs.go.unique())
    print(f"pairs={len(pairs)} proteins={len(proteins)} terms={len(terms)}",
          flush=True)

    name, defn = parse_obo_defs(terms)
    go_terms = [g for g in terms if g in defn]
    go_texts = [(name.get(g, "") + ". " + defn.get(g, "")).strip() for g in go_terms]
    print(f"GO terms with defs: {len(go_terms)}/{len(terms)}", flush=True)

    enc = build_encoder()
    print("encoding GO defs...", flush=True)
    go_emb = enc(go_texts)
    go_idx = {g: i for i, g in enumerate(go_terms)}

    print("building protein sentences...", flush=True)
    prot_sents = {}
    for p in proteins:
        rec = cache.get(p)
        prot_sents[p] = protein_sentences(rec) if rec else []
    uniq = sorted({s for ss in prot_sents.values() for s in ss})
    print(f"unique sentences: {len(uniq)}", flush=True)
    sent_emb = enc(uniq)
    sidx = {s: i for i, s in enumerate(uniq)}

    go_emb_t = torch.from_numpy(go_emb).to(DEV)        # C x d
    sent_emb_t = torch.from_numpy(sent_emb).to(DEV)    # N x d
    cand_by_prot = pairs.groupby("acc")
    accs_out, gos_out, cos_out = [], [], []
    done = 0
    for p, grp in cand_by_prot:
        cand = [g for g in grp.go.tolist() if g in go_idx]
        if not cand:
            continue
        sents = prot_sents.get(p, [])
        if not sents:
            accs_out.extend([p] * len(cand)); gos_out.extend(cand)
            cos_out.extend([0.0] * len(cand))
        else:
            S = sent_emb_t[[sidx[s] for s in sents]]         # S x d
            C = go_emb_t[[go_idx[g] for g in cand]]          # C x d
            mx = (S @ C.T).max(0).values.cpu().numpy()       # C
            accs_out.extend([p] * len(cand)); gos_out.extend(cand)
            cos_out.extend(mx.tolist())
        done += 1
        if done % 5000 == 0:
            print(f"  scored {done} proteins", flush=True)

    out = pd.DataFrame({"acc": accs_out, "go": gos_out,
                        "cos": np.asarray(cos_out, dtype=np.float32)})
    lo, hi = float(out.cos.min()), float(out.cos.max())
    out["text_score"] = ((out.cos - lo) / (hi - lo + 1e-9)).astype("float32")
    fn = os.path.join(HERE, "text_scores_infame.parquet")
    out.to_parquet(fn)
    cov = {"pairs_scored": len(out),
           "proteins": int(out.acc.nunique()),
           "pairs_with_text_gt0": int((out.cos > 0).sum()),
           "proteins_with_any_text": int(sum(1 for p in proteins if prot_sents.get(p))),
           "cos_min": round(lo, 4), "cos_med": round(float(out.cos.median()), 4),
           "cos_max": round(hi, 4)}
    json.dump(cov, open(os.path.join(HERE, "train_text_coverage.json"), "w"),
              indent=2)
    print("wrote", fn, cov, flush=True)


if __name__ == "__main__":
    main()
