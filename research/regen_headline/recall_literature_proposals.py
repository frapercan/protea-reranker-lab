"""Literature (CG.2, abstract->GO) proposal regenerator for the recall-ceiling union.

Per LK/PK-BPO target, emit the propagated BP toi-index set of the top-K (=50) BP terms ranked by
MAX cosine(pre-t0 abstract, GO-definition) that are NOT already in the pool. Verbatim CG.2
generator (S-PubMedBert-MS-MARCO, set membership). Top-50 is CG.2's established generous operating
point (literature ranks the whole vocabulary, so an unbounded cut is meaningless); this makes the
literature recall contribution a floor, reported alongside its proposal size.

Leakage: PMIDs restricted to pub year < 2025 (< 2025-09-04 t0). Abstracts cached, no network.
Output: recall_scratch/lit_prop_top50.pkl = {accession: np.int32 toi-idx array}.
"""
import json, collections, time, re, pickle
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq

t0 = time.time()
ROOT = Path("/home/frapercan/Thesis2")
R = ROOT / "repositories/protea-reranker-lab/results/sparse_classifier"
EVAL = R / "percut_rerank/eval.parquet"
GTDIR = R / "lafa_gt"
OBO = ROOT / "protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
CACHE = ROOT / "storage/cooc_experiment/abstract_cache"
W = ROOT / "storage/regen_headline"
SC = W / "recall_scratch"; SC.mkdir(exist_ok=True)
TOI_FILE = str(GTDIR / "groundtruth_terms_of_interest.txt")
KTOP = 50
T0_YEAR = 2025
def log(m): print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)

par = collections.defaultdict(set); ns = {}; alt = {}; name = {}; deftxt = {}; cur = None
for line in open(OBO):
    line = line.rstrip("\n")
    if line == "[Term]": cur = None
    elif line.startswith("id: GO:"): cur = line[4:]
    elif cur and line.startswith("namespace: "): ns[cur] = line[11:]
    elif cur and line.startswith("name: "): name[cur] = line[6:]
    elif cur and line.startswith("def: "):
        m = re.search(r'"([^"]+)"', line)
        if m: deftxt[cur] = m.group(1)
    elif cur and line.startswith("is_a: GO:"): par[cur].add(line[6:].split(" ! ")[0].strip())
    elif cur and line.startswith("relationship: part_of GO:"): par[cur].add(line[len("relationship: part_of "):].split(" ! ")[0].strip())
    elif cur and line.startswith("alt_id: GO:"): alt[line[8:]] = cur
BP = {t for t, n in ns.items() if n == "biological_process"}
_anc = {}
def anc(t):
    t = alt.get(t, t)
    if t in _anc: return _anc[t]
    seen, st = set(), [t]
    while st:
        x = st.pop()
        if x in seen: continue
        seen.add(x)
        for p in par.get(x, ()): st.append(p)
    r = frozenset(x for x in seen if x in BP); _anc[t] = r; return r
toi_go = sorted({g.strip() for g in open(TOI_FILE) if g.strip() in BP})
toi_idx = {g: i for i, g in enumerate(toi_go)}
def to_toi(terms):
    s = set()
    for t in terms:
        j = toi_idx.get(t)
        if j is not None: s.add(j)
    return s
BP_DEF = sorted(t for t in BP if t in deftxt)
log(f"BP with def {len(BP_DEF)}; toi {len(toi_go)}")

def pmids_for(acc):
    p = CACHE / f"uniprot_{acc}.json"; kept = []
    if p.exists():
        try:
            for r in json.loads(p.read_text()).get("references", []):
                cit = r.get("citation", {})
                pm = [x["id"] for x in cit.get("citationCrossReferences", []) if x.get("database") == "PubMed"]
                yr = str(cit.get("publicationDate", ""))[:4]
                if pm and yr.isdigit() and int(yr) < T0_YEAR: kept.append(pm[0])
        except Exception: pass
    return sorted(set(kept))
abstracts = {}
for xmlf in CACHE.glob("pubmed_*.xml"):
    xml = xmlf.read_text()
    for art in re.split(r"<PubmedArticle>", xml)[1:]:
        pm = re.search(r"<PMID[^>]*>(\d+)</PMID>", art)
        ab = " ".join(re.findall(r"<AbstractText[^>]*>(.*?)</AbstractText>", art, re.S))
        ti = re.search(r"<ArticleTitle[^>]*>(.*?)</ArticleTitle>", art, re.S)
        if pm:
            txt = (ti.group(1) if ti else "") + " " + ab
            abstracts[pm.group(1)] = re.sub(r"<[^>]+>", " ", txt).strip()
log(f"abstracts {len(abstracts)}")

def load_pool(target):
    tb = pq.read_table(EVAL, columns=["protein_accession", "go_term_id"]).to_pandas()
    tb = tb[tb.protein_accession.isin(target)]
    tb["go"] = tb.go_term_id.map(lambda g: alt.get(g, g))
    tb = tb[tb.go.isin(BP)]
    return tb.groupby("protein_accession").go.apply(set).to_dict()

# universe = atlas LK + PK
universe = set()
for cell in ("lk", "pk"):
    for r in json.load(open(W / f"bp_atlas_perprotein_{cell}.json")): universe.add(r["protein"])

import torch
from sentence_transformers import SentenceTransformer
enc = SentenceTransformer("pritamdeka/S-PubMedBert-MS-MARCO", device="cpu")
DEFV = enc.encode([f"{name.get(g,'')}. {deftxt.get(g,'')}" for g in BP_DEF], batch_size=256,
                  normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
DEFV_t = torch.from_numpy(DEFV)
log(f"encoded {len(BP_DEF)} defs")

pool = load_pool(universe)
prot_abs = {}
for p in universe:
    a = [x for x in pmids_for(p) if x in abstracts]
    if a: prot_abs[p] = a
used = sorted({x for a in prot_abs.values() for x in a})
PM_IDX = {p: i for i, p in enumerate(used)}
PMV = enc.encode([abstracts[p][:2000] for p in used], batch_size=128, normalize_embeddings=True,
                 show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
PMV_t = torch.from_numpy(PMV)
log(f"encoded {len(used)} abstracts; {len(prot_abs)}/{len(universe)} targets with abstract")

out = {}
for p, abs_list in prot_abs.items():
    rows = [PM_IDX[x] for x in abs_list]
    sc = torch.max(DEFV_t @ PMV_t[rows].T, dim=1).values.numpy()
    order = np.argsort(-sc)
    present = pool.get(p, set())
    prop = set()
    picks = 0
    for j in order:
        g = BP_DEF[j]
        if g in present: continue
        picks += 1
        prop |= anc(g)
        if picks >= KTOP: break
    s = to_toi(prop)
    if s: out[p] = np.fromiter(s, dtype=np.int32, count=len(s))
pickle.dump(out, open(SC / "lit_prop_top50.pkl", "wb"))
log(f"wrote {len(out)} literature proposal sets -> lit_prop_top50.pkl")
