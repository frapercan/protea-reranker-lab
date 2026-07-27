"""Phase 0.5: can a biomedical SEMANTIC encoder separate a protein's novel-missed BP terms
from random BP, using only its pre-t0 abstracts? The decisive gate before the build.

Phase 0 word-match gave 1.54x enrichment: real but diffuse at the word level (36% null base rate
because biology vocabulary is shared). The word floor cannot tell "the abstract IS ABOUT this
function" from "the abstract happens to use this word". A semantic encoder can. If it lifts the
separation well above the 1.54x word floor (>=3x, AUC well over 0.5), the signal is EXTRACTABLE and
Phase 1 is justified. If it stays ~1.5x (AUC ~0.55), the signal is too diffuse and we stop before
the multi-week build - it would replicate protst_text's +0.0016.

ONE variable vs Phase 0: same sample (SEED=42, 120 PK + 60 LK), same novelty guard (a) (novel = gt
minus v227 state), same cached abstracts. Only the matcher changes: exact/long-word string match ->
cosine(SentenceEncoder(abstract), SentenceEncoder(GO definition)).

Metric = a RETRIEVAL task per protein. Positives = the protein's novel-missed BP terms. Negatives =
all other BP terms. Score each term by MAX cosine over the protein's pre-t0 abstracts (a term is
supported if ANY of its papers is about it). Report:
  - per-protein AUC (0.5 = encoder cannot separate; ->1.0 = perfect semantic pointing)
  - top-5% enrichment = recall@top-5% / 0.05  (directly comparable to Phase 0's 1.54x word floor)
Encoder: pritamdeka/S-PubMedBert-MS-MARCO (cached, biomedical retrieval). Read-only DB, no network
(all abstracts cached from Phase 0), no dispatch.
"""
import json, time, collections, re
from pathlib import Path
import numpy as np, pyarrow.parquet as pq, yaml, psycopg2

W = Path("/home/frapercan/Thesis2/storage/cooc_experiment")
CACHE = W / "abstract_cache"
REL = Path("/home/frapercan/Thesis2/CAFA_forever/data/releases/Sep_2025_Mar_2026")
T0D = Path("/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025")
OBO = str(T0D / "go-basic.obo")
V227 = "c905dffa-a5ce-430b-b17b-503e88666adb"
T0_YEAR = 2025
N_PK, N_LK, SEED = 120, 60, 42
t0 = time.time()

# ---- ontology: names, defs, parents, alt_ids ----
par = collections.defaultdict(set); ns = {}; alt = {}; name = {}; deftxt = {}
cur_ = None
for line in open(OBO):
    line = line.rstrip("\n")
    if line == "[Term]":
        cur_ = None
    elif line.startswith("id: GO:"):
        cur_ = line[4:]
    elif cur_ and line.startswith("namespace: "):
        ns[cur_] = line[11:]
    elif cur_ and line.startswith("name: "):
        name[cur_] = line[6:]
    elif cur_ and line.startswith("def: "):
        m = re.search(r'"([^"]+)"', line)
        if m:
            deftxt[cur_] = m.group(1)
    elif cur_ and line.startswith("is_a: GO:"):
        par[cur_].add(line[6:].split(" ! ")[0].strip())
    elif cur_ and line.startswith("relationship: part_of GO:"):
        par[cur_].add(line[len("relationship: part_of "):].split(" ! ")[0].strip())
    elif cur_ and line.startswith("alt_id: GO:"):
        alt[line[8:]] = cur_
BP = {t for t, n in ns.items() if n == "biological_process"}
AC = {}
def anc(t):
    if t in AC:
        return AC[t]
    o, st = set(), [t]
    while st:
        x = st.pop()
        for p in par.get(x, ()):
            if p not in o:
                o.add(p); st.append(p)
    AC[t] = o
    return o

# BP terms that have a definition -> the retrieval universe. text = "name. definition"
BP_DEF = sorted(t for t in BP if t in deftxt)
BP_IDX = {t: i for i, t in enumerate(BP_DEF)}
def term_text(g):
    return f"{name.get(g,'')}. {deftxt.get(g,'')}"
print(f"BP terms with definitions (retrieval universe): {len(BP_DEF):,}  ({time.time()-t0:.0f}s)", flush=True)

# ---- gt for PK/LK (aspect P), + pools (reuse Phase 0 exactly) ----
def load_gt(fn):
    G = collections.defaultdict(set)
    with (REL / fn).open() as fh:
        next(fh)
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) >= 3 and f[2] == "P":
                G[f[0]].add(alt.get(f[1], f[1]))
    return G
gt_pk, gt_lk = load_gt("groundtruth_PK.tsv"), load_gt("groundtruth_LK.tsv")

t = pq.read_table(W / "rerank_out" / "eval_scores.parquet")
c = np.asarray(t.column("category").to_pylist()); a = np.asarray(t.column("aspect").to_pylist())
def pool_of(cell):
    m = (c == cell) & (a == "bpo")
    P = np.asarray(t.column("protein_accession").to_pylist())[m]
    G = np.array([alt.get(g, g) for g in np.asarray(t.column("go_term_id").to_pylist())[m]])
    d = collections.defaultdict(set)
    for p_, g_ in zip(P, G):
        d[p_].add(g_)
    return d
pool_pk, pool_lk = pool_of("pk"), pool_of("lk")

rng = np.random.default_rng(SEED)
def sample(gt, pool, n):
    prots = sorted(set(gt) & set(pool))
    return sorted(rng.choice(prots, min(n, len(prots)), replace=False).tolist())
samp = [(p, "pk", gt_pk, pool_pk) for p in sample(gt_pk, pool_pk, N_PK)] + \
       [(p, "lk", gt_lk, pool_lk) for p in sample(gt_lk, pool_lk, N_LK)]
accs = [p for p, *_ in samp]

# ---- t0 (v227) BP state, novelty guard (a) ----
cfg = yaml.safe_load(Path("/home/frapercan/Thesis2/repositories/PROTEA/protea/config/system.yaml").read_text())
url = None
def find(d):
    global url
    if isinstance(d, dict):
        for v in d.values():
            if isinstance(v, str) and v.startswith("postgresql"):
                url = v
            else:
                find(v)
    elif isinstance(d, list):
        for v in d:
            find(v)
find(cfg); url = url.replace("postgresql+psycopg://", "postgresql://")
t0state = collections.defaultdict(set)
conn = psycopg2.connect(url); conn.set_session(readonly=True); cur = conn.cursor()
for s in range(0, len(accs), 500):
    cur.execute("""select protein_accession, go_id from protein_go_annotation pga
                   join go_term gt on gt.id = pga.go_term_id
                   where pga.annotation_set_id = %s and pga.protein_accession = any(%s)""",
                (V227, accs[s:s + 500]))
    for acc, g in cur.fetchall():
        g = alt.get(g, g)
        if g in BP:
            t0state[acc] |= {g} | anc(g)
conn.close()
print(f"t0 (v227) BP state loaded for {len(t0state)} proteins  ({time.time()-t0:.0f}s)", flush=True)

# ---- reload cached abstracts (NO network: read the exact caches Phase 0 wrote) ----
prot_pmids = {}
for acc in accs:
    p = CACHE / f"uniprot_{acc}.json"
    pmids = []
    if p.exists():
        try:
            for r in json.loads(p.read_text()).get("references", []):
                cit = r.get("citation", {})
                pm = [x["id"] for x in cit.get("citationCrossReferences", [])
                      if x.get("database") == "PubMed"]
                yr = str(cit.get("publicationDate", ""))[:4]
                if pm and yr.isdigit() and int(yr) < T0_YEAR:
                    pmids.append(pm[0])
        except Exception:
            pass
    prot_pmids[acc] = sorted(set(pmids))

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
print(f"cached abstracts: {len(abstracts):,}  ({time.time()-t0:.0f}s)", flush=True)

# ---- encode ----
import torch
from sentence_transformers import SentenceTransformer
dev = "cuda" if torch.cuda.is_available() else "cpu"
enc = SentenceTransformer("pritamdeka/S-PubMedBert-MS-MARCO", device=dev)
print(f"encoder loaded on {dev}  ({time.time()-t0:.0f}s)", flush=True)

DEFV = enc.encode([term_text(g) for g in BP_DEF], batch_size=256, normalize_embeddings=True,
                  show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
print(f"encoded {len(BP_DEF):,} GO definitions -> {DEFV.shape}  ({time.time()-t0:.0f}s)", flush=True)

used_pmids = sorted({p for acc in accs for p in prot_pmids[acc] if p in abstracts})
PM_IDX = {p: i for i, p in enumerate(used_pmids)}
PMV = enc.encode([abstracts[p][:2000] for p in used_pmids], batch_size=128,
                 normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
print(f"encoded {len(used_pmids):,} abstracts -> {PMV.shape}  ({time.time()-t0:.0f}s)", flush=True)

# ---- per protein: score every BP term by MAX cosine over the protein's abstracts ----
def auc(scores, pos_mask):
    # rank-based AUC; pos_mask boolean over the same index as scores
    n_pos = int(pos_mask.sum()); n_neg = len(scores) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(scores)                # ascending
    ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    return (ranks[pos_mask].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

rows = []
TOPFRAC = 0.05
for acc, cell, gt, pool in samp:
    pms = [p for p in prot_pmids[acc] if p in PM_IDX]
    if not pms:
        continue
    closure = set().union(*[{g} | anc(g) for g in gt[acc]]) if gt[acc] else set()
    gt_bp = {g for g in closure if g in BP_IDX}
    novel = {g for g in gt_bp if g not in t0state.get(acc, set())}
    missed = {g for g in novel if g not in pool.get(acc, set())}
    pos = np.zeros(len(BP_DEF), dtype=bool)
    idx = [BP_IDX[g] for g in missed if g in BP_IDX]
    if not idx:
        continue
    pos[idx] = True
    sub = PMV[[PM_IDX[p] for p in pms]]            # (n_papers, d)
    sc = (DEFV @ sub.T).max(axis=1)                # (n_BP,) max cosine over papers
    a_ = auc(sc, pos)
    k = max(1, int(TOPFRAC * len(BP_DEF)))
    top = set(np.argpartition(-sc, k)[:k].tolist())
    recall_top = len(top & set(idx)) / len(idx)
    rows.append({"acc": acc, "cell": cell, "n_papers": len(pms), "n_missed": int(pos.sum()),
                 "auc": a_, "recall_top5": recall_top})

def agg(rs):
    rs = [r for r in rs if r["auc"] is not None]
    if not rs:
        return None
    aucs = np.array([r["auc"] for r in rs])
    rec = np.array([r["recall_top5"] for r in rs])
    return {"proteins": len(rs), "missed_terms": int(sum(r["n_missed"] for r in rs)),
            "mean_auc": round(float(aucs.mean()), 4), "median_auc": round(float(np.median(aucs)), 4),
            "mean_recall_top5": round(float(rec.mean()), 4),
            "enrichment_top5_over_random": round(float(rec.mean() / TOPFRAC), 3)}

res = {"encoder": "pritamdeka/S-PubMedBert-MS-MARCO", "universe_BP_with_def": len(BP_DEF),
       "top_frac": TOPFRAC, "overall": agg(rows),
       "pk": agg([r for r in rows if r["cell"] == "pk"]),
       "lk": agg([r for r in rows if r["cell"] == "lk"])}
json.dump({"summary": res, "per_protein": rows}, open(W / "phase05_semantic.json", "w"), indent=1)

print(f"\n=== SEMANTIC: can the encoder point at what we miss? ({time.time()-t0:.0f}s) ===", flush=True)
for k in ("overall", "pk", "lk"):
    a_ = res[k]
    if a_:
        print(f"  {k:8s}: mean AUC {a_['mean_auc']}  median {a_['median_auc']} | "
              f"recall@top5% {a_['mean_recall_top5']} -> {a_['enrichment_top5_over_random']}x over random "
              f"(n={a_['proteins']} prot, {a_['missed_terms']} missed)", flush=True)
print("  Phase-0 word floor was 1.54x. Gate: AUC>>0.5 & enrichment>=3x -> BUILD; ~1.5x -> STOP.", flush=True)
print("DONE", flush=True)
