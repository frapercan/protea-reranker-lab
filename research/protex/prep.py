"""ProtEx verification -- STAGE 1: leakage-clean t0 retrieval scaffold.

Builds, once, everything retrieval needs so later stages never touch the DB:
  - the frozen v227 ProtT5 reference matrix (574,627 x 1024), L2-normalised, fp16
  - per-query top-M cosine neighbours among the references (self + near-identical dropped)
  - per-reference PROPAGATED-BP annotation postings restricted to the candidate vocab (t0=v227)

Everything here is strictly t0 (<=v227): reference embeddings and reference annotations are the
frozen Sep-2025 (v227) bundle. Query proteins are the pk/lk BPO pool proteins (train + eval); their
POST-t0 candidate labels never enter this stage. Self-accession neighbours are removed and any
neighbour with cosine >= 0.9999 (near-identical vector = isoform/duplicate) is dropped as a leakage
guard, so a query can never retrieve a copy of itself.

Space choice: ProtT5 (frozen config 084943c6, mean-pooled 1024-d) is the ONLY representation in
which BOTH the full leakage-clean t0 reference set (574,627) AND every pool query (100% coverage)
are already materialised. Re-embedding 574k proteins with Ankh on a shared 12 GB GPU is unsafe and
unnecessary: retrieval-space consistency between query and reference is what the method needs, and
this is the canonical deployed KNN space.
"""
import json, collections, time
from pathlib import Path
import numpy as np, pyarrow.parquet as pq, torch

t0 = time.time()
def log(m): print(f"[{time.time()-t0:.0f}s] {m}", flush=True)
W = Path("/home/frapercan/Thesis2/storage/protex")
FR = Path("/home/frapercan/Thesis2/storage/protea-frozen-v227-2025-09-04")
DS = Path("/home/frapercan/Thesis2/repositories/protea-reranker-lab/datasets/protst-global-train227-test230")
T0D = Path("/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025")
OBO = str(T0D / "go-basic.obo")
M_NEIGH = 256
DUP_COS = 0.9999
dev = "cuda" if torch.cuda.is_available() else "cpu"

# ---- ontology (t0) ----
par = collections.defaultdict(set); ns = {}; alt = {}; cur = None
for line in open(OBO):
    line = line.rstrip("\n")
    if line == "[Term]": cur = None
    elif line.startswith("id: GO:"): cur = line[4:]
    elif cur and line.startswith("namespace: "): ns[cur] = line[11:]
    elif cur and line.startswith("is_a: GO:"): par[cur].add(line[6:].split(" ! ")[0].strip())
    elif cur and line.startswith("relationship: part_of GO:"): par[cur].add(line[len("relationship: part_of "):].split(" ! ")[0].strip())
    elif cur and line.startswith("alt_id: GO:"): alt[line[8:]] = cur
BP = {t for t, n in ns.items() if n == "biological_process"}
AC = {}
def anc(t):
    if t in AC: return AC[t]
    o, st = set(), [t]
    while st:
        x = st.pop()
        for p in par.get(x, ()):
            if p not in o: o.add(p); st.append(p)
    AC[t] = o; return o
def closure(terms):
    o = set()
    for g in terms: o.add(g); o |= anc(g)
    return o
log(f"ontology: BP={len(BP):,}")

# ---- candidate BP term vocab (train + eval pool, pk/lk bpo) ----
def pool_terms_prots(fn):
    t = pq.read_table(DS / fn, columns=["protein_accession", "go_term_id", "category", "aspect"])
    cat = np.asarray(t.column("category").to_pylist()); asp = np.asarray(t.column("aspect").to_pylist())
    m = ((cat == "pk") | (cat == "lk")) & (asp == "bpo")
    G = np.array([alt.get(g, g) for g in np.asarray(t.column("go_term_id").to_pylist())[m]])
    P = np.asarray(t.column("protein_accession").to_pylist())[m]
    return set(G.tolist()), set(P.tolist())
tterm, tprot = pool_terms_prots("train.parquet")
eterm, eprot = pool_terms_prots("eval.parquet")
cand_terms = sorted({g for g in (tterm | eterm) if g in BP})
cvocab = {g: i for i, g in enumerate(cand_terms)}
query_accs = sorted(tprot | eprot)
log(f"candidate BP terms={len(cand_terms):,} | query proteins={len(query_accs):,}")

# ---- frozen reference embeddings (t0), L2-normalised fp16 ----
log("loading reference embeddings (574k x 1024)...")
et = pq.read_table(FR / "reference_embeddings.parquet")
ref_accs = np.asarray(et.column("accession").to_pylist())
flat = et.column("embedding").combine_chunks().values.to_numpy(zero_copy_only=False).astype(np.float32)
E = flat.reshape(len(ref_accs), -1)
log(f"ref matrix {E.shape}")
E /= (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)
Ef16 = E.astype(np.float16)
np.save(W / "ref_emb_norm_fp16.npy", Ef16)
np.save(W / "ref_accs.npy", ref_accs)
racc2i = {a: i for i, a in enumerate(ref_accs)}
del flat, et

# ---- reference annotations (t0) -> per-ref propagated-BP postings in candidate vocab ----
# reference_annotations.go_term_id is an INTEGER internal id; map via go_term_metadata -> GO:xxxx
log("loading go_term_metadata id map...")
mt = pq.read_table(FR / "go_term_metadata.parquet", columns=["go_term_id", "go_id"])
id2go = {int(i): g for i, g in zip(mt.column("go_term_id").to_pylist(), mt.column("go_id").to_pylist())}
log("loading reference annotations (t0=v227)...")
at = pq.read_table(FR / "reference_annotations.parquet", columns=["accession", "go_term_id"])
ra = np.asarray(at.column("accession").to_pylist())
rg = np.array([alt.get(x, x) for x in (id2go.get(int(i), "") for i in at.column("go_term_id").to_pylist())])
cand_set = set(cand_terms)
# closure of every distinct annotated term, intersected with candidate vocab -- memoised
term_cand_closure = {}
def cand_anc(g):
    if g not in term_cand_closure:
        cc = closure([g]) if g in BP or par.get(g) else {g}
        term_cand_closure[g] = [cvocab[x] for x in cc if x in cand_set]
    return term_cand_closure[g]
post = collections.defaultdict(set)
for a, g in zip(ra, rg):
    ci = racc2i.get(a)
    if ci is None: continue
    cl = cand_anc(g)
    if cl: post[ci].update(cl)
log(f"refs with >=1 candidate-term posting: {len(post):,}")
indptr = np.zeros(len(ref_accs) + 1, np.int64); indices_l = []
for i in range(len(ref_accs)):
    s = post.get(i)
    if s:
        idx = sorted(s); indices_l.extend(idx)
    indptr[i + 1] = len(indices_l)
indices = np.asarray(indices_l, np.int32)
np.save(W / "post_indptr.npy", indptr); np.save(W / "post_indices.npy", indices)
log(f"postings CSR: nnz={len(indices):,}")
del at, ra, rg, post

# ---- kNN: top-M cosine neighbours per query (GPU, fp16), self + near-dup dropped ----
log(f"kNN top-{M_NEIGH} for {len(query_accs):,} queries on {dev}...")
Eg = torch.from_numpy(Ef16).to(dev)                       # (R,1024) fp16, normalised
q_ref_idx = np.array([racc2i[a] for a in query_accs], np.int64)
NB = np.zeros((len(query_accs), M_NEIGH), np.int32)
NS = np.zeros((len(query_accs), M_NEIGH), np.float16)
B = 128
with torch.no_grad():
    for s in range(0, len(query_accs), B):
        qi = q_ref_idx[s:s+B]
        Q = Eg[torch.from_numpy(qi).to(dev)]              # (b,1024)
        sim = Q @ Eg.t()                                  # (b,R) cosine (normalised)
        ar = torch.arange(len(qi), device=dev)
        sim[ar, torch.from_numpy(qi).to(dev)] = -1.0       # drop self
        sim[sim >= DUP_COS] = -1.0                         # drop near-identical
        v, idx = torch.topk(sim, M_NEIGH, dim=1)
        NB[s:s+len(qi)] = idx.cpu().numpy().astype(np.int32)
        NS[s:s+len(qi)] = v.cpu().numpy().astype(np.float16)
        if s % (B*50) == 0: log(f"  kNN {s+len(qi):,}/{len(query_accs):,}")
del Eg
np.save(W / "neigh_idx.npy", NB); np.save(W / "neigh_sim.npy", NS)
json.dump(query_accs, open(W / "query_accs.json", "w"))
json.dump(cand_terms, open(W / "cand_terms.json", "w"))
meta = {"space": "ProtT5 frozen 084943c6 (mean-pool 1024d), L2-normalised",
        "n_refs": len(ref_accs), "n_queries": len(query_accs), "n_cand_terms": len(cand_terms),
        "M_neighbours": M_NEIGH, "dup_cos_drop": DUP_COS,
        "leakage": "refs+ref-annotations strictly t0=v227; self-accession + cos>=0.9999 neighbours removed"}
json.dump(meta, open(W / "prep_meta.json", "w"), indent=1)
log(f"DONE prep. {json.dumps(meta)}")
