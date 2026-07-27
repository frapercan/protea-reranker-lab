"""Phase 1 prep: term-feature blocks for the learned k-WTA GO encoder.

CORE signal = CO-ANNOTATION from the FROZEN t0 corpus (reference_annotations.parquet,
v227 = t0). Computed DIRECTLY (never the live DB). PPMI-weighted term-term matrix ->
TruncatedSVD(256) with the basis SAVED (this is what closes the two-tower's unsaved-
basis gap). Auxiliary ABLATABLE blocks: ontology structure (ancestor incidence -> SVD128,
basis saved) and BioBERT GO-text (768, whitened; whitening mean saved).

Also precomputes, for the contrastive objective:
  - term-term co-occurrence (CSR) : defines POSITIVE pairs (share proteins) and the
    hard-negative mask (co-occurrence == 0).
  - text-kNN neighbours per term  : hard-negative candidates (text/ontology close).

Everything is strictly t0 (<= v227). Deterministic (seed 0), all bases saved.
Read-only w.r.t. repos; writes only under storage/kwta_go_encoder/.
"""
import json, time, collections
from pathlib import Path
import numpy as np, pyarrow.parquet as pq
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.decomposition import TruncatedSVD

t0 = time.time()
def log(m): print(f"[{time.time()-t0:.0f}s] {m}", flush=True)

OUT = Path("/home/frapercan/Thesis2/storage/kwta_go_encoder")
FROZEN = Path("/home/frapercan/Thesis2/storage/protea-frozen-v227-2025-09-04")
SC = Path("/home/frapercan/Thesis2/repositories/protea-reranker-lab/results/sparse_classifier")
T0D = Path("/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025")
OBO = str(T0D / "go-basic.obo")
SVD_COANN, SVD_STRUCT = 256, 128
SEED = 0

# ---------------- ontology ----------------
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
log(f"ontology BP {len(BP):,}")

# ---------------- vocab = BP terms with text ----------------
gt = np.load(SC / "go_text_emb.npz", allow_pickle=True)
text_ids = [alt.get(g, g) for g in gt["go_ids"].tolist()]
text_emb_all = gt["emb"].astype(np.float32)
seen = {}
for i, g in enumerate(text_ids):
    if g in BP and g not in seen:
        seen[g] = i
vocab = sorted(seen)
tid = {g: i for i, g in enumerate(vocab)}
N = len(vocab)
text = np.zeros((N, text_emb_all.shape[1]), np.float32)
for g, i in tid.items():
    text[i] = text_emb_all[seen[g]]
log(f"BP text vocab N={N}")

# ---------------- co-annotation from FROZEN corpus (DIRECT, t0) ----------------
meta = pq.read_table(FROZEN / "go_term_metadata.parquet")
id2go = {i: g for i, g in zip(meta.column("go_term_id").to_pylist(), meta.column("go_id").to_pylist())}
ra = pq.read_table(FROZEN / "reference_annotations.parquet")
acc = ra.column("accession").to_pylist(); gid = ra.column("go_term_id").to_pylist()
# protein index, term index (vocab only)
prot_ids = {}; rows_p = []; rows_t = []
for a, gi in zip(acc, gid):
    g = id2go.get(gi)
    if g is None: continue
    g = alt.get(g, g)
    j = tid.get(g)
    if j is None: continue
    pi = prot_ids.get(a)
    if pi is None: pi = prot_ids[a] = len(prot_ids)
    rows_p.append(pi); rows_t.append(j)
P = len(prot_ids)
A = coo_matrix((np.ones(len(rows_p), np.float32), (rows_p, rows_t)), shape=(P, N)).tocsr()
A.data[:] = 1.0  # binary presence
log(f"incidence A: proteins={P:,} nnz={A.nnz:,}")

# term-term co-occurrence counts C = A^T A  (# proteins sharing both terms)
C = (A.T @ A).tocsr()
C.setdiag(0); C.eliminate_zeros()
log(f"co-occurrence C: nnz={C.nnz:,} (term-term)")

# PPMI over C
Cc = C.tocoo(); tot = float(Cc.data.sum())
rs = np.asarray(C.sum(1)).ravel(); rs[rs == 0] = 1.0
pmi = np.log((Cc.data * tot) / (rs[Cc.row] * rs[Cc.col]) + 1e-12)
pmi[pmi < 0] = 0.0
PPMI = csr_matrix((pmi, (Cc.row, Cc.col)), shape=(N, N))
log(f"PPMI nnz={PPMI.nnz:,}")

# TruncatedSVD(256) on PPMI, basis SAVED
svd_c = TruncatedSVD(n_components=SVD_COANN, random_state=SEED).fit(PPMI)
coann = svd_c.transform(PPMI).astype(np.float32)
coann /= (np.linalg.norm(coann, axis=1, keepdims=True) + 1e-8)
term_cov = int((np.asarray((PPMI != 0).sum(1)).ravel() > 0).sum())
log(f"coann SVD {coann.shape}; terms with any co-occurrence: {term_cov:,}/{N:,}")

# ---------------- structure block: ancestor incidence -> SVD128 ----------------
anc_cols = {}; sr = []; sc = []
for g, i in tid.items():
    for a_ in (anc(g) | {g}):
        c = anc_cols.get(a_)
        if c is None: c = anc_cols[a_] = len(anc_cols)
        sr.append(i); sc.append(c)
S = coo_matrix((np.ones(len(sr), np.float32), (sr, sc)), shape=(N, len(anc_cols))).tocsr()
svd_s = TruncatedSVD(n_components=SVD_STRUCT, random_state=SEED).fit(S)
struct = svd_s.transform(S).astype(np.float32)
struct /= (np.linalg.norm(struct, axis=1, keepdims=True) + 1e-8)
log(f"struct SVD {struct.shape} (ancestor incidence, {len(anc_cols):,} cols)")

# ---------------- text block: whiten ----------------
text_mean = text.mean(0, keepdims=True)
textw = text - text_mean
textw /= (np.linalg.norm(textw, axis=1, keepdims=True) + 1e-8)

# ---------------- text kNN for hard-neg mining (top-32 per term) ----------------
import torch
dev = "cuda" if torch.cuda.is_available() else "cpu"
Tt = torch.from_numpy(textw).to(dev)
KNN = 32
knn_idx = np.zeros((N, KNN), np.int32)
with torch.no_grad():
    for s in range(0, N, 2048):
        sim = Tt[s:s+2048] @ Tt.t()
        for r in range(sim.shape[0]):
            sim[r, s+r] = -1e9
        knn_idx[s:s+sim.shape[0]] = torch.topk(sim, KNN, dim=1).indices.cpu().numpy()
log(f"text kNN {knn_idx.shape}")

# ---------------- save everything ----------------
np.savez(OUT / "term_features.npz",
         vocab=np.array(vocab),
         coann=coann, struct=struct, text=textw,
         knn_idx=knn_idx)
# co-occurrence graph (CSR) for positive sampling + hard-neg mask
np.savez(OUT / "cooc_graph.npz",
         indptr=C.indptr.astype(np.int64), indices=C.indices.astype(np.int32),
         data=C.data.astype(np.float32), N=N)
# reproducible bases
np.savez(OUT / "bases.npz",
         coann_components=svd_c.components_.astype(np.float32),
         struct_components=svd_s.components_.astype(np.float32),
         text_mean=text_mean.astype(np.float32),
         anc_cols=np.array(sorted(anc_cols, key=anc_cols.get)),
         vocab=np.array(vocab), seed=SEED,
         svd_coann=SVD_COANN, svd_struct=SVD_STRUCT)
meta_out = {"N": N, "proteins": P, "cooc_nnz": int(C.nnz), "term_cov_coann": term_cov,
            "svd_coann": SVD_COANN, "svd_struct": SVD_STRUCT, "text_dim": int(text.shape[1]),
            "seed": SEED, "source": "frozen v227 reference_annotations.parquet (DIRECT, t0)"}
json.dump(meta_out, open(OUT / "term_features_meta.json", "w"), indent=1)
log(f"SAVED term_features.npz / cooc_graph.npz / bases.npz  {meta_out}")
