"""ProtEx verification -- STAGE 2: per-(protein, candidate-term) POSITIVE and NEGATIVE exemplars.

For every candidate pair we split the query's t0 neighbours into
  POSITIVE exemplars = neighbours that CARRY the candidate term (propagated, t0)   -- ProtEx yes-set
  NEGATIVE exemplars = neighbours that LACK it (hard: similar sequence, no term)    -- ProtEx no-set
and keep the K closest of each. Only reference indices are stored; the pooled embeddings are gathered
at train time from the resident fp16 matrix, so nothing 3000-d is ever materialised for millions of
rows.

split=train/val -> DS train.parquet rows, temporal split by snapshot_pair (past = != v225-v227,
val = v225-v227), subsampled to all positives + NEG_PER_POS negatives per cell (the submission cut is
a quantile of the score so a shifted base rate is harmless and the positives are the signal).
split=test -> the DEPLOYED pool eval_scores.parquet (the anchor-A candidate set), ALL rows, no labels
here (labels come from the release ground truth in the evaluator).
"""
import sys, json, collections, time
from pathlib import Path
import numpy as np, pyarrow.parquet as pq

t0 = time.time()
def log(m): print(f"[{time.time()-t0:.0f}s] {m}", flush=True)
W = Path("/home/frapercan/Thesis2/storage/protex")
DS = Path("/home/frapercan/Thesis2/repositories/protea-reranker-lab/datasets/protst-global-train227-test230")
CW = Path("/home/frapercan/Thesis2/storage/cooc_experiment/rerank_out/eval_scores.parquet")
T0D = Path("/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025")
OBO = str(T0D / "go-basic.obo")
K = 8
NEG_PER_POS = 5
VAL_PAIR = "v225-v227"
SPLIT = sys.argv[1]                     # train | val | test
rng = np.random.default_rng(0)

# alt-id map (for term normalisation, matches prep)
alt = {}; cur = None
for line in open(OBO):
    line = line.rstrip("\n")
    if line == "[Term]": cur = None
    elif line.startswith("id: GO:"): cur = line[4:]
    elif cur and line.startswith("alt_id: GO:"): alt[line[8:]] = cur

# prep artifacts
ref_accs = np.load(W / "ref_accs.npy", allow_pickle=True)
racc2i = {a: i for i, a in enumerate(ref_accs)}
query_accs = json.load(open(W / "query_accs.json")); qacc2i = {a: i for i, a in enumerate(query_accs)}
cand_terms = json.load(open(W / "cand_terms.json")); cvocab = {g: i for i, g in enumerate(cand_terms)}
NB = np.load(W / "neigh_idx.npy"); NS = np.load(W / "neigh_sim.npy").astype(np.float32)
p_indptr = np.load(W / "post_indptr.npy"); p_indices = np.load(W / "post_indices.npy")
log(f"prep loaded: refs={len(ref_accs):,} queries={len(query_accs):,} cand_terms={len(cand_terms):,} M={NB.shape[1]}")

# ---- rows for this split ----
if SPLIT in ("train", "val"):
    t = pq.read_table(DS / "train.parquet",
                      columns=["protein_accession", "go_term_id", "category", "snapshot_pair", "label"])
    cat = np.asarray(t.column("category").to_pylist()); asp = None
    aspt = pq.read_table(DS / "train.parquet", columns=["aspect"])
    asp = np.asarray(aspt.column("aspect").to_pylist())
    sp = np.asarray(t.column("snapshot_pair").to_pylist())
    base = ((asp == "bpo") & ((cat == "pk") | (cat == "lk")))
    want = (sp != VAL_PAIR) if SPLIT == "train" else (sp == VAL_PAIR)
    sel = base & want
    P = np.asarray(t.column("protein_accession").to_pylist())[sel]
    G = np.array([alt.get(g, g) for g in np.asarray(t.column("go_term_id").to_pylist())[sel]])
    Y = (t.column("label").to_numpy(zero_copy_only=False).astype(np.int64)[sel] > 0).astype(np.float32)
    C = cat[sel]
    # subsample: all pos + NEG_PER_POS neg, per cell
    keep = np.zeros(len(P), bool)
    for c in ("pk", "lk"):
        cm = np.where(C == c)[0]
        pos = cm[Y[cm] > 0]; neg = cm[Y[cm] == 0]
        keep[pos] = True
        nsamp = min(len(neg), NEG_PER_POS * len(pos))
        keep[rng.choice(neg, nsamp, replace=False)] = True
    P, G, Y, C = P[keep], G[keep], Y[keep], C[keep]
    Sc = np.zeros(len(P), np.float32)      # no reranker score needed for train/val
    Dist = np.zeros(len(P), np.float32)
    log(f"{SPLIT}: {len(P):,} rows | pos {int(Y.sum()):,} | pk {int((C=='pk').sum()):,} lk {int((C=='lk').sum()):,}")
elif SPLIT == "test":  # deployed pool anchor set
    t = pq.read_table(CW)
    cat = np.asarray(t.column("category").to_pylist()); asp = np.asarray(t.column("aspect").to_pylist())
    sel = (asp == "bpo") & ((cat == "pk") | (cat == "lk"))
    P = np.asarray(t.column("protein_accession").to_pylist())[sel]
    G = np.array([alt.get(g, g) for g in np.asarray(t.column("go_term_id").to_pylist())[sel]])
    Sc = t.column("reranker_score").to_numpy(zero_copy_only=False).astype(np.float32)[sel]
    Dist = t.column("distance").to_numpy(zero_copy_only=False).astype(np.float32)[sel]
    Y = np.full(len(P), -1.0, np.float32)  # labels assigned in evaluator via release GT
    C = cat[sel]
    log(f"test: {len(P):,} rows | pk {int((C=='pk').sum()):,} lk {int((C=='lk').sum()):,}")
else:
    raise SystemExit(f"unknown split {SPLIT}")

# ---- exemplar assignment, grouped by protein ----
N = len(P)
pos_ref = np.full((N, K), -1, np.int32); neg_ref = np.full((N, K), -1, np.int32)
n_pos = np.zeros(N, np.int16); n_neg = np.zeros(N, np.int16)
pos_sim = np.zeros(N, np.float32); neg_sim = np.zeros(N, np.float32)
p_ref = np.full(N, -1, np.int32); term_ci = np.full(N, -1, np.int32)
order = np.argsort(P, kind="stable")
i = 0
done = 0
while i < N:
    j = i
    while j < N and P[order[j]] == P[order[i]]: j += 1
    rows = order[i:j]; acc = P[order[i]]
    qi = qacc2i.get(acc)
    if qi is not None:
        neigh = NB[qi]                       # (M,) ref indices, sim-desc
        sims = NS[qi]
        pri = racc2i.get(acc, -1)
        # invert neighbour postings -> term_ci -> list of neighbour positions (already sim-sorted)
        tmap = collections.defaultdict(list)
        for pos_j, r in enumerate(neigh):
            for ci in p_indices[p_indptr[r]:p_indptr[r + 1]]:
                tmap[ci].append(pos_j)
        for ri in rows:
            ci = cvocab.get(G[ri])
            if ci is None:
                continue
            p_ref[ri] = pri; term_ci[ri] = ci
            posj = tmap.get(ci, [])
            posset = set(posj)
            negj = [pj for pj in range(len(neigh)) if pj not in posset]
            pk_ = posj[:K]; nk_ = negj[:K]
            n_pos[ri] = len(pk_); n_neg[ri] = len(nk_)
            if pk_:
                pos_ref[ri, :len(pk_)] = neigh[pk_]; pos_sim[ri] = float(sims[pk_].mean())
            if nk_:
                neg_ref[ri, :len(nk_)] = neigh[nk_]; neg_sim[ri] = float(sims[nk_].mean())
    done += (j - i); i = j
    if done % 200000 < (j - i):
        log(f"  assigned {done:,}/{N:,}")

np.savez(W / f"exemplars_{SPLIT}.npz",
         p_ref=p_ref, term_ci=term_ci, pos_ref=pos_ref, neg_ref=neg_ref,
         n_pos=n_pos, n_neg=n_neg, pos_sim=pos_sim.astype(np.float16), neg_sim=neg_sim.astype(np.float16),
         label=Y, cell=C, reranker_score=Sc, distance=Dist, protein=P, term=G)
frac_haspos = float((n_pos > 0).mean())
log(f"DONE {SPLIT}: N={N:,} rows_with_pos={frac_haspos:.3f} mean_npos={n_pos.mean():.2f} mean_nneg={n_neg.mean():.2f}")
