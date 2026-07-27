"""P2b prep: build prep_big.npz for the LARGE curated v227 corpus.

Identical recipe to p2/prep.py (the 88K-experimental prep) but consuming the big
corpus dump: clf_labels_big.json + clf_protein_codes_big.npz. Vocab rule recomputed
on the big corpus (GO terms that are a propagated label for >=2 proteins). Same
seed-0 90/10 protein split style. GO_V / DAG edges from the same frozen
go_sparse_codes.npz + go_parents.json so either prep is interchangeable in Data.

Outputs: p2b/prep_big.npz
"""
import json, numpy as np, time, sys, os

BASE = "/home/frapercan/Thesis2/repositories/protea-reranker-lab/results/sparse_classifier"
OUT = f"{BASE}/p2b"
N = int(os.environ.get("MINCOUNT", "2"))

t0 = time.time()
GO_CODES_NPZ = os.environ.get("GO_CODES_NPZ", f"{BASE}/go_sparse_codes.npz")
PREP_OUT = os.environ.get("PREP_OUT", f"{OUT}/prep_big.npz")
g = np.load(GO_CODES_NPZ, allow_pickle=True)
go_ids = list(g['go_ids']); go_idx = {x: i for i, x in enumerate(go_ids)}
go_codes = g['codes'].astype(np.float32)  # 39906x1024 norm sqrt2
par = json.load(open(f"{BASE}/go_parents.json"))
labels = json.load(open(f"{BASE}/clf_labels_big.json"))
p = np.load(f"{BASE}/clf_protein_codes_big.npz", allow_pickle=True)
accs = list(p['accs'])
print(f"[{time.time()-t0:.1f}s] big corpus: {len(accs)} proteins, {len(labels)} label rows", flush=True)

sys.setrecursionlimit(2000000)
anc = {}
def ancestors(term):
    if term in anc:
        return anc[term]
    acc = set()
    for q in par.get(term, []):
        acc.add(q); acc |= ancestors(q)
    anc[term] = acc
    return acc

from collections import Counter
cnt = Counter(); prop = []
for acc in accs:
    s = set()
    for L in labels.get(acc, []):
        s.add(L); s |= ancestors(L)
    s = {x for x in s if x in go_idx}
    prop.append(s)
    for x in s:
        cnt[x] += 1
print(f"[{time.time()-t0:.1f}s] propagated. distinct terms seen={len(cnt)}", flush=True)

# vocab: propagated label for >= N proteins (recomputed on big corpus)
vocab_go = [gid for gid in go_ids if cnt.get(gid, 0) >= N]
vocab_idx = {gid: i for i, gid in enumerate(vocab_go)}
vocab_go_global = np.array([go_idx[gid] for gid in vocab_go])
GO_V = go_codes[vocab_go_global]  # V x 1024
print(f"[{time.time()-t0:.1f}s] vocab size={len(vocab_go)}", flush=True)

# per-protein positives in vocab space (ragged CSR)
indptr = [0]; indices = []
for s in prop:
    ids = sorted(vocab_idx[x] for x in s if x in vocab_idx)
    indices.extend(ids); indptr.append(len(indices))
indptr = np.array(indptr, dtype=np.int64); indices = np.array(indices, dtype=np.int32)

# term freq (vocab space) for baseline
tf = np.array([cnt[gid] for gid in vocab_go], dtype=np.int64)

# DAG edges within vocab (child_idx, parent_idx)
ch = []; pa = []
for gid in vocab_go:
    ci = vocab_idx[gid]
    for pp in par.get(gid, []):
        if pp in vocab_idx:
            ch.append(ci); pa.append(vocab_idx[pp])
edges = np.stack([np.array(ch, dtype=np.int32), np.array(pa, dtype=np.int32)], 0)
print(f"[{time.time()-t0:.1f}s] dag edges in vocab={edges.shape[1]}", flush=True)

# protein split 90/10 seed 0 (same style as p2)
rng = np.random.default_rng(0)
perm = rng.permutation(len(accs))
ntest = len(accs) // 10
test_idx = np.sort(perm[:ntest]); train_idx = np.sort(perm[ntest:])
print(f"[{time.time()-t0:.1f}s] train={len(train_idx)} test={len(test_idx)}", flush=True)

np.savez(PREP_OUT,
         GO_V=GO_V.astype(np.float32),
         indptr=indptr, indices=indices, tf=tf,
         edges=edges, train_idx=train_idx, test_idx=test_idx,
         vocab_go=np.array(vocab_go))
print(f"[{time.time()-t0:.1f}s] saved {PREP_OUT}", flush=True)
