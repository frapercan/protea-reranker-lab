"""Cross-aspect association prior feature (targets PK), vectorized.

For an eval protein p with KNOWN t0 experimental terms K(p) (specific, freq<=FCAP), a candidate
term t gets two scores from the training co-occurrence cooc = Y^T Y:
  a_all   = sum_{k in K(p)}            cooc[k,t]/freq[k]   (= sum_k P(t|k))
  a_cross = sum_{k in K(p),asp(k)!=asp(t)} cooc[k,t]/freq[k]
a_cross is the genuine cross-aspect prior knowledge that drives PK (same-aspect is largely
ancestor-trivial). Computed as three sparse matmuls (one per known-term aspect):
  assoc_A = Kw_A @ cooc ;  a_all = sum_A assoc_A ;  a_cross[:,t] = a_all[:,t] - assoc_{asp(t)}[:,t]
Common known terms (freq>FCAP) are dropped (their P(t|k) ~ marginal, no signal).

Usage: assoc_feature.py FRAME_NPZ EVAL_LABELS_NPZ OUT.tsv
"""
import sys
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix

FRAME = sys.argv[1]; LABELS = sys.argv[2]; OUT = sys.argv[3]
OBO = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
FCAP = 1000          # exclude known terms more frequent than this (uninformative)
TOPN = 80            # cap emitted terms per protein

NS = {"biological_process": "P", "molecular_function": "F", "cellular_component": "C"}
aspect = {}; cur = None
for line in open(OBO):
    line = line.strip()
    if line == "[Term]": cur = None
    elif line.startswith("id: GO:"): cur = line[4:]
    elif line.startswith("namespace:") and cur: aspect[cur] = NS.get(line.split()[1], "?")

d = np.load(FRAME, allow_pickle=True)
vocab = [str(x) for x in d["vocab"]]; V = len(vocab)
rows, cols = d["rows"], d["cols"]
N = int(d["Xtr"].shape[0])
Y = csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(N, V))
freq = np.asarray(Y.sum(0)).ravel()
asp = np.array([aspect.get(t, "?") for t in vocab])

spec = np.where((freq >= 1) & (freq <= FCAP))[0]
inv = freq.copy(); inv[inv == 0] = 1.0
cooc = (Y[:, spec].T @ Y).tocsr().astype(np.float32)       # |spec| x V counts
# divide each spec row by its freq -> P(t|k)
cooc = cooc.multiply((1.0 / freq[spec])[:, None]).tocsr()

lab = np.load(LABELS, allow_pickle=True)
ev_acc = [str(x) for x in lab["ev_acc"]]
P = len(ev_acc)
lr, lc = lab["lab_rows"].astype(np.int64), lab["lab_cols"].astype(np.int64)
# keep only known terms that are "specific" (in spec); remap term idx -> spec-row idx
specpos = -np.ones(V, np.int64); specpos[spec] = np.arange(len(spec))
keep = specpos[lc] >= 0
kr, kc = lr[keep], specpos[lc[keep]]
spec_asp = asp[spec]

# K[p, j]=1 for known spec term j; split by aspect of the KNOWN term
assoc = {}
for A in ("F", "P", "C"):
    sel = spec_asp[kc] == A
    if not sel.any():
        assoc[A] = csr_matrix((P, V), dtype=np.float32); continue
    K_A = csr_matrix((np.ones(sel.sum(), np.float32), (kr[sel], kc[sel])), shape=(P, len(spec)))
    assoc[A] = (K_A @ cooc).tocsr()                        # P x V, contributions from aspect-A known terms

# a_cross[:,t] = a_all[:,t] - assoc[asp(t)][:,t]: subtract the same-aspect contribution per
# candidate column at emit time using the per-aspect matrices.
a_all = (assoc["F"] + assoc["P"] + assoc["C"]).tocsr()
aF, aP, aC = assoc["F"].tocsr(), assoc["P"].tocsr(), assoc["C"].tocsr()

n_out = 0
with open(OUT, "w") as w:
    for pi, acc in enumerate(ev_acc):
        r = a_all.getrow(pi)
        if r.nnz == 0: continue
        idx = r.indices; val = r.data
        # same-aspect contribution for each candidate term column
        same_val = np.zeros(len(idx), np.float32)
        for arr, A in ((aF, "F"), (aP, "P"), (aC, "C")):
            rr = arr.getrow(pi)
            if rr.nnz == 0: continue
            m = {j: v for j, v in zip(rr.indices, rr.data)}
            for q, t in enumerate(idx):
                if asp[t] == A:
                    same_val[q] += m.get(t, 0.0)
        cross = val - same_val
        order = np.argsort(-val)[:TOPN]
        for q in order:
            t = idx[q]
            w.write(f"{acc}\t{vocab[t]}\t{val[q]:.5f}\t{max(cross[q], 0.0):.5f}\n")
            n_out += 1
print(f"{OUT}: {n_out} rows, {P} eval prots, |spec|={len(spec)}", flush=True)
