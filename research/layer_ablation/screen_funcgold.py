"""Functional-gold screen across layer x sparsity for the representation ablation.

For each (layer, sparsity in {dense, kWTA-k}) computes two cheap functional
metrics over the substrate, stratified by category and length:
  (A) global Spearman between embedding cosine and propagated-GO Jaccard over a
      sample of protein pairs;
  (B) top-k neighbour functional purity: mean propagated-GO Jaccard to the top-10
      nearest neighbours in embedding space.
Higher = the representation places functionally-similar proteins closer.
"""
import json, os, numpy as np
from collections import defaultdict
from scipy.stats import spearmanr

W = "/home/frapercan/Thesis2/storage/layer_ablation"
OBO = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
KS = [64, 128, 256]
RNG = np.random.default_rng(0)
N_PAIRS = 200_000
TOPK = 10

meta = json.load(open(os.path.join(W, "emb_ankh_base/meta.json")))
accs = meta["accs"]; layers = meta["sampled_layers"]
prot_go = json.load(open(os.path.join(W, "prot_go.json")))
sub = {r["acc"]: r for r in json.load(open(os.path.join(W, "substrate.json")))}

# propagate GO to ancestors (is_a + part_of) for a functional Jaccard
par = defaultdict(set); cur = None
for line in open(OBO):
    line = line.strip()
    if line == "[Term]": cur = None
    elif line.startswith("id: GO:"): cur = line[4:]
    elif line.startswith("is_a:") and cur: par[cur].add(line.split()[1])
    elif line.startswith("relationship: part_of") and cur:
        p = line.split()
        if len(p) >= 3: par[cur].add(p[2])
_anc = {}
def anc(t):
    if t in _anc: return _anc[t]
    o = set(); st = list(par.get(t, ()))
    while st:
        a = st.pop()
        if a not in o: o.add(a); st.extend(par.get(a, ()))
    _anc[t] = o; return o
prop = {}
for a in accs:
    s = set(prot_go.get(a, []))
    for t in list(s): s |= anc(t)
    prop[a] = s

def jac(i, j):
    a, b = prop[accs[i]], prop[accs[j]]
    if not a or not b: return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)

def sparsify(X, k):
    if k is None: return X
    Xs = X.copy()
    for r in range(Xs.shape[0]):
        row = Xs[r]
        if k < row.size:
            thr = np.partition(np.abs(row), -k)[-k]
            row[np.abs(row) < thr] = 0.0
    return Xs

def l2(X):
    n = np.linalg.norm(X, axis=1, keepdims=True); n[n == 0] = 1; return X / n

# category / length strata
cat = {a: sub[a]["cat"] for a in accs}
def lbucket(L): return "short" if L <= 318 else ("long" if L >= 970 else "med")
lb = {a: lbucket(sub[a]["len"]) for a in accs}
idx = np.arange(len(accs))
# fixed pair sample for Spearman
pi = RNG.integers(0, len(accs), N_PAIRS); pj = RNG.integers(0, len(accs), N_PAIRS)
keep = pi != pj; pi, pj = pi[keep], pj[keep]
gold_pair = np.array([jac(i, j) for i, j in zip(pi, pj)], dtype=np.float32)
# same-length-bucket masks so Spearman is also length-stratified
_lb_arr = np.array([lb[accs[i]] for i in range(len(accs))])
sp_len_mask = {b: (_lb_arr[pi] == b) & (_lb_arr[pj] == b) for b in ("short", "med", "long")}

results = []
for L in layers:
    X0 = np.load(os.path.join(W, f"emb_ankh_base/layer_{L}.npy")).astype(np.float32)
    # per-dimension standardization stats over the corpus (transductive; a screen
    # convenience -- the real pipeline would fit these on the reference pool).
    mu = X0.mean(0); sd = X0.std(0) + 1e-6
    Xstd = (X0 - mu) / sd
    for norm, base in (("raw", X0), ("std", Xstd)):
      for k in [None] + KS:
        X = l2(sparsify(base, k))
        tag = ("dense" if k is None else f"kwta{k}") + f":{norm}"
        # (A) Spearman over pairs
        cos = np.einsum("ij,ij->i", X[pi], X[pj])
        rho = spearmanr(cos, gold_pair).correlation
        rho_len = {}
        for b, m in sp_len_mask.items():
            if m.sum() > 500:
                rho_len[b] = round(float(spearmanr(cos[m], gold_pair[m]).correlation), 4)
        # (B) top-k neighbour functional purity (sample 800 queries for speed)
        q = RNG.choice(idx, size=min(800, len(accs)), replace=False)
        S = X[q] @ X.T  # (Q, N)
        pur: dict = defaultdict(list)
        for qi, srow in zip(q, S):
            nn = np.argpartition(-srow, TOPK + 1)[:TOPK + 1]
            nn = [n for n in nn if n != qi][:TOPK]
            val = np.mean([jac(qi, n) for n in nn]) if nn else 0.0
            purity = val
            purl = purity
            purd = purity
            pur_all = purity
            purcat = cat[accs[qi]]; purlb = lb[accs[qi]]
            pur["all"].append(purity); pur[f"cat:{purcat}"].append(purity); pur[f"len:{purlb}"].append(purity)
        row = {"layer": L, "sparsity": tag, "spearman": round(float(rho), 4),
               "spearman_by_len": rho_len,
               "purity_all": round(float(np.mean(pur["all"])), 4)}
        for key in ("cat:nk", "cat:lk", "cat:pk", "len:short", "len:med", "len:long"):
            if pur.get(key): row[f"purity_{key}"] = round(float(np.mean(pur[key])), 4)
        results.append(row)
        print(row, flush=True)

json.dump(results, open(os.path.join(W, "screen_funcgold_results.json"), "w"), indent=2)
# best per metric
best_sp = max(results, key=lambda r: r["spearman"])
best_pu = max(results, key=lambda r: r["purity_all"])
print(f"\nBEST Spearman: layer {best_sp['layer']} {best_sp['sparsity']} = {best_sp['spearman']}")
print(f"BEST purity:   layer {best_pu['layer']} {best_pu['sparsity']} = {best_pu['purity_all']}")
