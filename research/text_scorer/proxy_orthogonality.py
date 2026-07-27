"""Orthogonality proxy: does ProtST (text-aligned) add over the champion d8979601?

Functional-gold top-k neighbour purity on the 7401 substrate, per representation:
d8979601 (learned k-WTA champion), ProtST (text-aligned), and COMBINED (sum of
cosine sims). Stratified by category (NK/LK/PK) and aspect (all-GO vs BP-only
propagated Jaccard). NK is the leakage-free anchor; BP is the wall. If COMBINED
does not beat d8979601-alone (esp. on NK-BP / LK-BP), text adds nothing -> stop.
"""
import json, os, numpy as np
from collections import defaultdict

W = "/home/frapercan/Thesis2/storage/text_scorer"
LA = "/home/frapercan/Thesis2/storage/layer_ablation"
OBO = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
TOPK = 10
RNG = np.random.default_rng(0)

meta = json.load(open(os.path.join(W, "meta.json"))); accs = meta["accs"]
prot_go = json.load(open(os.path.join(LA, "prot_go.json")))
sub = {r["acc"]: r for r in json.load(open(os.path.join(LA, "substrate.json")))}
# d8979601 champion embeddings are keyed to the layer_ablation query order
qla = json.load(open(os.path.join(LA, "emb_ankh_base/meta.json")))["accs"]
d89 = np.load(os.path.join(LA, "query_d8979601.npy")).astype(np.float32)
d89_by = {a: d89[i] for i, a in enumerate(qla)}
protst = np.load(os.path.join(W, "query_protst.npy")).astype(np.float32)

# align both to a common acc order (intersection)
common = [a for a in accs if a in d89_by and a in prot_go]
idx = {a: i for i, a in enumerate(accs)}
P = np.stack([protst[idx[a]] for a in common])
D = np.stack([d89_by[a] for a in common])

# GO propagation (is_a + part_of) + namespace
par = defaultdict(set); ns = {}; cur = None
for line in open(OBO):
    line = line.strip()
    if line == "[Term]": cur = None
    elif line.startswith("id: GO:"): cur = line[4:]
    elif line.startswith("namespace:") and cur: ns[cur] = line.split(" ", 1)[1]
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
BP = {g for g, n in ns.items() if n == "biological_process"}
prop = {}; propBP = {}
for a in common:
    s = set(prot_go.get(a, []))
    for t in list(s): s |= anc(t)
    prop[a] = s; propBP[a] = s & BP

def jac(i, j, bp=False):
    A = propBP if bp else prop
    a, b = A[common[i]], A[common[j]]
    if not a or not b: return 0.0
    inter = len(a & b); return inter / (len(a) + len(b) - inter)

def l2(X):
    n = np.linalg.norm(X, axis=1, keepdims=True); n[n == 0] = 1; return X / n
Pn, Dn = l2(P), l2(D)
cat = {a: sub[a]["cat"] for a in common}
def lb(L): return "short" if L <= 318 else ("long" if L >= 970 else "med")
lbk = {a: lb(sub[a]["len"]) for a in common}

# per-query top-k purity under each similarity, stratified
q = RNG.choice(len(common), size=min(1500, len(common)), replace=False)
SP_ = Pn[q] @ Pn.T; SD_ = Dn[q] @ Dn.T
SC_ = SP_ + SD_  # combined (equal-weight cosine sum)
def purity(S, bp):
    out = defaultdict(list)
    for qi, srow in zip(q, S):
        nn = np.argpartition(-srow, TOPK + 1)[:TOPK + 1]
        nn = [n for n in nn if n != qi][:TOPK]
        val = float(np.mean([jac(qi, n, bp) for n in nn])) if nn else 0.0
        out["all"].append(val); out[f"cat:{cat[common[qi]]}"].append(val); out[f"len:{lbk[common[qi]]}"].append(val)
    return {k: round(float(np.mean(v)), 4) for k, v in out.items()}

print("=== ORTHOGONALITY PROXY: ProtST vs champion d8979601 (top-10 purity) ===")
for bp, tag in ((False, "ALL-GO"), (True, "BP-only")):
    print(f"\n--- {tag} ---")
    rd = purity(SD_, bp); rp = purity(SP_, bp); rc = purity(SC_, bp)
    for key in ("all", "cat:nk", "cat:lk", "cat:pk", "len:long"):
        d, p, c = rd.get(key, 0), rp.get(key, 0), rc.get(key, 0)
        flag = "  <== text ADDS" if c > d + 0.003 else ("  (redundant)" if c <= d + 0.001 else "")
        print(f"  {key:9s} d8979601={d:.4f}  ProtST={p:.4f}  COMBINED={c:.4f}  (comb-d89={c-d:+.4f}){flag}")
json.dump({"note": "top-10 functional purity; combined=cos(ProtST)+cos(d8979601)"},
          open(os.path.join(W, "proxy_orthogonality_result.json"), "w"))
print("\nVERDICT anchor: NK-BP combined vs d8979601 (NK = leakage-free).")
