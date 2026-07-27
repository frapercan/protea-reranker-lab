"""PHYLO leakage ablation: recompute added-true IA-precision (top5/10) with EVERY neighbour partner
that is itself a test target REMOVED from partner BP sets. If precision holds, the signal comes from
independent, separately-annotated proteins in co-evolving OGs, not target-to-target laundering.
(Partner annotations are t0-frozen and target gains post-t0, so no temporal leakage by construction;
this is the empirical analog of CG.3's target-as-partner ablation.)
"""
import json, collections, time
from pathlib import Path
import numpy as np, pandas as pd
import pyarrow.parquet as pq
from scipy import sparse

t0 = time.time()
ROOT = Path("/home/frapercan/Thesis2")
F = ROOT / "storage/protea-frozen-v227-2025-09-04"
R = ROOT / "repositories/protea-reranker-lab/results/sparse_classifier"
EVAL = R / "percut_rerank/eval.parquet"; GTDIR = R / "lafa_gt"
OBO = ROOT / "protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
IA_F = ROOT / "protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv"
PH = ROOT / "storage/phylo_profile"; OUT = ROOT / "storage/regen_headline"
TOPN_NEIGH, SIM_MIN, MIN_PREV, MAX_PREV_FRAC = 50, 0.20, 4, 0.95


def log(m): print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)


par = collections.defaultdict(set); ns = {}; alt = {}; cur = None
for line in open(OBO):
    line = line.rstrip("\n")
    if line == "[Term]": cur = None
    elif line.startswith("id: GO:"): cur = line[4:]
    elif cur and line.startswith("namespace: "): ns[cur] = line[11:]
    elif cur and line.startswith("is_a: GO:"): par[cur].add(line[6:].split(" ! ")[0].strip())
    elif cur and line.startswith("relationship: part_of GO:"):
        par[cur].add(line[len("relationship: part_of "):].split(" ! ")[0].strip())
    elif cur and line.startswith("alt_id: GO:"): alt[line[8:]] = cur
BP = {t for t, n in ns.items() if n == "biological_process"}
_anc = {}
def anc(t):
    t = alt.get(t, t)
    if t in _anc: return _anc[t]
    seen, stack = set(), [t]
    while stack:
        x = stack.pop()
        if x in seen: continue
        seen.add(x)
        for p in par.get(x, ()): stack.append(p)
    r = frozenset(x for x in seen if x in BP); _anc[t] = r; return r
IA = {}
for line in open(IA_F):
    p = line.rstrip("\n").split("\t")
    if len(p) >= 2:
        try: IA[p[0]] = float(p[1])
        except ValueError: pass
def iamass(terms): return float(sum(IA.get(t, 0.0) for t in terms))

meta = pq.read_table(F / "go_term_metadata.parquet").to_pandas()
id2go = dict(zip(meta.go_term_id, meta.go_id)); id2asp = dict(zip(meta.go_term_id, meta.aspect))
ref = pq.read_table(F / "reference_annotations.parquet", columns=["accession", "go_term_id"]).to_pandas()
ref["go"] = ref.go_term_id.map(id2go).map(lambda g: alt.get(g, g) if isinstance(g, str) else g)
ref["asp"] = ref.go_term_id.map(id2asp)
acc2bpraw = ref.dropna(subset=["go"]).query("asp=='P'").groupby("accession").go.apply(lambda s: frozenset(s)).to_dict()
_bpprop = {}
def bp_prop(acc):
    r = _bpprop.get(acc)
    if r is None:
        raw = acc2bpraw.get(acc); r = frozenset().union(*[anc(t) for t in raw]) if raw else frozenset(); _bpprop[acc] = r
    return r

Pmat = sparse.load_npz(PH / "og_presence.npz").tocsr()
mm = np.load(PH / "og_meta.npz", allow_pickle=True)
ogs = list(mm["ogs"]); prevalence = mm["prevalence"]; ntax = Pmat.shape[1]
og_idx = {o: i for i, o in enumerate(ogs)}
acc2og = json.load(open(PH / "acc2og.json")); og_members = json.load(open(PH / "og_members.json"))
D = Pmat.toarray().astype(np.float32); mu = D.mean(axis=1, keepdims=True); Dc = D - mu
nn = np.sqrt((Dc * Dc).sum(axis=1, keepdims=True)); nn[nn == 0] = 1.0; Dn = Dc / nn; del D, Dc
informative = (prevalence >= MIN_PREV) & (prevalence <= MAX_PREV_FRAC * ntax)
cand_idx = np.where(informative)[0]; cand_norm = Dn[cand_idx]
log("profiles loaded")


def neighbours(xi, exclude):
    best = {}
    for i in xi:
        v = Dn[i]
        if not np.any(v): continue
        sims = cand_norm @ v
        for j in np.argsort(-sims)[:TOPN_NEIGH * 3]:
            s = float(sims[j])
            if s < SIM_MIN: break
            og = ogs[cand_idx[j]]
            if og in exclude: continue
            if s > best.get(og, -1): best[og] = s
    return sorted(best.items(), key=lambda kv: -kv[1])[:TOPN_NEIGH]


res = {}
for cell, gt_fn in [("LK-BPO", "groundtruth_LK.tsv"), ("PK-BPO", "groundtruth_PK.tsv")]:
    gt = pd.read_csv(GTDIR / gt_fn, sep="\t"); gt = gt[gt.aspect == "P"].copy()
    gt["term"] = gt.term.map(lambda g: alt.get(g, g))
    gt_leaf = gt.groupby("EntryID").term.apply(set).to_dict()
    targets = sorted(gt_leaf); target_set = set(targets)
    tb = pq.read_table(EVAL, columns=["protein_accession", "go_term_id"]).to_pandas()
    tb = tb[tb.protein_accession.isin(target_set)]; tb["go"] = tb.go_term_id.map(lambda g: alt.get(g, g))
    pool = tb[tb.go.isin(BP)].groupby("protein_accession").go.apply(set).to_dict()
    gt_prop = {p: frozenset().union(*[anc(t) for t in ts]) if ts else frozenset() for p, ts in gt_leaf.items()}
    pool_prop = {p: (frozenset().union(*[anc(t) for t in pool[p]]) if pool.get(p) else frozenset()) for p in targets}

    def measure(drop_target_partners):
        at = {5: 0.0, 10: 0.0}; af = {5: 0.0, 10: 0.0}
        for p in targets:
            ogx = acc2og.get(p, [])
            if not ogx: continue
            nb = neighbours([og_idx[o] for o in ogx if o in og_idx], set(ogx))
            sd = collections.defaultdict(float)
            for og_y, s in nb:
                accs = og_members.get(og_y, [])
                for a in accs:
                    if a == p: continue
                    if drop_target_partners and a in target_set: continue
                    for t in bp_prop(a): sd[t] += s
            if not sd: continue
            present = pool.get(p, set()); gtp = gt_prop[p]; covered = set(pool_prop[p]); picks = 0
            for g, sv in sorted(sd.items(), key=lambda kv: -kv[1]):
                if sv <= 0: break
                if g in present: continue
                picks += 1
                new = anc(g) - covered; tm = iamass(new & gtp); fm = iamass(new - gtp); covered |= new
                for k in (5, 10):
                    if picks <= k: at[k] += tm; af[k] += fm
                if picks >= 10: break
        return {f"top{k}": {"true": round(at[k], 2), "false": round(af[k], 2),
                            "prec": round(at[k] / (at[k] + af[k]), 4) if (at[k] + af[k]) > 0 else None} for k in (5, 10)}

    res[cell] = {"baseline_all_partners": measure(False), "ablation_target_partners_removed": measure(True)}
    log(f"{cell}: {json.dumps(res[cell])}")

json.dump(res, open(OUT / "phylo_leakage.json", "w"), indent=2)
log("wrote phylo_leakage.json")
print(json.dumps(res, indent=2))
