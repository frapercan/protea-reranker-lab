"""PHYLO STEP 1 -- the GATE (cross-species phylogenetic profiling channel).

Generator: for target protein p in OG_x (Eukaryota-2759 orthologous group), find OGs OG_y whose
species presence/absence PROFILE is most similar to OG_x (Pearson over the 5668-euk-taxon presence
vectors). Each neighbour OG_y is a "partner" carrying B(y) = union of frozen t0 BP annotations of the
reference proteins that belong to OG_y. OG_x itself is excluded (same OG = homolog => orthogonal reach).
    s(t | p) = sum over top-N similar OG_y (excl OG_x) of  sim(OG_x, OG_y) * 1[t in B(y)]

Metric machinery (obo BP ancestors, IA, gt/pool propagation, added-true vs added-false IA-mass,
generation precision, stratification) is REUSED VERBATIM from cg3_step1_gate.py. Only the proposal
source changed: profile-similar OG partners instead of STRING network partners.

Leakage: profiles from OrthoDB v12.2 genome content (2024, pre-t0 2025-09-04); orthology/presence is
annotation-independent. Partner annotations ONLY from frozen v227=t0 reference_annotations.parquet.
Self accession and OG_x (homolog OG) excluded. gene_xrefs.tab (GO) never read.
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
EVAL = R / "percut_rerank/eval.parquet"
GTDIR = R / "lafa_gt"
OBO = ROOT / "protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
IA_F = ROOT / "protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv"
PH = ROOT / "storage/phylo_profile"
OUT = ROOT / "storage/regen_headline"
TOPKS = [5, 10, 25, 50]
TOPN_NEIGH = 50          # neighbour OGs per target OG
SIM_MIN = 0.20           # Pearson floor for a neighbour edge
MIN_PREV, MAX_PREV_FRAC = 4, 0.95   # informative-profile band (drop housekeeping/ultra-rare)


def log(m): print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)


# ---------- obo  [VERBATIM CG.3] ----------
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
log(f"obo {len(BP):,} BP; IA {len(IA):,}")

# ---------- frozen reference BP sets  [VERBATIM CG.3] ----------
meta = pq.read_table(F / "go_term_metadata.parquet").to_pandas()
id2go = dict(zip(meta.go_term_id, meta.go_id)); id2asp = dict(zip(meta.go_term_id, meta.aspect))
ref = pq.read_table(F / "reference_annotations.parquet", columns=["accession", "go_term_id"]).to_pandas()
ref["go"] = ref.go_term_id.map(id2go).map(lambda g: alt.get(g, g) if isinstance(g, str) else g)
ref["asp"] = ref.go_term_id.map(id2asp)
ref = ref.dropna(subset=["go"])
acc2bpraw = ref[ref.asp == "P"].groupby("accession").go.apply(lambda s: frozenset(s)).to_dict()
_bpprop = {}
def bp_prop(acc):
    r = _bpprop.get(acc)
    if r is None:
        raw = acc2bpraw.get(acc)
        r = frozenset().union(*[anc(t) for t in raw]) if raw else frozenset()
        _bpprop[acc] = r
    return r
log(f"reference BP: {len(acc2bpraw):,} proteins")

# ---------- pool  [VERBATIM CG.3] ----------
def load_pool(target_prots):
    tb = pq.read_table(EVAL, columns=["protein_accession", "go_term_id"]).to_pandas()
    tb = tb[tb.protein_accession.isin(target_prots)]
    tb["go"] = tb.go_term_id.map(lambda g: alt.get(g, g))
    tb = tb[tb.go.isin(BP)]
    return tb.groupby("protein_accession").go.apply(set).to_dict()

# ---------- phylo profiles ----------
Pmat = sparse.load_npz(PH / "og_presence.npz").tocsr()
mm = np.load(PH / "og_meta.npz", allow_pickle=True)
ogs = list(mm["ogs"]); prevalence = mm["prevalence"]
og_idx = {o: i for i, o in enumerate(ogs)}
ntax = Pmat.shape[1]
acc2og = {k: v for k, v in json.load(open(PH / "acc2og.json")).items()}
og_members = json.load(open(PH / "og_members.json"))
log(f"profiles: {Pmat.shape} nnz {Pmat.nnz:,}; acc2og {len(acc2og):,}")

# informative-profile mask
maxprev = MAX_PREV_FRAC * ntax
informative = (prevalence >= MIN_PREV) & (prevalence <= maxprev)
log(f"informative OGs (prev in [{MIN_PREV},{maxprev:.0f}]): {int(informative.sum()):,} / {len(ogs):,}")

# dense centered+normalised rows for Pearson; only for informative OGs (neighbour candidates)
D = Pmat.toarray().astype(np.float32)          # (nOG, ntax)
mu = D.mean(axis=1, keepdims=True)
Dc = D - mu
norm = np.sqrt((Dc * Dc).sum(axis=1, keepdims=True))
norm[norm == 0] = 1.0
Dn = (Dc / norm)                                # unit rows; zero-variance rows -> ~0 vector
del D, Dc
cand_idx = np.where(informative)[0]
cand_norm = Dn[cand_idx]                         # (nCand, ntax)
log(f"cand matrix {cand_norm.shape} ({cand_norm.nbytes/1e9:.1f} GB)")

# precompute B(y) for every OG: union of frozen BP of member accessions (all our AC-set members)
og_bp = {}
for og, accs in og_members.items():
    s = frozenset()
    for a in accs:
        s = s | bp_prop(a)
    if s:
        og_bp[og] = s
log(f"OGs with non-empty partner BP set: {len(og_bp):,}")


def neighbours(og_x_indices, exclude_ogs):
    """Return list of (og_id, sim) top-N over informative candidate OGs, excluding exclude_ogs and
    the target's own OG indices. Aggregates across multiple target OGs by max sim."""
    best = {}
    for xi in og_x_indices:
        v = Dn[xi]
        if not np.any(v):        # zero-variance target profile: cannot discriminate
            continue
        sims = cand_norm @ v      # (nCand,)
        order = np.argsort(-sims)[:TOPN_NEIGH * 3]
        for j in order:
            s = float(sims[j])
            if s < SIM_MIN:
                break
            og = ogs[cand_idx[j]]
            if og in exclude_ogs:
                continue
            if s > best.get(og, -1):
                best[og] = s
    return sorted(best.items(), key=lambda kv: -kv[1])[:TOPN_NEIGH]


ARMS = ["phylo"]
results, coverage, strat = {}, {}, {}

for cell, gt_fn in [("LK-BPO", "groundtruth_LK.tsv"), ("PK-BPO", "groundtruth_PK.tsv")]:
    log(f"===== {cell} =====")
    gt = pd.read_csv(GTDIR / gt_fn, sep="\t")
    gt = gt[gt.aspect == "P"].copy()
    gt["term"] = gt.term.map(lambda g: alt.get(g, g))
    gt_leaf = gt.groupby("EntryID").term.apply(set).to_dict()
    targets = sorted(gt_leaf); target_set = set(targets)
    pool = load_pool(target_set)
    gt_prop = {p: frozenset().union(*[anc(t) for t in ts]) if ts else frozenset() for p, ts in gt_leaf.items()}
    pool_prop = {p: (frozenset().union(*[anc(t) for t in pool[p]]) if pool.get(p) else frozenset()) for p in targets}
    tp_pool = sum(iamass(gt_prop[p] & pool_prop[p]) for p in targets)
    fn_total = sum(iamass(gt_prop[p] - pool_prop[p]) for p in targets)
    gt_total = sum(iamass(gt_prop[p]) for p in targets)
    log(f"  {len(targets)} targets; gt IA {gt_total:.1f}; pool-captured {100*tp_pool/gt_total:.1f}%; FN {100*fn_total/gt_total:.1f}%")

    # ---- build phylo scores ----
    score = {}          # p -> {term: summed sim over neighbour OGs carrying term}
    neigh_count = {}
    n_mapped = 0
    for p in targets:
        ogx = acc2og.get(p, [])
        if not ogx:
            continue
        n_mapped += 1
        xi = [og_idx[o] for o in ogx if o in og_idx]
        exclude = set(ogx)
        nb = neighbours(xi, exclude)
        neigh_count[p] = len(nb)
        sd = collections.defaultdict(float)
        for og_y, s in nb:
            By = og_bp.get(og_y)
            if not By:
                continue
            # drop self accession from the partner set (recompute B excluding p if p is a member)
            if p in og_members.get(og_y, ()):
                By = frozenset().union(*[bp_prop(a) for a in og_members[og_y] if a != p]) or frozenset()
            for t in By:
                sd[t] += s
        if sd:
            score[p] = sd
    log(f"  targets mapped to >=1 OG: {n_mapped}; with >=1 phylo proposal: {len(score)}")

    # ---- added true/false IA-mass  [VERBATIM CG.3] ----
    KMAX = max(TOPKS)
    cell_res = {
        "target_proteins": len(targets), "gt_true_IA_mass": round(gt_total, 2),
        "pool_captured_IA_mass": round(tp_pool, 2), "pool_captured_frac": round(tp_pool / gt_total, 4),
        "FN_IA_mass_pool_misses": round(fn_total, 2), "FN_frac_of_gt": round(fn_total / gt_total, 4),
        "arms": {},
    }
    per_prot_top5 = {}
    arm = "phylo"
    added_true = {k: 0.0 for k in TOPKS}; added_false = {k: 0.0 for k in TOPKS}; n_prop = {k: 0 for k in TOPKS}
    n_scored = 0
    for p in targets:
        sd = score.get(p)
        if not sd: continue
        n_scored += 1
        present = pool.get(p, set()); gtp = gt_prop[p]; covered = set(pool_prop[p])
        picks = 0; p_true5 = 0.0; p_false5 = 0.0
        for g, sv in sorted(sd.items(), key=lambda kv: -kv[1]):
            if sv <= 0: break
            if g in present: continue
            picks += 1
            new = anc(g) - covered
            tm = iamass(new & gtp); fm = iamass(new - gtp)
            covered |= new
            for k in TOPKS:
                if picks <= k:
                    added_true[k] += tm; added_false[k] += fm; n_prop[k] += 1
            if picks <= 5: p_true5 += tm; p_false5 += fm
            if picks >= KMAX: break
        per_prot_top5[p] = (p_true5, p_false5)
    arm_res = {"targets_scored": n_scored, "per_k": {}}
    for k in TOPKS:
        at, af = added_true[k], added_false[k]
        arm_res["per_k"][f"top{k}"] = {
            "proposals": n_prop[k], "true_IA_mass_added": round(at, 3), "false_IA_mass_added": round(af, 3),
            "generation_precision_IA": round(at / (at + af), 4) if (at + af) > 0 else None,
            "frac_of_FN_recovered": round(at / fn_total, 4) if fn_total > 0 else None,
        }
        rr = arm_res["per_k"][f"top{k}"]
        log(f"    top{k:<2}: +true {at:8.2f} +false {af:10.2f} prec {rr['generation_precision_IA']} FNrec {rr['frac_of_FN_recovered']}")
    cell_res["arms"][arm] = arm_res

    # stratification by taxon (top5)
    acc_tax = pd.read_csv(ROOT / "storage/regen_headline/cg3_acc_taxon.tsv", sep="\t", header=None,
                          names=["accession", "taxon"], dtype=str)
    tax_of = dict(zip(acc_tax.accession, acc_tax.taxon))
    by_tax = collections.defaultdict(lambda: [0.0, 0.0, 0])
    for p, (tr, fa) in per_prot_top5.items():
        tx = tax_of.get(p, "NA"); by_tax[tx][0] += tr; by_tax[tx][1] += fa; by_tax[tx][2] += 1
    strat[cell] = {"by_taxon_top5": {tx: {"true": round(v[0], 2), "false": round(v[1], 2),
                    "prec": round(v[0] / (v[0] + v[1]), 4) if (v[0] + v[1]) > 0 else None, "n": v[2]}
                    for tx, v in sorted(by_tax.items(), key=lambda x: -x[1][2])[:15]}}
    coverage[cell] = {"targets": len(targets), "targets_mapped_to_OG": n_mapped,
                      "targets_with_proposal": len(score),
                      "coverage_frac": round(len(score) / len(targets), 4),
                      "median_neighbours": float(np.median(list(neigh_count.values()))) if neigh_count else 0}
    results[cell] = cell_res

out = {"slice": "PHYLO", "orthodb": "v12.2 Eukaryota-2759", "topN_neigh": TOPN_NEIGH, "sim_min": SIM_MIN,
       "prev_band": [MIN_PREV, MAX_PREV_FRAC], "results": results, "coverage": coverage, "stratification": strat,
       "bars": {"cooccurrence_top5": {"LK-BPO": 0.0161, "PK-BPO": 0.0103},
                "network_CG3_top5": {"LK-BPO": 0.0578, "PK-BPO": 0.0241}, "classifier": 0.1159}}
json.dump(out, open(OUT / "phylo_step1_gate.json", "w"), indent=2)
log("wrote phylo_step1_gate.json")
print(json.dumps(out, indent=2))
