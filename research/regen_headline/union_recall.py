"""MAXIMAL-UNION BP RECALL CEILING (READ-ONLY, frozen).

Question: is the missing true BP-tail (propagated-true BP terms NOT in the deployed pool) REACHABLE
from ANY t0 retrieval/generation signal we have, or fundamentally un-retrievable?

Target (board frame, matches bp_failure_atlas): per LK/PK-BPO target with >=1 valid true BP term,
  gt_valid = propagated-true BP toi terms (PK: minus t0-known, propagated);
  pool     = propagated eval.parquet BP candidates (the deployed pool);
  missing  = gt_valid - pool  (the unreachable tail; IA-weighted).
Cross-checked against bp_failure_atlas.json recall_fail_ia_not_in_pool + pct unreachable.

Sources (each leakage-clean, t0; proposal = propagated BP toi set per protein):
  kNN_<plm>   8-PLM sequence kNN over t0 (v227) refs, neighbours' t0 BP annotations transferred.
  classifier  full-BP-vocab classifier top-50 extras not in pool (selgen_cand, v227-trained).
  network     STRING v12.0 (<<t0) partners' frozen t0 BP annotations (clean=exp+coexp).
  literature  abstract->GO top-50 (S-PubMedBert, pre-t0 PMIDs).

Headline = IA-weighted recall of the missing tail reached by the maximal union, per cell; per-source
marginal; residual unreachable IA-mass and its character. Precision context = proposed-term volume
and false IA added. Set membership only (no cafaeval).
"""
import json, collections, time, pickle
from pathlib import Path
from collections import deque
import numpy as np, pandas as pd, pyarrow.parquet as pq

t0 = time.time()
ROOT = Path("/home/frapercan/Thesis2")
W = ROOT / "storage/regen_headline"; SC = W / "recall_scratch"
R = ROOT / "repositories/protea-reranker-lab/results/sparse_classifier"
DS = ROOT / "repositories/protea-reranker-lab/datasets/protst-global-train227-test230"
EVAL = R / "percut_rerank/eval.parquet"
GTDIR = R / "lafa_gt"
OBO = ROOT / "protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
IA_F = ROOT / "protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv"
TOI_FILE = str(GTDIR / "groundtruth_terms_of_interest.txt")
def log(m): print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)

# ---------- ontology ----------
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
_anc = {}
def anc(t):
    t = alt.get(t, t)
    if t in _anc: return _anc[t]
    seen, st = set(), [t]
    while st:
        x = st.pop()
        if x in seen: continue
        seen.add(x)
        for p in par.get(x, ()): st.append(p)
    r = frozenset(x for x in seen if x in BP); _anc[t] = r; return r
IA = {}
for line in open(IA_F):
    p = line.rstrip("\n").split("\t")
    if len(p) >= 2:
        try: IA[p[0]] = float(p[1])
        except ValueError: pass
toi_go = sorted({g.strip() for g in open(TOI_FILE) if g.strip() in BP})
toi_idx = {g: i for i, g in enumerate(toi_go)}
ntoi = len(toi_go)
ia_toi = np.array([IA.get(g, 0.0) for g in toi_go], dtype=np.float64)
def toiset(terms):
    s = set()
    for t in terms:
        j = toi_idx.get(alt.get(t, t))
        if j is not None: s.add(j)
    return s
def prop_toiset(leaf_terms):
    s = set()
    for t in leaf_terms:
        for a in anc(t):
            j = toi_idx.get(a)
            if j is not None: s.add(j)
    return s
# DAG depth over toi
depth_go = {}
roots = [t for t in BP if not (par.get(t, set()) & BP)]
dq = deque((r, 0) for r in roots)
children = collections.defaultdict(set)
for t in BP:
    for p in par.get(t, ()):
        if p in BP: children[p].add(t)
seen = set()
while dq:
    u, d = dq.popleft()
    if u in seen: continue
    seen.add(u); depth_go[u] = d
    for c in children[u]: dq.append((c, d + 1))
depth_toi = np.array([depth_go.get(g, -1) for g in toi_go], dtype=np.int32)
log(f"BP {len(BP)}; toi {ntoi}; IA loaded {len(IA)}")

# ---------- pool (propagated) per protein ----------
tb = pq.read_table(EVAL, columns=["protein_accession", "go_term_id"]).to_pandas()
tb["go"] = tb.go_term_id.map(lambda g: alt.get(g, g))
tb = tb[tb.go.isin(BP)]
pool_leaf = tb.groupby("protein_accession").go.apply(set).to_dict()
log(f"pool proteins {len(pool_leaf)}")

# ---------- gt + known ----------
def read_tsv_P(fn, has_header):
    d = collections.defaultdict(set)
    with open(fn) as fh:
        if has_header: next(fh)
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) >= 3 and f[2] == "P":
                d[f[0]].add(alt.get(f[1], f[1]))
    return d
gt_lk = read_tsv_P(GTDIR / "groundtruth_LK.tsv", True)
# PK gt from DS/gt_pk_bp.tsv (2-col, atlas source) -> BP already
gt_pk = collections.defaultdict(set)
with open(DS / "gt_pk_bp.tsv") as fh:
    for line in fh:
        f = line.rstrip("\n").split("\t")
        if len(f) >= 2: gt_pk[f[0]].add(alt.get(f[1], f[1]))
known_pk = read_tsv_P(GTDIR / "groundtruth_PK_known.tsv", True)

CELLS = {"lk": {"gt": gt_lk, "known": None}, "pk": {"gt": gt_pk, "known": known_pk}}

# ---------- source proposals ----------
def load_pkl(fn):
    p = SC / fn
    return pickle.load(open(p, "rb")) if p.exists() else {}
PLMS = ["ankh_base", "ankh_large", "esm2_150m", "esm2_650m", "esm2_3b", "esmc_600m", "learned_champion"]
knn_src = {f"kNN_{p}": load_pkl(f"knn_prop_{p}.pkl") for p in PLMS}
net_src = load_pkl("net_prop_clean.pkl")
lit_src = load_pkl("lit_prop_top50.pkl")
# classifier extras from selgen_cand
clf_src = {}
for cell in ("lk", "pk"):
    a = np.load(W / f"selgen_cand_{cell}.npy", allow_pickle=True)
    by = collections.defaultdict(set)
    for rec in a:
        by[str(rec["p"])].add(str(rec["g"]))
    for p, gos in by.items():
        clf_src[p] = np.fromiter(prop_toiset(gos), dtype=np.int32, count=len(prop_toiset(gos)))
log(f"sources loaded: knn {[ (k,len(v)) for k,v in knn_src.items()]}; net {len(net_src)}; "
    f"lit {len(lit_src)}; clf {len(clf_src)}")

# proposal order (retrieval diversity first, then generators)
SRC_ORDER = [f"kNN_{p}" for p in PLMS] + ["classifier", "network", "literature"]
def get_src(name):
    if name.startswith("kNN_"): return knn_src[name]
    if name == "classifier": return clf_src
    if name == "network": return net_src
    if name == "literature": return lit_src
    return {}

OUT = {"frame": "board (-known PK), toi-restricted, IA-weighted set membership; refs+annotations v227=t0",
       "sources": SRC_ORDER, "plms_used": PLMS,
       "plms_absent_t0clean": ["prot_t5", "prostt5", "protst_text (no c905dffa t0 ref materialized)"],
       "cells": {}}

for cell, cfg in CELLS.items():
    gt = cfg["gt"]; known = cfg["known"]
    # per-protein valid-gt toi set, pool toi set, missing
    universe = json.load(open(W / f"bp_atlas_perprotein_{cell}.json"))
    accs = [r["protein"] for r in universe]
    rows = {}
    miss_terms_iamass = collections.defaultdict(float)  # toi idx -> summed IA over proteins (missing)
    tot_gt = tot_pool_cap = tot_missing = 0.0
    n_fully_unreach = 0
    per_src_cover = {s: 0.0 for s in SRC_ORDER}          # standalone: missing IA reached by source alone
    union_cover = 0.0
    marginal = {s: 0.0 for s in SRC_ORDER}               # ordered marginal
    residual_terms = collections.defaultdict(float)      # toi idx -> IA (missing, reached by nobody)
    resid_depth_ia = collections.defaultdict(float)
    resid_iaband_ia = collections.defaultdict(float)
    union_size_terms = 0                                  # total proposed (protein,term) cells in union
    union_false_ia = 0.0                                  # proposed cells that are NOT true (over gt_valid)
    proposed_cells = 0
    for r in universe:
        p = r["protein"]
        gv = prop_toiset(gt.get(p, set()))
        if known is not None and known.get(p):
            gv -= prop_toiset(known[p])
        if not gv:
            continue
        pl = prop_toiset(pool_leaf.get(p, set())) & set(range(ntoi))
        gv_arr = np.fromiter(gv, dtype=np.int64, count=len(gv))
        pool_hit = gv & pl
        missing = gv - pl
        gt_ia = ia_toi[gv_arr].sum()
        tot_gt += gt_ia
        tot_pool_cap += ia_toi[np.fromiter(pool_hit, np.int64, len(pool_hit))].sum() if pool_hit else 0.0
        if not missing:
            continue
        m_arr = np.fromiter(missing, np.int64, len(missing))
        tot_missing += ia_toi[m_arr].sum()
        if not pool_hit:
            n_fully_unreach += 1
        for j in missing:
            miss_terms_iamass[j] += ia_toi[j]
        # source coverage of THIS protein's missing tail
        covered_union = set()
        for s in SRC_ORDER:
            prop = get_src(s).get(p)
            if prop is None:
                continue
            ps = set(int(x) for x in prop)
            hitm = missing & ps
            if hitm:
                per_src_cover[s] += ia_toi[np.fromiter(hitm, np.int64, len(hitm))].sum()
            # ordered marginal
            new = (missing & ps) - covered_union
            if new:
                marginal[s] += ia_toi[np.fromiter(new, np.int64, len(new))].sum()
            covered_union |= (missing & ps)
        if covered_union:
            union_cover += ia_toi[np.fromiter(covered_union, np.int64, len(covered_union))].sum()
        resid = missing - covered_union
        for j in resid:
            residual_terms[j] += ia_toi[j]
            resid_depth_ia[int(depth_toi[j])] += ia_toi[j]
            resid_iaband_ia[min(int(ia_toi[j]), 10)] += ia_toi[j]
        # precision context: total union proposed cells (any source) intersect gv for true, else false
        all_prop = set()
        for s in SRC_ORDER:
            prop = get_src(s).get(p)
            if prop is not None: all_prop |= set(int(x) for x in prop)
        # restrict to terms not already in pool (new proposals)
        newprop = all_prop - pl
        proposed_cells += len(newprop)
        if newprop:
            npa = np.fromiter(newprop, np.int64, len(newprop))
            true_mask = np.array([j in gv for j in newprop])
            union_false_ia += ia_toi[npa[~true_mask]].sum() if (~true_mask).any() else 0.0

    n = len(accs)
    recall_pool = tot_pool_cap / tot_gt if tot_gt else 0.0
    recall_union_of_missing = union_cover / tot_missing if tot_missing else 0.0
    recall_total_union = (tot_pool_cap + union_cover) / tot_gt if tot_gt else 0.0
    residual_ia = tot_missing - union_cover
    OUT["cells"][cell + "-bpo"] = {
        "n_proteins": n,
        "gt_valid_IA_total": round(tot_gt, 1),
        "pool_captured_IA": round(tot_pool_cap, 1),
        "pool_recall_of_total_true": round(recall_pool, 4),
        "missing_tail_IA": round(tot_missing, 1),
        "missing_tail_frac_of_gt": round(tot_missing / tot_gt, 4) if tot_gt else None,
        "n_proteins_fully_unreachable_by_pool": n_fully_unreach,
        "pct_proteins_fully_unreachable": round(100 * n_fully_unreach / n, 2),
        "HEADLINE_maximal_union_recall_of_missing_tail": round(recall_union_of_missing, 4),
        "maximal_union_recall_of_total_true": round(recall_total_union, 4),
        "residual_unreachable_IA": round(residual_ia, 1),
        "residual_frac_of_missing_tail": round(residual_ia / tot_missing, 4) if tot_missing else None,
        "residual_frac_of_total_gt": round(residual_ia / tot_gt, 4) if tot_gt else None,
        "per_source_standalone_recall_of_missing": {s: round(per_src_cover[s] / tot_missing, 4) if tot_missing else None for s in SRC_ORDER},
        "per_source_marginal_recall_ordered": {s: round(marginal[s] / tot_missing, 4) if tot_missing else None for s in SRC_ORDER},
        "precision_context": {
            "union_new_proposed_cells": proposed_cells,
            "union_false_IA_added_over_gtvalid": round(union_false_ia, 1),
            "note": "false IA counts proposed terms outside gt_valid; separability wall, not the headline",
        },
        "residual_character": {
            "DAG_depth_IA_share": {str(k): round(v / residual_ia, 4) for k, v in sorted(resid_depth_ia.items())} if residual_ia else {},
            "IA_band_IA_share": {str(k): round(v / residual_ia, 4) for k, v in sorted(resid_iaband_ia.items())} if residual_ia else {},
        },
    }
    log(f"{cell}: pool-recall {recall_pool:.3f} | missing {tot_missing:.0f} | "
        f"UNION recall-of-missing {recall_union_of_missing:.3f} | residual {residual_ia:.0f} "
        f"({100*residual_ia/tot_missing:.1f}% of missing)")

json.dump(OUT, open(W / "RECALL_CEILING_MAXIMAL_UNION.json", "w"), indent=2)
log("wrote RECALL_CEILING_MAXIMAL_UNION.json")
print(json.dumps(OUT, indent=2))
