"""Stage 2/3 evaluation: trained contrastive aligner (arm A EXP, arm B EXP+IEA) vs classifier,
through the EXACT tax discipline of seq2ann_aligner_ceiling.py, PLUS taxonomic-lineage stratification.
Q_TAX / Q_TAIL / Q_RESIDUAL / fixed-score control; near-taxon vs far-taxon slices; EXP/IEA transfer.
Read-only w.r.t. repos.
"""
import json, collections, time
from pathlib import Path
import numpy as np, pyarrow.parquet as pq

t0 = time.time()
W = Path("/home/frapercan/Thesis2/storage/cooc_experiment/seq2ann_infonce")
CW = Path("/home/frapercan/Thesis2/storage/cooc_experiment")
REL = Path("/home/frapercan/Thesis2/CAFA_forever/data/releases/Sep_2025_Mar_2026")
T0D = Path("/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025")
FROZEN = Path("/home/frapercan/Thesis2/storage/protea-frozen-v227-2025-09-04")
SC = Path("/home/frapercan/Thesis2/repositories/protea-reranker-lab/results/sparse_classifier")
OBO = str(T0D / "go-basic.obo"); IA_F = str(T0D / "IA.tsv")
V_MATCH = 113000
CLS_ANCHOR = (6332.0, 99966.0)

# ---- ontology / IA (verbatim) ----
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
    for g in terms:
        o.add(g); o |= anc(g)
    return o
IA = {}
for line in open(IA_F):
    p = line.rstrip("\n").split("\t")
    if len(p) >= 2:
        try: IA[p[0]] = float(p[1])
        except ValueError: pass
def iaw(terms): return float(sum(IA.get(g, 0.0) for g in terms))

# ---- gt / known / pools / classifier (verbatim from ceiling) ----
def load_gt(fn):
    G = collections.defaultdict(set)
    with (REL / fn).open() as fh:
        next(fh)
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) >= 3 and f[2] == "P": G[f[0]].add(alt.get(f[1], f[1]))
    return {p: {g for g in closure(ts) if g in BP} for p, ts in G.items()}
gt_pk = load_gt("groundtruth_PK.tsv"); gt_lk = load_gt("groundtruth_LK.tsv")
pk_known_bp = collections.defaultdict(set)
with (REL / "groundtruth_PK_known.tsv").open() as fh:
    next(fh)
    for line in fh:
        f = line.rstrip("\n").split("\t"); g = alt.get(f[1], f[1])
        if f[2] == "P": pk_known_bp[f[0]] |= {x for x in closure([g]) if x in BP}
meta = pq.read_table(FROZEN / "go_term_metadata.parquet")
id2go = {i: g for i, g in zip(meta.column("go_term_id").to_pylist(), meta.column("go_id").to_pylist())}
esc = pq.read_table(CW / "rerank_out" / "eval_scores.parquet")
ecat = np.asarray(esc.column("category").to_pylist()); easp = np.asarray(esc.column("aspect").to_pylist())
def knn_pool(cell):
    m = (ecat == cell) & (easp == "bpo")
    P = np.asarray(esc.column("protein_accession").to_pylist())[m]
    Gr = np.asarray(esc.column("go_term_id").to_pylist())[m]
    d = collections.defaultdict(set)
    for p_, g_ in zip(P, Gr): d[p_].add(alt.get(g_, g_))
    return d
knn_pk = knn_pool("pk"); knn_lk = knn_pool("lk")
uc = np.load(CW / "union_candidates_scored.npz", allow_pickle=True)
uc_keep = (uc["src"] == "cls") | (uc["src"] == "both")
uc_prot = uc["prot"][uc_keep]; uc_term = np.array([alt.get(g, g) for g in uc["term"][uc_keep]]); uc_score = uc["score"][uc_keep]
cls_pk = collections.defaultdict(set)
for p_, g_ in zip(uc_prot, uc_term): cls_pk[p_].add(g_)
print(f"[{time.time()-t0:.0f}s] gt_pk {len(gt_pk):,} gt_lk {len(gt_lk):,} knn_pk {len(knn_pk):,}", flush=True)

# ---- taxon: eval lineage (uniprot cache) + train/eval taxid (GAF stream) ----
lin = json.load(open(W / "eval_lineage.json"))            # acc -> {taxid,lineage,sci}
acc2taxid = {}
gaf = W / "acc2taxid_gaf.tsv"
if gaf.exists():
    for line in open(gaf):
        p = line.rstrip("\n").split("\t")
        if len(p) == 2 and p[1].isdigit(): acc2taxid[p[0]] = int(p[1])
tr_accs = np.load(SC / "clf_protein_codes.npz", allow_pickle=True)["accs"].tolist()
ev_accs = np.load(SC / "eval_protein_codes.npz", allow_pickle=True)["accs"].tolist()
# species density: # train proteins sharing eval protein's taxid (excl self)
train_tax_count = collections.Counter(acc2taxid[a] for a in tr_accs if a in acc2taxid)
def eval_taxid(a):
    if a in acc2taxid: return acc2taxid[a]
    if a in lin: return lin[a]["taxid"]
    return None
def species_density(a):
    tid = eval_taxid(a)
    if tid is None: return None
    n = train_tax_count.get(tid, 0)
    if a in acc2taxid and acc2taxid[a] == tid: n -= 1     # exclude self
    return max(n, 0)
def superkingdom(a):
    return lin[a]["lineage"][0] if a in lin and lin[a]["lineage"] else None
gaf_cov_ev = sum(1 for a in ev_accs if a in acc2taxid)
gaf_cov_tr = sum(1 for a in tr_accs if a in acc2taxid)
print(f"[{time.time()-t0:.0f}s] taxid: GAF eval-cov {gaf_cov_ev}/{len(ev_accs)} train-cov {gaf_cov_tr}/{len(tr_accs)}; lineage eval-cov {len(lin)}", flush=True)

# ---- measure (verbatim logic from ceiling; returns Q_TAX/Q_TAIL/Q_RESIDUAL) ----
BUCKETS = [(0.0, 2.0, "IA[0,2) common"), (2.0, 4.0, "IA[2,4)"), (4.0, 6.0, "IA[4,6)"),
           (6.0, 8.0, "IA[6,8)"), (8.0, 1e9, "IA>=8 rare/deep")]
def bucket_of(w):
    for i, (lo, hi, _) in enumerate(BUCKETS):
        if lo <= w < hi: return i
    return len(BUCKETS) - 1
def measure(prots, gt, known_bp, knn, cls_extras, cand_iter):
    per = {}
    for p in prots:
        gtc = gt[p]; kbp = known_bp.get(p, set())
        bothc = closure(knn.get(p, set()) | cls_extras.get(p, set())) & BP
        knnc = closure(knn.get(p, set())) & BP
        novel = gtc - kbp
        per[p] = (gtc, knnc, bothc, novel)
    residual_total = sum(iaw(per[p][3] - per[p][2]) for p in prots)
    rows = []; seen = set()
    for prot, term, sc in cand_iter:
        if prot not in per or term not in BP: continue
        gtc, knnc, bothc, novel = per[prot]
        if term in knnc: continue
        key = (prot, term)
        if key in seen: continue
        seen.add(key)
        w = IA.get(term, 0.0); is_true = term in gtc
        in_both = term in bothc; in_res = is_true and (term in novel) and (not in_both)
        rows.append((sc, w, is_true, in_both, in_res, bucket_of(w)))
    if not rows: return None
    A = np.array(rows, np.float64); A = A[np.argsort(-A[:, 0])]
    V = min(V_MATCH, len(A)); top = A[:V]
    def ratio(mat):
        w = mat[:, 1]; tt = float((w * mat[:, 2]).sum()); ff = float((w * (1 - mat[:, 2])).sum())
        return {"n": int(len(mat)), "true_ia": round(tt, 1), "false_ia": round(ff, 1),
                "ratio": round(tt / ff, 4) if ff else None,
                "true_cnt": int(mat[:, 2].sum()), "false_cnt": int((1 - mat[:, 2]).sum())}
    out = {"n_prots": len(prots), "cand_universe": int(len(A)), "V_used": int(V),
           "residual_total_ia": round(residual_total, 1), "Q_TAX_at_V": ratio(top)}
    buckets = {}
    for i, (lo, hi, name) in enumerate(BUCKETS):
        m = top[top[:, 5] == i]
        buckets[name] = ratio(m) if len(m) else {"n": 0, "ratio": None, "true_ia": 0.0, "false_ia": 0.0, "true_cnt": 0, "false_cnt": 0}
    out["Q_TAIL_buckets"] = buckets
    res_reached = float((top[:, 1] * top[:, 4]).sum()); false_cost = float((top[:, 1] * (1 - top[:, 2])).sum())
    out["Q_RESIDUAL"] = {"residual_total_ia": round(residual_total, 1),
                         "residual_reached_ia": round(res_reached, 1),
                         "residual_reached_frac": round(res_reached / residual_total, 4) if residual_total else None,
                         "false_ia_cost": round(false_cost, 1)}
    res_full = float((A[:, 1] * A[:, 4]).sum())
    out["Q_RESIDUAL_full_universe_frac"] = round(res_full / residual_total, 4) if residual_total else None
    return out

# ---- candidate iterators ----
def arm_iter_factory(arm):
    d = np.load(W / f"eval_scores_{arm}.npz", allow_pickle=True)
    P = d["prot"]; Tm = np.array([alt.get(g, g) for g in d["term"]]); Sc = d["score"]
    def it(prots):
        ps = set(prots)
        for p_, t_, s_ in zip(P, Tm, Sc):
            if p_ in ps: yield p_, t_, float(s_)
    return it
def classifier_iter_factory():
    def it(prots):
        ps = set(prots)
        for p_, t_, s_ in zip(uc_prot, uc_term, uc_score):
            if p_ in ps: yield p_, t_, float(s_)
    return it

# ---- fixed-score control: score arm-A candidate ROWS with arm-B scores replaced by a FIXED constant,
#      and vice versa, to test whether cross-arm Q_TAX differences are real ranking or volume artifacts.
def fixed_score_iter(arm, prots):
    """Yield arm's candidate (prot,term) but with score = constant 1.0 (rank-random within, volume test)."""
    d = np.load(W / f"eval_scores_{arm}.npz", allow_pickle=True)
    P = d["prot"]; Tm = np.array([alt.get(g, g) for g in d["term"]])
    ps = set(prots)
    rng = np.random.default_rng(0)
    for p_, t_ in zip(P, Tm):
        if p_ in ps: yield p_, t_, float(rng.random())

armA_it = arm_iter_factory("armA"); armB_it = arm_iter_factory("armB"); cls_it = classifier_iter_factory()

# ---- taxon strata over PK proteins (richest pool) ----
def build_strata(prots):
    strata = {"ALL": list(prots)}
    # species density
    dens = {p: species_density(p) for p in prots}
    known = [p for p in prots if dens[p] is not None]
    strata["taxdensity_dense_ge100"] = [p for p in known if dens[p] >= 100]
    strata["taxdensity_sparse_1to99"] = [p for p in known if 1 <= dens[p] < 100]
    strata["taxdensity_singleton_0"] = [p for p in known if dens[p] == 0]
    strata["taxdensity_FAR_lt100"] = [p for p in known if dens[p] < 100]  # decisive far slice
    # superkingdom
    strata["euk"] = [p for p in prots if superkingdom(p) == "Eukaryota"]
    strata["nonEuk_FAR"] = [p for p in prots if superkingdom(p) in ("Bacteria", "Archaea", "Viruses")]
    return strata

res = {"scope": "BP; trained contrastive InfoNCE aligner (frozen d8979601 + BioBERT text towers, "
                "MLP proj + temperature, full-vocab multi-positive IA-weighted). arm A=EXP-only, arm B=EXP+IEA. "
                "novel = gt(v230) closure - v227 known closure. Q_TAX at matched V=113k, marginal over kNN.",
       "classifier_anchor_ratio": round(CLS_ANCHOR[0] / CLS_ANCHOR[1], 4),
       "taxid_coverage": {"gaf_eval": gaf_cov_ev, "gaf_train": gaf_cov_tr, "eval_total": len(ev_accs),
                          "lineage_eval": len(lin)},
       "cells": {}}

pk_prots = sorted(set(gt_pk) & set(knn_pk))
lk_prots = sorted(set(gt_lk) & set(knn_lk))

for cell, prots, gt, kbp, knn, cls in [
        ("PK_BP", pk_prots, gt_pk, pk_known_bp, knn_pk, cls_pk),
        ("LK_BP", lk_prots, gt_lk, collections.defaultdict(set), knn_lk, collections.defaultdict(set))]:
    strata = build_strata(prots) if cell == "PK_BP" else {"ALL": list(prots)}
    cellout = {"strata_sizes": {k: len(v) for k, v in strata.items()}, "strata": {}}
    for sname, sprots in strata.items():
        if len(sprots) < 5:
            cellout["strata"][sname] = {"skip_small": len(sprots)}; continue
        entry = {}
        entry["ALIGNER_armA_EXP"] = measure(sprots, gt, kbp, knn, cls, armA_it(sprots))
        entry["ALIGNER_armB_EXPIEA"] = measure(sprots, gt, kbp, knn, cls, armB_it(sprots))
        entry["CLASSIFIER"] = measure(sprots, gt, kbp, knn, cls, cls_it(sprots))
        # fixed-score control only on ALL (expensive-ish)
        if sname == "ALL":
            entry["armA_FIXED_score_control"] = measure(sprots, gt, kbp, knn, cls, fixed_score_iter("armA", sprots))
            entry["armB_FIXED_score_control"] = measure(sprots, gt, kbp, knn, cls, fixed_score_iter("armB", sprots))
        cellout["strata"][sname] = entry
        print(f"[{time.time()-t0:.0f}s] {cell}/{sname} ({len(sprots)}) done", flush=True)
    res["cells"][cell] = cellout

json.dump(res, open(W / "seq2ann_infonce_eval.json", "w"), indent=1)
print(f"[{time.time()-t0:.0f}s] WROTE seq2ann_infonce_eval.json", flush=True)
