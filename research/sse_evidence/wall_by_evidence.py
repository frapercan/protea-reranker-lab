"""The BP wall characterized BY evidence (analysis-only; test-side evidence NOT on disk -> v225 proxies).

For lk-bpo and pk-bpo (board -known frame): take the TRUE tail (propagated true v227->v230 terms, TOI, PK
-known), split into DELIVERED vs MISSING (deployed predictions file) and REACHABLE vs UNREACHABLE (deployed
pool). Overlay evidence THREE ways, IA-weighted:
  (1) term-level v225 corpus evidence profile (term_evidence_direct.npz): each true term classed EXP-dominant
      / IEA-dominant / mixed / absent-in-v225 -> is the unreachable mass experimental-type or electronic-type biology.
  (2) donor evidence (eval.parquet evidence_code = the KNN NEIGHBOR's pre-cutoff evidence, provenance-confirmed):
      for REACHABLE true terms, the tier of the donor that carried it into the pool (reachable slice only).
  (3) per-protein t0 characterisation quality (calib_features): are unreachable-carrying proteins
      experimentally characterised at t0 or IEA-only.
NOTE: the v227->v230 annotating evidence of each true term is NOT on disk (only the v225 GAF); (1)/(3) are
v225 proxies and (2) is donor (not annotation) evidence. Stated explicitly in the receipt.
"""
import json, time, collections
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)
ROOT = Path("/home/frapercan/Thesis2")
R = ROOT / "repositories/protea-reranker-lab/results/sparse_classifier"
EVAL = R / "percut_rerank/eval.parquet"; PREDDIR = R / "percut_rerank/predictions"; GTDIR = R / "lafa_gt"
OBO = str(ROOT / "protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo")
IA_F = str(ROOT / "protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv")
TOI = str(GTDIR / "groundtruth_terms_of_interest.txt")
OUT = ROOT / "storage/sse_evidence"

EXP = {"EXP","IDA","IPI","IMP","IGI","IEP","HTP","HDA","HMP","HGI","HEP"}
PHY = {"IBA","IBD","IKR","IRD"}; COMP = {"ISS","ISO","ISA","ISM","IGC","RCA"}
AUTH = {"TAS","NAS","IC"}; ELEC = {"IEA"}
def tier_of(ev):
    if ev == "" or ev is None: return "EMPTY_nonKNN"   # classifier/interpro-only or NULL-evidence neighbor
    if ev in EXP: return "EXP"
    if ev in PHY: return "PHY"
    if ev in COMP: return "COMP"
    if ev in AUTH: return "AUTH"
    if ev in ELEC: return "ELEC"
    return "OTHER"

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
    _anc[t] = seen; return seen
BPNS = frozenset(t for t, n in ns.items() if n == "biological_process")
def anc_bp(t): return frozenset(x for x in anc(t) if x in BPNS)

IA = {}
for line in open(IA_F):
    p = line.rstrip("\n").split("\t")
    if len(p) >= 2:
        try: IA[alt.get(p[0], p[0])] = float(p[1])
        except ValueError: pass
def iam(ts): return float(sum(IA.get(t, 0.0) for t in ts))
toi = set(alt.get(x, x) for x in open(TOI).read().split())

# term-level v225 evidence profile
tp = np.load(OUT / "term_evidence_direct.npz", allow_pickle=True)
tp_terms = [alt.get(t, t) for t in tp["terms"].tolist()]; tp_counts = tp["counts"]
prof = {t: tp_counts[i] for i, t in enumerate(tp_terms)}   # [EXP,PHY,COMP,AUTH,ELEC]
def term_class(t):
    c = prof.get(t)
    if c is None or c.sum() == 0: return "absent_v225"
    tot = c.sum(); ef = c[0] / tot; el = c[4] / tot
    if ef >= 0.5: return "exp_dominant"
    if el >= 0.5: return "iea_dominant"
    return "mixed"
def term_expfrac(t):
    c = prof.get(t)
    if c is None or c.sum() == 0: return None
    return float(c[0] / c.sum())
def term_elecfrac(t):
    c = prof.get(t)
    if c is None or c.sum() == 0: return None
    return float(c[4] / c.sum())

# per-protein t0 calibration quality
cf = np.load(OUT / "calib_features.npz", allow_pickle=True)
accs_all = json.load(open(ROOT / "storage/cooc_experiment/generator_frames/accs.json"))
q = {}
for i, r in enumerate(cf["acc_row"]):
    q[accs_all[r]] = dict(exp_frac=float(cf["exp_frac"][i]), has_exp=int(cf["has_exp"][i]), total=int(cf["total"][i]))

RES = {"note": "test-side (v227->v230) annotating evidence NOT on disk; term-level(1)+per-protein(3) are "
       "v225 corpus proxies, donor(2) is the KNN neighbor's pre-cutoff evidence (provenance-confirmed).",
       "cells": {}}

for cell in ["lk", "pk"]:
    gt = pd.read_csv(GTDIR / f"groundtruth_{cell.upper()}.tsv", sep="\t")
    gt = gt[gt.aspect == "P"]
    gt_leaf = gt.groupby("EntryID").term.apply(lambda s: set(alt.get(g, g) for g in s)).to_dict()
    known = {}
    if cell == "pk":
        kk = pd.read_csv(GTDIR / "groundtruth_PK_known.tsv", sep="\t"); kk = kk[kk.aspect == "P"]
        known = kk.groupby("EntryID").term.apply(lambda s: frozenset().union(
            *[anc_bp(alt.get(g, g)) for g in s]) if len(s) else frozenset()).to_dict()
    # deployed delivered predictions (the finalized output file = delivered set)
    dep = pd.read_csv(PREDDIR / cell / f"{cell}.tsv", sep="\t", header=None, names=["prot","term","score"])
    dep["term"] = dep.term.map(lambda g: alt.get(g, g))
    delivered = collections.defaultdict(set)
    for r in dep.itertuples():
        for a in anc_bp(r.term): delivered[r.prot].add(a)
    # pool candidates (reachable) + donor evidence for label==1
    tb = pq.read_table(EVAL, columns=["protein_accession","go_term_id","label","aspect","category","evidence_code"]).to_pandas()
    tb = tb[(tb.category == cell) & (tb.aspect == "bpo")].copy()
    tb["go"] = tb.go_term_id.map(lambda g: alt.get(g, g))
    pool = collections.defaultdict(set)
    for r in tb.itertuples():
        for a in anc_bp(r.go): pool[r.protein_accession].add(a)
    donor = {}   # (prot, leaf-go) -> evidence_code, for reachable-true attribution (leaf pool candidates)
    for r in tb[tb.label > 0].itertuples():
        donor[(r.protein_accession, r.go)] = r.evidence_code

    # aggregate over targets
    agg = {k: collections.Counter() for k in ["true","deliv","missing","reach","unreach"]}
    ia = {k: collections.Counter() for k in ["true","deliv","missing","reach","unreach"]}
    ia_tot = {k: 0.0 for k in ["true","deliv","missing","reach","unreach"]}
    exp_acc = {k: [0.0, 0.0] for k in ["missing","unreach"]}   # [sum IA*expfrac, sum IA over terms with profile]
    elec_acc = {k: [0.0, 0.0] for k in ["missing","unreach"]}
    donor_ia = collections.Counter(); donor_ia_tot = 0.0
    unreach_prot_q = []; deliv_prot_q = []
    targets = [p for p in gt_leaf if p in pool or p in delivered]
    n_with_unreach = 0; n_scored = 0
    for p in targets:
        tp_set = frozenset().union(*[anc_bp(t) for t in gt_leaf[p]]) if gt_leaf[p] else frozenset()
        tp_set = frozenset(t for t in tp_set if t in toi)
        if cell == "pk": tp_set = tp_set - known.get(p, frozenset())
        if not tp_set: continue
        n_scored += 1
        dv = delivered.get(p, set()); pl = pool.get(p, set())
        miss = tp_set - dv; reach = tp_set & pl; unre = tp_set - pl
        for tag, S in [("true", tp_set), ("deliv", tp_set & dv), ("missing", miss), ("reach", reach), ("unreach", unre)]:
            for t in S:
                w = IA.get(t, 0.0); cl = term_class(t)
                agg[tag][cl] += 1; ia[tag][cl] += w; ia_tot[tag] += w
        for tag, S in [("missing", miss), ("unreach", unre)]:
            for t in S:
                ef = term_expfrac(t); el = term_elecfrac(t); w = IA.get(t, 0.0)
                if ef is not None: exp_acc[tag][0] += w * ef; exp_acc[tag][1] += w
                if el is not None: elec_acc[tag][0] += w * el; elec_acc[tag][1] += w
        # donor evidence for reachable leaf true terms
        for t in gt_leaf[p]:
            if t in pl and (p, t) in donor:
                w = IA.get(t, 0.0); donor_ia[tier_of(donor[(p, t)])] += w; donor_ia_tot += w
        if unre:
            n_with_unreach += 1
            if p in q: unreach_prot_q.append(q[p]["exp_frac"])
        else:
            if p in q: deliv_prot_q.append(q[p]["exp_frac"])

    def pct(counter, tot): return {k: round(v / tot, 4) for k, v in counter.items()} if tot > 0 else {}
    cellres = {
        "n_targets_scored": n_scored,
        "n_with_unreachable": n_with_unreach,
        "IA_mass": {k: round(ia_tot[k], 1) for k in ia_tot},
        "term_class_IA_frac": {tag: pct(ia[tag], ia_tot[tag]) for tag in ia},
        "term_class_count_frac": {tag: pct(agg[tag], sum(agg[tag].values())) for tag in agg},
        "unreachable_IAweighted_mean_exp_frac": round(exp_acc["unreach"][0] / exp_acc["unreach"][1], 4) if exp_acc["unreach"][1] > 0 else None,
        "unreachable_IAweighted_mean_elec_frac": round(elec_acc["unreach"][0] / elec_acc["unreach"][1], 4) if elec_acc["unreach"][1] > 0 else None,
        "missing_IAweighted_mean_exp_frac": round(exp_acc["missing"][0] / exp_acc["missing"][1], 4) if exp_acc["missing"][1] > 0 else None,
        "missing_IAweighted_mean_elec_frac": round(elec_acc["missing"][0] / elec_acc["missing"][1], 4) if elec_acc["missing"][1] > 0 else None,
        "donor_evidence_reachable_IA_frac": pct(donor_ia, donor_ia_tot),
        "perprotein_t0_exp_frac_mean": {
            "unreachable_carrying": round(float(np.mean(unreach_prot_q)), 4) if unreach_prot_q else None,
            "fully_reachable": round(float(np.mean(deliv_prot_q)), 4) if deliv_prot_q else None,
            "n_unreach_prot_with_t0": len(unreach_prot_q), "n_reach_prot_with_t0": len(deliv_prot_q)},
    }
    RES["cells"][f"{cell}-bpo"] = cellres
    log(f"{cell}-bpo: unreachIA {ia_tot['unreach']:.0f}/{ia_tot['true']:.0f} "
        f"class {cellres['term_class_IA_frac']['unreach']}")
    json.dump(RES, open(OUT / "wall_by_evidence.json", "w"), indent=1, default=float)

json.dump(RES, open(OUT / "wall_by_evidence.json", "w"), indent=1, default=float)
log("WALL BY EVIDENCE DONE")
print(json.dumps(RES, indent=1, default=float))
