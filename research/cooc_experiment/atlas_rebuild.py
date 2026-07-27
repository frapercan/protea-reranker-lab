"""PKW.3 -- the per-protein error atlas, REBUILT on a real oracle. THIS IS THE DELIVERABLE.

atlas_build.py + atlas_controls.py established three corrections that force a rebuild:

 1. THE ORACLE WAS NOT AN ORACLE. It was built from eval.parquet's `label`, which is
    computed against the DIRECT annotations. cafaeval propagates gt and predictions
    (prop=fill), so a pool term with label=0 is still scored TP when it is an ancestor of
    a true term. 35,293 of 616,223 pool rows are exactly that. A ranker allowed to see the
    pool can put those on top, so they belong in the ceiling. Pool ceiling is 0.7519, not
    0.6077, and "unreachable" is 101 proteins, not 1,521.
 2. neighbor_min_distance is byte-identical to `distance` on every non-NaN pk/bpo row --
    a per-CANDIDATE distance, not a per-protein min. Its per-protein min is ~0 for all
    4,402 proteins (a PK protein always has a near-exact neighbour), so it is a constant
    and cannot carry a gradient. Neighbour identity is cut on the per-protein MEDIAN
    distance instead, which has real spread (q25 0.099 / q50 0.233 / q75 0.437).
 3. taxonomic_relation's vocabulary is {same, close, intermediate, distant, root-only, ""},
    never "same_species". The QUERY's taxon is absent from every frozen input.

GATE: the in-house sweep must reproduce the anchor 0.2131 @ tau=0.393 or the script aborts.
No transform is applied to any score anywhere.

ATTRIBUTION (stated because a per-protein f from a global-threshold metric is a choice):
f_micro_w at a fixed tau is a function of three POOLED sums TP_w/FP_w/FN_w, each a plain
sum over proteins of an IA-weighted mass. So the masses are exactly additive per protein
even though F is not linear in them. We report
  delivered_f_i / oracle_f_i = the protein's OWN IA-weighted F at the GLOBAL optimal tau;
  headroom_i = the MARGINAL move in the GLOBAL f_micro_w when this one protein alone is
               served by the pool oracle and every other protein is left as-is, tau held
               at the global optimum.
WHAT THIS HIDES: (a) headroom_i is marginal, not a share -- micro-F is nonlinear in the
pooled sums, so the headrooms do NOT sum to the global gap (they sum to 0.23 against a
0.39 gap); (b) tau is frozen at the global optimum, so a protein whose scores are well
ORDERED but badly CALIBRATED is charged for the miscalibration; (c) per-protein F at a
global tau is not the per-protein best F -- we are deliberately not letting each protein
pick its own threshold, because the deployed system cannot do that either.
"""
import json
import sys
from collections import Counter, deque
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from cafaeval.parser import obo_parser, gt_parser, pred_parser

W = Path("/home/frapercan/Thesis2/storage/cooc_experiment")
DS = Path("/home/frapercan/Thesis2/repositories/protea-reranker-lab/datasets/protst-global-train227-test230")
OBO = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
IA = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv"
KNOWN = "/home/frapercan/Thesis2/CAFA_forever/data/releases/Sep_2025_Mar_2026/groundtruth_PK_known.tsv"
NS = "biological_process"
TAU = np.arange(0.001, 1, 0.001)
out = {"what": "PKW.3 per-protein error atlas for PK-BPO, rebuilt on a gt-membership pool oracle"}

# ---------------------------------------------------------------- inputs
t = pq.read_table(W / "rerank_out" / "eval_scores.parquet",
                  columns=["protein_accession", "go_term_id", "category", "aspect", "reranker_score"])
ev = pq.read_table(DS / "eval.parquet",
                   columns=["protein_accession", "go_term_id", "category", "aspect", "label",
                            "distance", "length_query", "taxonomic_relation"])
cat = np.asarray(t.column("category").to_pylist())
asp = np.asarray(t.column("aspect").to_pylist())
m = (cat == "pk") & (asp == "bpo")
prot = np.asarray(t.column("protein_accession").to_pylist())[m]
go = np.asarray(t.column("go_term_id").to_pylist())[m]
score = t.column("reranker_score").to_numpy(zero_copy_only=False).astype(np.float64)[m]

ecat = np.asarray(ev.column("category").to_pylist())
easp = np.asarray(ev.column("aspect").to_pylist())
em = (ecat == "pk") & (easp == "bpo")
assert np.array_equal(prot, np.asarray(ev.column("protein_accession").to_pylist())[em])
assert np.array_equal(go, np.asarray(ev.column("go_term_id").to_pylist())[em])
dist = ev.column("distance").to_numpy(zero_copy_only=False).astype(np.float64)[em]
lenq = ev.column("length_query").to_numpy(zero_copy_only=False).astype(np.float64)[em]
taxrel = np.asarray(ev.column("taxonomic_relation").to_pylist())[em]

scratch = W / "atlas_scratch"
(scratch / "rb_dep").mkdir(parents=True, exist_ok=True)
(scratch / "rb_orc").mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- parse via cafaeval itself
ont = obo_parser(OBO, ("is_a", "part_of"), IA, False)
gts = gt_parser(str(DS / "gt_pk_bp.tsv"), ont)
ont = ont[NS]
gt = gts[NS]
toi = ont.toi_ia
ia_w = ont.ia[toi].astype(np.float64)
G_full = gt.matrix[:, toi]

tidx = {tt: i["index"] for tt, i in ont.terms_dict.items()}
col_of = np.array([tidx.get(g, -1) for g in go])
row_of = np.array([gt.ids.get(p, -1) for p in prot])
ok = (col_of >= 0) & (row_of >= 0)
in_gt = np.zeros(len(go), dtype=bool)
in_gt[ok] = gt.matrix[row_of[ok], col_of[ok]]


def parse(sub, s):
    with open(scratch / sub / "p.tsv", "w") as fh:
        for p, g, v in zip(prot, go, s):
            fh.write(f"{p}\t{g}\t{v:.6f}\n")
    return pred_parser(str(scratch / sub / "p.tsv"), {NS: ont}, gts, "fill", 500, 1)[NS].matrix[:, toi]


P_full = parse("rb_dep", score)
O_full = parse("rb_orc", in_gt.astype(np.float64))  # the REAL pool oracle

has_gt = G_full.any(axis=1)
G, P, O = G_full[has_gt], P_full[has_gt], O_full[has_gt]
acc = np.array([None] * G_full.shape[0], dtype=object)
for a_, i in gt.ids.items():
    acc[i] = a_
acc = acc[has_gt]
n = len(acc)
gt_w = G @ ia_w


def sweep(pred, g):
    gw = float((g @ ia_w).sum())
    r, c = np.nonzero(pred)
    if r.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    last = np.searchsorted(TAU, pred[r, c], side="right") - 1
    k = last >= 0
    r, c, last = r[k], c[k], last[k]
    w = ia_w[c]
    nt = len(TAU)
    pa = np.cumsum(np.bincount(last, weights=w, minlength=nt)[::-1])[::-1]
    ta = np.cumsum(np.bincount(last, weights=w * g[r, c].astype(np.float64), minlength=nt)[::-1])[::-1]
    pr = np.divide(ta, pa, out=np.zeros(nt), where=pa > 0)
    rc = ta / gw
    f = np.divide(2 * pr * rc, pr + rc, out=np.zeros(nt), where=(pr + rc) > 0)
    i = int(np.argmax(f))
    return float(f[i]), float(TAU[i]), float(pr[i]), float(rc[i])


f_dep, tau_star, pr_dep, rc_dep = sweep(P, G)
out["GATE_deployed"] = {"f_micro_w": round(f_dep, 4), "tau": round(tau_star, 3),
                        "pr_micro_w": round(pr_dep, 4), "rc_micro_w": round(rc_dep, 4)}
out["GATE_reproduces_anchor_0.2131"] = bool(abs(f_dep - 0.2131) < 5e-4)
print(f"[GATE] deployed f={f_dep:.4f} tau={tau_star:.3f}", flush=True)
if not out["GATE_reproduces_anchor_0.2131"]:
    sys.exit("GATE FAILED")

f_orc, tau_orc, _, _ = sweep(O, G)
out["pool_oracle"] = {"gt_membership_f_micro_w": round(f_orc, 4), "tau": round(tau_orc, 3),
                      "label_built_f_micro_w_SUPERSEDED": 0.6077}
print(f"[GATE] real pool oracle f={f_orc:.4f}", flush=True)


def masses(pred, tau):
    hit = pred >= tau
    return (hit & G) @ ia_w, (hit & ~G) @ ia_w


tp_d, fp_d = masses(P, tau_star)
tp_o, fp_o = masses(O, tau_orc)
TP, FP, GW = tp_d.sum(), fp_d.sum(), gt_w.sum()
f_base = fmicro_base = None


def fmicro(tp, fp, fn):
    tp, fp, fn = float(tp), float(fp), float(fn)
    pr = tp / (tp + fp) if tp + fp > 0 else 0.0
    rc = tp / (tp + fn) if tp + fn > 0 else 0.0
    return (2 * pr * rc / (pr + rc)) if pr + rc > 0 else 0.0


f_base = fmicro(TP, FP, GW - TP)
delivered_f = np.array([fmicro(tp_d[i], fp_d[i], gt_w[i] - tp_d[i]) for i in range(n)])
oracle_f = np.array([fmicro(tp_o[i], fp_o[i], gt_w[i] - tp_o[i]) for i in range(n)])
headroom = np.array([fmicro(TP - tp_d[i] + tp_o[i], FP - fp_d[i] + fp_o[i],
                            GW - (TP - tp_d[i] + tp_o[i])) - f_base for i in range(n)])

# ---------------------------------------------------------------- UNREACHABLE vs MISORDERED
reach = tp_o > 0
out["split"] = {"n_proteins": int(n), "n_reachable": int(reach.sum()),
                "n_unreachable": int((~reach).sum()),
                "pct_unreachable": round(float((~reach).mean()) * 100, 2),
                "pct_unreachable_label_built_SUPERSEDED": 34.55,
                "gt_ia_mass_total": round(float(GW), 1),
                "pct_gt_ia_mass_unreachable": round(float(gt_w[~reach].sum() / GW) * 100, 2)}
f_dep_r = sweep(P[reach], G[reach])
f_orc_r = sweep(O[reach], G[reach])
out["reachable_only"] = {"deployed_f_micro_w": round(f_dep_r[0], 4), "deployed_tau": round(f_dep_r[1], 3),
                         "oracle_f_micro_w": round(f_orc_r[0], 4),
                         "capture_ratio": round(f_dep_r[0] / f_orc_r[0], 4)}
out["all_proteins"] = {"deployed_f_micro_w": round(f_dep, 4), "oracle_f_micro_w": round(f_orc, 4),
                       "capture_ratio": round(f_dep / f_orc, 4)}
print(f"[SPLIT] unreachable {(~reach).sum()}/{n}; capture(reachable) {f_dep_r[0]/f_orc_r[0]:.1%}", flush=True)

# ---------------------------------------------------------------- concentration
o = np.argsort(-headroom)
cum = np.cumsum(headroom[o])
tot = cum[-1]
conc = {"sum_of_marginal_headrooms": round(float(tot), 4),
        "global_gap_oracle_minus_deployed": round(float(f_orc - f_dep), 4),
        "note": "marginal headrooms are not a partition of the gap (micro-F is nonlinear)",
        "n_positive": int((headroom > 1e-9).sum()), "n_negative": int((headroom < -1e-9).sum())}
for q in (5, 10, 20, 30, 50):
    k = max(1, int(round(n * q / 100)))
    conc[f"top_{q}pct_proteins_carry"] = round(float(cum[k - 1] / tot), 4)
conc["pct_of_proteins_carrying_80pct"] = round(100 * (int(np.searchsorted(cum / tot, 0.80)) + 1) / n, 2)
k20 = max(1, int(round(n * 0.2)))
conc["CONTROL_top20pct_by_gt_ia_mass_carry"] = round(
    float(np.cumsum(headroom[np.argsort(-gt_w)])[k20 - 1] / tot), 4)
conc["CONTROL_random_top20pct_carry"] = round(
    float(np.cumsum(headroom[np.random.default_rng(42).permutation(n)])[k20 - 1] / tot), 4)
out["concentration"] = conc

# ---------------------------------------------------------------- covariates
pool_idx = {}
for i, p in enumerate(prot):
    pool_idx.setdefault(p, []).append(i)
known_ct = Counter()
with open(KNOWN) as fh:
    next(fh)
    for line in fh:
        p, _, asp_ = line.rstrip("\n").split("\t")
        if asp_ == "P":
            known_ct[p] += 1

RANK = {"same": 0, "close": 1, "intermediate": 2, "distant": 3, "root-only": 4}
rec, med_d, best_rel, psize, precall = [], [], [], [], []
for i in range(n):
    a_ = acc[i]
    ix = pool_idx.get(a_, [])
    md = float(np.nanmedian(dist[ix])) if ix and not np.all(np.isnan(dist[ix])) else np.nan
    rr = [RANK[x] for x in taxrel[ix] if x in RANK]
    br = min(rr) if rr else 9
    pr_ = float(tp_o[i] / gt_w[i]) if gt_w[i] > 0 else 0.0
    med_d.append(md); best_rel.append(br); psize.append(len(ix)); precall.append(pr_)
    rec.append({"protein": a_, "delivered_f": round(float(delivered_f[i]), 4),
                "oracle_f": round(float(oracle_f[i]), 4),
                "headroom": round(float(headroom[i]), 6),
                "tp_ia": round(float(tp_d[i]), 3), "fp_ia": round(float(fp_d[i]), 3),
                "fn_ia": round(float(gt_w[i] - tp_d[i]), 3), "gt_ia": round(float(gt_w[i]), 3),
                "oracle_tp_ia": round(float(tp_o[i]), 3), "reachable": bool(reach[i]),
                "pool_recall_ia": round(pr_, 4), "pool_size": len(ix),
                "length_query": float(lenq[ix][0]) if ix else None,
                "median_candidate_distance": None if np.isnan(md) else round(md, 4),
                "closest_taxonomic_relation": {v: k for k, v in RANK.items()}.get(br, "no-knn-donor"),
                "n_t0_known_bp": known_ct.get(a_, 0)})
json.dump(rec, open(W / "atlas_per_protein.json", "w"))

med_d = np.array(med_d); psize = np.array(psize, float); precall = np.array(precall)
best_rel = np.array(best_rel, float)
nk = np.array([r["n_t0_known_bp"] for r in rec], float)
lq = np.array([r["length_query"] if r["length_query"] is not None else np.nan for r in rec])


def cut(name, v, edges, labels):
    rows = []
    b = np.digitize(np.asarray(v, float), edges)
    tot_r = headroom[reach].sum()
    for k, lab in enumerate(labels):
        s = (b == k) & reach  # reachable ONLY: the ranker is never charged for generation
        if s.sum() == 0:
            continue
        rows.append({"band": lab, "n": int(s.sum()),
                     "headroom_sum": round(float(headroom[s].sum()), 4),
                     "headroom_share": round(float(headroom[s].sum() / tot_r), 4),
                     "headroom_per_protein_x1e4": round(float(headroom[s].mean()) * 1e4, 3),
                     "mean_delivered_f": round(float(delivered_f[s].mean()), 4),
                     "mean_oracle_f": round(float(oracle_f[s].mean()), 4),
                     "mean_pool_recall_ia": round(float(precall[s].mean()), 4)})
    out.setdefault("cuts", {})[name] = rows


cut("protein_length", lq, [0, 200, 400, 700, 1200], ["<0", "0-200", "200-400", "400-700", "700-1200", ">=1200"])
cut("median_candidate_distance_NEIGHBOUR_IDENTITY", med_d, [0, .1, .25, .45, .65],
    ["neg", "0-0.1", "0.1-0.25", "0.25-0.45", "0.45-0.65", ">=0.65"])
cut("n_t0_known_bp", nk, [0, 1, 5, 15, 40], ["neg", "0", "1-4", "5-14", "15-39", ">=40"])
cut("pool_recall_ia", precall, [0, .001, .25, .5, .75], ["neg", "0", "0-0.25", "0.25-0.5", "0.5-0.75", ">=0.75"])
cut("pool_size", psize, [0, 50, 100, 200, 400], ["neg", "0-49", "50-99", "100-199", "200-399", ">=400"])
cut("closest_neighbour_taxonomic_relation", best_rel, [0, 1, 2, 3, 4, 5],
    ["same", "close", "intermediate", "distant", "root-only", "no-knn-donor", "x"])
out["taxon_note"] = ("the QUERY's taxon is absent from every frozen input (no taxid in eval.parquet, "
                     "no OX= in lafa_queries_7401.fasta, none in the CAFA_forever release files). "
                     "This axis is NEIGHBOUR taxonomy and is degenerate: 4,271 of 4,301 reachable "
                     "proteins (99.3%) have a 'close'-or-better donor, so it cannot testify. The "
                     "taxon axis in the brief CANNOT be cut from frozen data.")

# ---------------------------------------------------------------- term-level cuts of FN mass
parents = [ont.terms_list[i]["adj"] for i in range(ont.idxs)]
children = [ont.terms_list[i]["children"] for i in range(ont.idxs)]
depth = np.full(ont.idxs, -1, np.int32)
dq = deque([i for i in range(ont.idxs) if not parents[i]])
for r_ in dq:
    depth[r_] = 0
while dq:
    u = dq.popleft()
    for c_ in children[u]:
        if depth[c_] < 0:
            depth[c_] = depth[u] + 1
            dq.append(c_)
depth_toi = depth[toi]

miss = ((P < tau_star) & G)[reach]
in_pool = (O[reach] > 0)
cells = np.nonzero(miss)
mi_ia, mi_dep = ia_w[cells[1]], depth_toi[cells[1]]
cells_rk = np.nonzero(miss & in_pool)
rk_ia, rk_dep = ia_w[cells_rk[1]], depth_toi[cells_rk[1]]
out["fn_mass"] = {"reachable_proteins_total": round(float(mi_ia.sum()), 1),
                  "ranker_fixable_in_pool": round(float(rk_ia.sum()), 1),
                  "ranker_fixable_share": round(float(rk_ia.sum() / mi_ia.sum()), 4)}

ia_e, ia_l = [0, 1, 3, 6, 10], ["<0", "0-1", "1-3", "3-6", "6-10", ">10"]
d_e, d_l = [0, 3, 5, 7, 9], ["<0", "0-2", "3-4", "5-6", "7-8", ">=9"]
for nm, vals, w_, e, l in [("missed_term_IA_band_ALL", mi_ia, mi_ia, ia_e, ia_l),
                           ("missed_term_IA_band_RANKER_FIXABLE", rk_ia, rk_ia, ia_e, ia_l),
                           ("missed_term_DAG_depth_ALL", mi_dep, mi_ia, d_e, d_l),
                           ("missed_term_DAG_depth_RANKER_FIXABLE", rk_dep, rk_ia, d_e, d_l)]:
    b = np.digitize(vals, e)
    tt = w_.sum()
    out.setdefault("cuts", {})[nm] = [
        {"band": l[k], "n_missed_cells": int((b == k).sum()),
         "fn_ia_mass": round(float(w_[b == k].sum()), 1),
         "share_of_fn_mass": round(float(w_[b == k].sum() / tt), 4),
         "mean_ia": round(float(w_[b == k].mean()), 3)}
        for k in range(len(l)) if (b == k).sum()]

json.dump(out, open(W / "atlas_atlas.json", "w"), indent=1)
print(json.dumps(out, indent=1), flush=True)
print("DONE", flush=True)
