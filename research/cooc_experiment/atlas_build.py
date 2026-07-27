"""PKW.3 -- the PER-PROTEIN ERROR ATLAS for PK-BPO.

WHY AN IN-HOUSE SWEEP AND NOT cafa_eval.
f_micro_w is a MICRO metric: at a fixed tau it is a function of exactly three pooled
sums, TP_w / FP_w / FN_w, each of which is a plain sum over proteins of an IA-weighted
mass. So the metric IS additive per protein even though F itself is not linear in those
sums. That is the whole reason a per-protein atlas is possible at all here.

To get the per-protein masses we do NOT re-implement cafaeval: we call its own
obo_parser / gt_parser / pred_parser (prop=fill, norm=cafa, no_orphans, max_terms=500)
and then read the masses straight off the parsed, propagated matrices. The sweep kernel
below is a transcription of cafaeval.evaluation.compute_confusion_matrix restricted to
the weighted branch. GATE: it must reproduce the anchor (f_micro_w=0.2131 @ tau=0.393)
to 4 decimals, or the atlas is measuring something else and the script aborts.

NO TRANSFORM is applied to any score anywhere (see the rankpct incident). Raw booster
score straight to the tau grid, exactly as the anchor scores it.
"""
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, "/home/frapercan/Thesis2/repositories/PROTEA/.venv/lib/python3.12/site-packages")
from cafaeval.parser import obo_parser, gt_parser, pred_parser  # noqa: E402

W = Path("/home/frapercan/Thesis2/storage/cooc_experiment")
DS = Path("/home/frapercan/Thesis2/repositories/protea-reranker-lab/datasets/protst-global-train227-test230")
OBO = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
IA = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv"
KNOWN = "/home/frapercan/Thesis2/CAFA_forever/data/releases/Sep_2025_Mar_2026/groundtruth_PK_known.tsv"
NS = "biological_process"
TAU = np.arange(0.001, 1, 0.001)

out = {"what": "PKW.3 per-protein error atlas, PK-BPO"}
scratch = W / "atlas_scratch"
scratch.mkdir(exist_ok=True)


# ---------------------------------------------------------------- inputs
t = pq.read_table(W / "rerank_out" / "eval_scores.parquet",
                  columns=["protein_accession", "go_term_id", "category", "aspect", "reranker_score"])
ev = pq.read_table(DS / "eval.parquet",
                   columns=["protein_accession", "go_term_id", "category", "aspect", "label",
                            "distance", "length_query", "neighbor_min_distance",
                            "taxonomic_relation", "taxonomic_distance"])

cat = np.asarray(t.column("category").to_pylist())
asp = np.asarray(t.column("aspect").to_pylist())
m = (cat == "pk") & (asp == "bpo")
prot = np.asarray(t.column("protein_accession").to_pylist())[m]
go = np.asarray(t.column("go_term_id").to_pylist())[m]
score = t.column("reranker_score").to_numpy(zero_copy_only=False).astype(np.float64)[m]

# row-order identity between the two files is the premise of this whole atlas: GATE it.
ecat = np.asarray(ev.column("category").to_pylist())
easp = np.asarray(ev.column("aspect").to_pylist())
em = (ecat == "pk") & (easp == "bpo")
eprot = np.asarray(ev.column("protein_accession").to_pylist())[em]
ego = np.asarray(ev.column("go_term_id").to_pylist())[em]
assert np.array_equal(prot, eprot) and np.array_equal(go, ego), "ROW ORDER MISMATCH -- abort"
label = ev.column("label").to_numpy(zero_copy_only=False).astype(np.float64)[em]
out["rows_pk_bpo"] = int(m.sum())
out["proteins_in_pool"] = int(len(set(prot.tolist())))
print(f"pool rows={m.sum()} proteins={out['proteins_in_pool']}", flush=True)


def write_pred(path, s):
    with open(path, "w") as fh:
        for p, g, v in zip(prot, go, s):
            fh.write(f"{p}\t{g}\t{v:.6f}\n")


# ---------------------------------------------------------------- parse via cafaeval itself
ont = obo_parser(OBO, ("is_a", "part_of"), IA, False)[NS]  # orphans=not no_orphans=False
gts_all = gt_parser(str(DS / "gt_pk_bp.tsv"), {NS: ont})
gt = gts_all[NS]
toi = ont.toi_ia
ia_w = ont.ia[toi].astype(np.float64)
G = gt.matrix[:, toi]
n_prot_gt = G.shape[0]
print(f"gt proteins={n_prot_gt} toi_ia={len(toi)}", flush=True)

dep = scratch / "deployed"
dep.mkdir(exist_ok=True)
write_pred(dep / "dep.tsv", score)
P = pred_parser(str(dep / "dep.tsv"), {NS: ont}, gts_all, "fill", 500, 1)[NS].matrix[:, toi]

orc = scratch / "oracle"
orc.mkdir(exist_ok=True)
write_pred(orc / "orc.tsv", label)
O = pred_parser(str(orc / "orc.tsv"), {NS: ont}, gts_all, "fill", 500, 1)[NS].matrix[:, toi]

# cafaeval drops proteins with no gt in the toi before scoring; mirror that exactly.
has_gt = G.any(axis=1)
G, P, O = G[has_gt], P[has_gt], O[has_gt]
acc = np.array([None] * n_prot_gt, dtype=object)
for a, i in gt.ids.items():
    acc[i] = a
acc = acc[has_gt]
n = len(acc)
out["proteins_scored"] = int(n)
print(f"proteins scored={n}", flush=True)


# ---------------------------------------------------------------- masses
def masses(pred, tau):
    """Per-protein IA-weighted (tp, fp) at a fixed tau. Transcribes the weighted branch
    of cafaeval.evaluation.compute_confusion_matrix."""
    hit = pred >= tau
    tp = (hit & G) @ ia_w
    fp = (hit & ~G) @ ia_w
    return tp, fp


gt_w = G @ ia_w  # per-protein total IA mass of the truth. FN_i = gt_w_i - tp_i.


def fmicro(tp, fp, fn):
    tp, fp, fn = float(tp), float(fp), float(fn)
    pr = tp / (tp + fp) if tp + fp > 0 else 0.0
    rc = tp / (tp + fn) if tp + fn > 0 else 0.0
    return (2 * pr * rc / (pr + rc)) if pr + rc > 0 else 0.0, pr, rc


def sweep(pred, rows=None):
    """Global f_micro_w over the whole tau grid in one pass, optionally restricted to a
    subset of proteins. Returns (best_f, best_tau, pr, rc).

    Same scatter + right-to-left cumsum as cafaeval.evaluation.compute_confusion_matrix_sparse:
    each non-zero prediction is dropped into the HIGHEST tau bin at which it still fires,
    then a reverse cumsum recovers the active mass at every tau. A dense 999 x n_prot x
    n_toi scan would be 1.2e11 cells; this is O(nnz + n_tau).
    """
    g = G if rows is None else G[rows]
    p = pred if rows is None else pred[rows]
    gw = float((g @ ia_w).sum())
    r, c = np.nonzero(p)
    if r.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    s = p[r, c]
    last = np.searchsorted(TAU, s, side="right") - 1  # active at TAU[k] iff k <= last
    keep = last >= 0
    r, c, last = r[keep], c[keep], last[keep]
    w = ia_w[c]
    w_tp = w * g[r, c].astype(np.float64)
    n_tau = len(TAU)
    pred_at = np.cumsum(np.bincount(last, weights=w, minlength=n_tau)[::-1])[::-1]
    tp_at = np.cumsum(np.bincount(last, weights=w_tp, minlength=n_tau)[::-1])[::-1]
    fp_at = pred_at - tp_at
    pr = np.divide(tp_at, pred_at, out=np.zeros(n_tau), where=pred_at > 0)
    rc = np.divide(tp_at, gw, out=np.zeros(n_tau), where=gw > 0) if gw > 0 else np.zeros(n_tau)
    f = np.divide(2 * pr * rc, pr + rc, out=np.zeros(n_tau), where=(pr + rc) > 0)
    k = int(np.argmax(f))
    return float(f[k]), float(TAU[k]), float(pr[k]), float(rc[k])


# GATE 1 -- the in-house sweep must reproduce the anchor or nothing below is valid.
f_dep, tau_star, pr_dep, rc_dep = sweep(P)
out["GATE_deployed_sweep"] = {"f_micro_w": round(f_dep, 4), "tau": round(tau_star, 3),
                              "pr_micro_w": round(pr_dep, 4), "rc_micro_w": round(rc_dep, 4)}
out["GATE_anchor_expected"] = {"f_micro_w": 0.2131, "tau": 0.393}
ok = abs(f_dep - 0.2131) < 5e-4
out["GATE_reproduces_anchor"] = bool(ok)
print(f"[GATE] deployed sweep f={f_dep:.4f} tau={tau_star:.3f} (anchor 0.2131@0.393) ok={ok}", flush=True)
if not ok:
    json.dump(out, open(W / "atlas_atlas.json", "w"), indent=1)
    sys.exit("GATE FAILED -- in-house sweep does not reproduce the anchor")

f_orc, tau_orc, pr_orc, rc_orc = sweep(O)
out["GATE_pool_oracle_sweep"] = {"f_micro_w": round(f_orc, 4), "tau": round(tau_orc, 3),
                                 "pr_micro_w": round(pr_orc, 4), "rc_micro_w": round(rc_orc, 4)}
out["GATE_pool_oracle_expected"] = 0.6077
print(f"[GATE] pool oracle    f={f_orc:.4f} tau={tau_orc:.3f} (known 0.6077)", flush=True)

# ---------------------------------------------------------------- the atlas
tp_d, fp_d = masses(P, tau_star)
tp_o, fp_o = masses(O, tau_orc)
TP, FP = tp_d.sum(), fp_d.sum()
GW = gt_w.sum()

# ATTRIBUTION. delivered_f_i / oracle_f_i = the protein's OWN IA-weighted F at the global
# tau. headroom_i = the MARGINAL move in the GLOBAL f_micro_w if this one protein alone
# were served by the pool oracle and every other protein were left exactly as it is, tau
# held at the global optimum. Marginal, not a share: micro F is nonlinear in the pooled
# sums so the headrooms do not sum to the 0.6077-0.2131 gap. It is the honest per-protein
# quantity because it answers the only actionable question ("what do I gain if I fix this
# one?"), and it is measured in the units the campaign decides on.
f_base = fmicro(TP, FP, GW - TP)[0]
delivered_f = np.array([fmicro(tp_d[i], fp_d[i], gt_w[i] - tp_d[i])[0] for i in range(n)])
oracle_f = np.array([fmicro(tp_o[i], fp_o[i], gt_w[i] - tp_o[i])[0] for i in range(n)])
headroom = np.array([fmicro(TP - tp_d[i] + tp_o[i], FP - fp_d[i] + fp_o[i],
                            GW - (TP - tp_d[i] + tp_o[i]))[0] - f_base for i in range(n)])

# REACHABLE vs UNREACHABLE: does this protein's candidate pool contain ANY of its true
# terms? Charged to generation, never to the ranker.
reach = tp_o > 0
out["split"] = {
    "n_proteins": int(n),
    "n_reachable": int(reach.sum()),
    "n_unreachable": int((~reach).sum()),
    "pct_unreachable": round(float((~reach).mean()) * 100, 2),
    "gt_ia_mass_total": round(float(GW), 1),
    "gt_ia_mass_unreachable": round(float(gt_w[~reach].sum()), 1),
    "pct_gt_ia_mass_unreachable": round(float(gt_w[~reach].sum() / GW) * 100, 2),
}

f_dep_r, tau_dep_r, _, _ = sweep(P, reach)
f_orc_r, tau_orc_r, _, _ = sweep(O, reach)
out["reachable_only"] = {
    "deployed_f_micro_w": round(f_dep_r, 4), "deployed_tau": round(tau_dep_r, 3),
    "oracle_f_micro_w": round(f_orc_r, 4), "oracle_tau": round(tau_orc_r, 3),
    "capture_ratio": round(f_dep_r / f_orc_r, 4),
}
out["all_proteins"] = {"deployed_f_micro_w": round(f_dep, 4), "oracle_f_micro_w": round(f_orc, 4),
                       "capture_ratio": round(f_dep / f_orc, 4)}
print(f"[SPLIT] unreachable {(~reach).sum()}/{n} = {(~reach).mean():.1%}", flush=True)
print(f"[REACH] deployed {f_dep_r:.4f} / oracle {f_orc_r:.4f} = capture {f_dep_r/f_orc_r:.1%}", flush=True)

# ---------------------------------------------------------------- concentration curve
o = np.argsort(-headroom)
cum = np.cumsum(headroom[o])
tot = cum[-1]
out["concentration"] = {
    "sum_of_marginal_headrooms": round(float(tot), 4),
    "global_gap_oracle_minus_deployed": round(float(f_orc - f_dep), 4),
    "note": "the two differ because micro-F is nonlinear in the pooled sums; the marginal "
            "headrooms are not a partition of the gap",
}
for q in (5, 10, 20, 30, 50, 80):
    k = max(1, int(round(n * q / 100)))
    out["concentration"][f"top_{q}pct_proteins_carry"] = round(float(cum[k - 1] / tot), 4)
# where does the cumulative share cross 80%?
cross = int(np.searchsorted(cum / tot, 0.80) + 1)
out["concentration"]["pct_of_proteins_carrying_80pct_of_headroom"] = round(100 * cross / n, 2)
out["concentration"]["n_proteins_with_positive_headroom"] = int((headroom > 1e-9).sum())
out["concentration"]["n_proteins_with_negative_headroom"] = int((headroom < -1e-9).sum())

# ---------------------------------------------------------------- covariates
# per-protein features off the frozen pool (aggregate the candidate rows)
pool_idx = {}
for i, p in enumerate(prot):
    pool_idx.setdefault(p, []).append(i)
lenq = ev.column("length_query").to_numpy(zero_copy_only=False).astype(np.float64)[em]
nmd = ev.column("neighbor_min_distance").to_numpy(zero_copy_only=False).astype(np.float64)[em]
taxrel = np.asarray(ev.column("taxonomic_relation").to_pylist())[em]

known_ct = {}
with open(KNOWN) as fh:
    next(fh)
    for line in fh:
        p, _, a = line.rstrip("\n").split("\t")
        if a == "P":
            known_ct[p] = known_ct.get(p, 0) + 1

# DAG depth (shortest path to a root) over the BP graph
parents = [ont.terms_list[i]["adj"] for i in range(ont.idxs)]
roots = [i for i in range(ont.idxs) if not parents[i]]
depth = np.full(ont.idxs, -1, dtype=np.int32)
children = [ont.terms_list[i]["children"] for i in range(ont.idxs)]
dq = deque(roots)
for r in roots:
    depth[r] = 0
while dq:
    u = dq.popleft()
    for c in children[u]:
        if depth[c] < 0:
            depth[c] = depth[u] + 1
            dq.append(c)
depth_toi = depth[toi]

rec = []
for i in range(n):
    a = acc[i]
    ix = pool_idx.get(a, [])
    rec.append({
        "protein": a,
        "delivered_f": round(float(delivered_f[i]), 4),
        "oracle_f": round(float(oracle_f[i]), 4),
        "headroom": round(float(headroom[i]), 6),
        "tp_ia": round(float(tp_d[i]), 3),
        "fp_ia": round(float(fp_d[i]), 3),
        "fn_ia": round(float(gt_w[i] - tp_d[i]), 3),
        "gt_ia": round(float(gt_w[i]), 3),
        "oracle_tp_ia": round(float(tp_o[i]), 3),
        "reachable": bool(reach[i]),
        "pool_recall_ia": round(float(tp_o[i] / gt_w[i]), 4) if gt_w[i] > 0 else 0.0,
        "pool_size": len(ix),
        "n_true_in_pool": int(label[ix].sum()) if ix else 0,
        "length_query": float(lenq[ix][0]) if ix else None,
        "neighbor_min_distance": float(np.nanmin(nmd[ix])) if ix else None,
        "n_t0_known_bp": known_ct.get(a, 0),
        "frac_same_taxon_candidates": (round(float((taxrel[ix] == "same_species").mean()), 4)
                                       if ix else None),
    })
json.dump(rec, open(W / "atlas_per_protein.json", "w"))


# ---------------------------------------------------------------- cuts
def cut(name, vals, edges, labels):
    rows = []
    v = np.asarray([np.nan if x is None else x for x in vals], dtype=np.float64)
    b = np.digitize(v, edges)
    for k, lab in enumerate(labels):
        s = (b == k) & reach  # ONLY the reachable subset: the ranker is not charged for generation
        if s.sum() == 0:
            continue
        rows.append({
            "band": lab, "n": int(s.sum()),
            "headroom_sum": round(float(headroom[s].sum()), 4),
            "headroom_share": round(float(headroom[s].sum() / headroom[reach].sum()), 4),
            "headroom_per_protein_x1e4": round(float(headroom[s].mean()) * 1e4, 3),
            "mean_delivered_f": round(float(delivered_f[s].mean()), 4),
            "mean_oracle_f": round(float(oracle_f[s].mean()), 4),
            "mean_pool_recall_ia": round(float((tp_o[s] / np.maximum(gt_w[s], 1e-9)).mean()), 4),
        })
    out.setdefault("cuts", {})[name] = rows


cut("protein_length", [r["length_query"] for r in rec],
    [0, 200, 400, 700, 1200], ["<0", "0-200", "200-400", "400-700", "700-1200", ">1200"])
cut("neighbor_min_distance", [r["neighbor_min_distance"] for r in rec],
    [0, .2, .4, .6, .8], ["<0", "0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", ">0.8"])
cut("n_t0_known_bp", [r["n_t0_known_bp"] for r in rec],
    [0, 1, 5, 15, 40], ["neg", "0", "1-4", "5-14", "15-39", ">=40"])
cut("pool_recall_ia", [r["pool_recall_ia"] for r in rec],
    [0, .001, .25, .5, .75], ["neg", "0", "0-0.25", "0.25-0.5", "0.5-0.75", ">0.75"])
# NB np.digitize(x, edges) returns 1 for edges[0] <= x < edges[1], so the label list must
# start at the "below edges[0]" band. An earlier revision shifted these by one.
cut("pool_size", [r["pool_size"] for r in rec],
    [0, 50, 100, 200, 400], ["neg", "0-49", "50-99", "100-199", "200-399", ">=400"])
cut("frac_same_taxon_candidates", [r["frac_same_taxon_candidates"] for r in rec],
    [0, .001, .25, .5, .9], ["neg", "0", "0-0.25", "0.25-0.5", "0.5-0.9", ">0.9"])
out["taxon_note"] = ("query TAXON is NOT present in any frozen input (eval.parquet has no "
                     "taxid; the CAFA_forever release files and lafa_queries_7401.fasta carry "
                     "no OX). frac_same_taxon_candidates is a NEIGHBOUR-taxonomy proxy, not "
                     "the query's clade.")

# TERM-LEVEL cuts of the FN IA mass (a protein has many missed terms; band the terms, not
# the proteins). Restricted to reachable proteins.
hit_d = P >= tau_star
miss = (~hit_d) & G  # FN cells
miss_r = miss[reach]
cells = np.nonzero(miss_r)
missed_ia = ia_w[cells[1]]
missed_depth = depth_toi[cells[1]]
tot_fn = missed_ia.sum()

ia_edges = [0, 1, 3, 6, 10]
ia_labels = ["<0", "0-1", "1-3", "3-6", "6-10", ">10"]
b = np.digitize(missed_ia, ia_edges)
out["cuts"]["missed_term_IA_band"] = [
    {"band": ia_labels[k], "n_missed_cells": int((b == k).sum()),
     "fn_ia_mass": round(float(missed_ia[b == k].sum()), 1),
     "share_of_fn_mass": round(float(missed_ia[b == k].sum() / tot_fn), 4)}
    for k in range(len(ia_labels)) if (b == k).sum()
]
# same banding over the pool-reachable FN only (what the ranker could actually have fixed)
in_pool = O[reach] > 0
cells_rk = np.nonzero(miss_r & in_pool)
ia_rk = ia_w[cells_rk[1]]
b2 = np.digitize(ia_rk, ia_edges)
out["cuts"]["missed_term_IA_band_RANKER_FIXABLE"] = [
    {"band": ia_labels[k], "n_missed_cells": int((b2 == k).sum()),
     "fn_ia_mass": round(float(ia_rk[b2 == k].sum()), 1),
     "share_of_fn_mass": round(float(ia_rk[b2 == k].sum() / max(ia_rk.sum(), 1e-9)), 4)}
    for k in range(len(ia_labels)) if (b2 == k).sum()
]
out["fn_mass_reachable_proteins"] = round(float(tot_fn), 1)
out["fn_mass_ranker_fixable"] = round(float(ia_rk.sum()), 1)
out["fn_mass_ranker_fixable_share"] = round(float(ia_rk.sum() / tot_fn), 4)

d_edges = [0, 3, 5, 7, 9]
d_labels = ["<0", "0-2", "3-4", "5-6", "7-8", ">=9"]
bd = np.digitize(missed_depth, d_edges)
out["cuts"]["missed_term_DAG_depth"] = [
    {"band": d_labels[k], "n_missed_cells": int((bd == k).sum()),
     "fn_ia_mass": round(float(missed_ia[bd == k].sum()), 1),
     "share_of_fn_mass": round(float(missed_ia[bd == k].sum() / tot_fn), 4),
     "mean_ia": round(float(missed_ia[bd == k].mean()), 3)}
    for k in range(len(d_labels)) if (bd == k).sum()
]

# CONTROL for the concentration curve (rule 4: a big number is a suspect). If headroom
# were a pure size effect, ranking by gt IA mass alone would concentrate identically.
o2 = np.argsort(-gt_w)
cum2 = np.cumsum(headroom[o2])
k20 = max(1, int(round(n * 0.2)))
out["concentration"]["CONTROL_top20pct_by_gt_ia_mass_carry"] = round(float(cum2[k20 - 1] / tot), 4)
rnd = np.random.default_rng(42).permutation(n)
out["concentration"]["CONTROL_random_top20pct_carry"] = round(float(np.cumsum(headroom[rnd])[k20 - 1] / tot), 4)

json.dump(out, open(W / "atlas_atlas.json", "w"), indent=1)
print(json.dumps(out, indent=1), flush=True)
print("DONE", flush=True)
