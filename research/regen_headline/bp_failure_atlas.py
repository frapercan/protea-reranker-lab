"""BP FAILURE CHARACTERIZATION -- per-protein error atlas for LK-BPO and PK-BPO.

Reuses atlas_rebuild.py's oracle/propagation machinery (cafaeval parser, gt-membership
pool oracle, IA-weighted per-protein micro-F decomposition) and extends it:
  * BOTH lost cells (LK-BPO ~523, PK-BPO ~4402), not just PK.
  * DELIVERED comes from the deployed percut+graft predictions, not the raw reranker.
  * Board frame: toi restriction + PK -known exclusion (per-protein mask).
  * NEW axes: query TAXON (eval_acc_taxon.tsv) and STRING network degree (merged later).

Frame notes (honest):
  f_micro_w at a global tau = harmonic mean of POOLED IA-weighted pr/rc (coverage NOT applied,
  matches cafa_eval f_micro_w). Per-protein tp/fp/fn IA masses are exactly additive; F is not.
  delivered_f_i / oracle_f_i = the protein's OWN IA-weighted F at the GLOBAL optimal tau of the
  respective system. headroom_i = the MARGINAL move in global delivered f_micro_w when protein i
  alone is served by the pool oracle (tau held at delivered global optimum). Marginal headrooms do
  NOT partition the global gap (micro-F nonlinear). tau frozen at global optimum = a protein well
  ORDERED but badly CALIBRATED is charged for miscalibration.
  PK -known: a (protein,term) is scored only if term in toi AND not in the protein's t0-known set
  (propagated). Known-true terms are neither TP nor FN; predictions on them are not FP. This is the
  board's PK frame and it re-targets scoring onto the NOVEL BP tail.
"""
import json, sys
from collections import Counter, deque
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from cafaeval.parser import obo_parser, gt_parser, gt_exclude_parser, update_toi, pred_parser

ROOT = Path("/home/frapercan/Thesis2")
W = ROOT / "storage/regen_headline"
DS = ROOT / "repositories/protea-reranker-lab/datasets/protst-global-train227-test230"
LAFA_GT = ROOT / "repositories/protea-reranker-lab/results/sparse_classifier/lafa_gt"
PRED = ROOT / "repositories/protea-reranker-lab/results/sparse_classifier/percut_rerank/predictions"
REL = ROOT / "CAFA_forever/data/releases/Sep_2025_Mar_2026"
OBO = str(ROOT / "protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo")
IA_F = str(ROOT / "protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv")
TOI_FILE = str(LAFA_GT / "groundtruth_terms_of_interest.txt")
TAXON_F = W / "eval_acc_taxon.tsv"
FASTA = ROOT / "storage/text_scorer/ident/query.fasta"
NS = "biological_process"
TAU = np.arange(0.001, 1, 0.001)
SC = W / "bp_atlas_scratch"
SC.mkdir(exist_ok=True)

# ---- shared resources ----
print("[load] obo + toi", flush=True)
ONT = obo_parser(OBO, ("is_a", "part_of"), IA_F, False)
ONT = update_toi(ONT, TOI_FILE)
ont = ONT[NS]
toi = ont.toi_ia
ia_w = ont.ia[toi].astype(np.float64)
tidx = {tt: i["index"] for tt, i in ont.terms_dict.items()}

# query length map
LEN = {}
acc = None
with open(FASTA) as fh:
    for line in fh:
        if line.startswith(">"):
            acc = line[1:].strip().split()[0]
            LEN[acc] = 0
        elif acc is not None:
            LEN[acc] += len(line.strip())

# taxon map
TAX = {}
for line in open(TAXON_F):
    p = line.rstrip("\n").split("\t")
    if len(p) >= 2:
        TAX[p[0]] = p[1]

# DAG depth over toi
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


def sweep(pred, gv):
    """Pooled IA-weighted micro-F over a tau grid. pred already masked to valid cells."""
    gw = float((gv @ ia_w).sum())
    r, c = np.nonzero(pred)
    if r.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    last = np.searchsorted(TAU, pred[r, c], side="right") - 1
    k = last >= 0
    r, c, last = r[k], c[k], last[k]
    w = ia_w[c]
    nt = len(TAU)
    pa = np.cumsum(np.bincount(last, weights=w, minlength=nt)[::-1])[::-1]
    ta = np.cumsum(np.bincount(last, weights=w * gv[r, c].astype(np.float64), minlength=nt)[::-1])[::-1]
    pr = np.divide(ta, pa, out=np.zeros(nt), where=pa > 0)
    rc = ta / gw if gw > 0 else np.zeros(nt)
    f = np.divide(2 * pr * rc, pr + rc, out=np.zeros(nt), where=(pr + rc) > 0)
    i = int(np.argmax(f))
    return float(f[i]), float(TAU[i]), float(pr[i]), float(rc[i])


def fmicro(tp, fp, fn):
    tp, fp, fn = float(tp), float(fp), float(fn)
    pr = tp / (tp + fp) if tp + fp > 0 else 0.0
    rc = tp / (tp + fn) if tp + fn > 0 else 0.0
    return (2 * pr * rc / (pr + rc)) if pr + rc > 0 else 0.0


CELLS = {
    "lk": {"gt": None, "known": None, "pred": PRED / "lk" / "lk.tsv", "anchor": 0.3110},
    "pk": {"gt": str(DS / "gt_pk_bp.tsv"), "known": str(LAFA_GT / "groundtruth_PK_known.tsv"),
           "pred": PRED / "pk" / "pk.tsv", "anchor": 0.1402},
}


def build_bp_gt_from_lafa():
    """LK gt: filter lafa_gt/groundtruth_LK.tsv to aspect P -> 2-col tsv."""
    src = LAFA_GT / "groundtruth_LK.tsv"
    dst = SC / "gt_lk_bp.tsv"
    with open(src) as fh, open(dst, "w") as out:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3 and parts[2] == "P":
                out.write(f"{parts[0]}\t{parts[1]}\n")
    return str(dst)


CELLS["lk"]["gt"] = build_bp_gt_from_lafa()


def run_cell(cell, cfg):
    print(f"\n=== CELL {cell} ===", flush=True)
    out = {"cell": cell + "-bpo"}
    gts = gt_parser(cfg["gt"], ONT)
    gt = gts[NS]
    # exclude (PK -known)
    if cfg["known"]:
        # build BP-only known tsv
        known_bp = SC / f"known_{cell}_bp.tsv"
        with open(cfg["known"]) as fh, open(known_bp, "w") as o:
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 3 and parts[2] == "P":
                    o.write(f"{parts[0]}\t{parts[1]}\n")
        exclude = gt_exclude_parser(str(known_bp), gts, ONT)[NS]
        excl_toi = exclude.matrix[:, toi]  # aligned to gt.ids rows
    else:
        excl_toi = np.zeros((gt.matrix.shape[0], len(toi)), dtype=bool)

    G_full = gt.matrix[:, toi]
    valid_full = ~excl_toi  # toi already applied by column selection

    # ---- pool candidate set + covariates from DS/eval.parquet ----
    ev = pq.read_table(DS / "eval.parquet",
                       columns=["protein_accession", "go_term_id", "category", "aspect",
                                "distance", "length_query", "taxonomic_relation",
                                "anc2vec_query_known_count"])
    ecat = np.asarray(ev.column("category").to_pylist())
    easp = np.asarray(ev.column("aspect").to_pylist())
    em = (ecat == cell) & (easp == "bpo")
    prot = np.asarray(ev.column("protein_accession").to_pylist())[em]
    goa = np.asarray(ev.column("go_term_id").to_pylist())[em]
    dist = ev.column("distance").to_numpy(zero_copy_only=False).astype(np.float64)[em]
    knownc = ev.column("anc2vec_query_known_count").to_numpy(zero_copy_only=False).astype(np.float64)[em]
    taxrel = np.asarray(ev.column("taxonomic_relation").to_pylist())[em]

    # oracle membership: for each pool cell, is (protein,term) in propagated gt?
    col_of = np.array([tidx.get(g, -1) for g in goa])
    row_of = np.array([gt.ids.get(p, -1) for p in prot])
    ok = (col_of >= 0) & (row_of >= 0)
    in_gt = np.zeros(len(goa), dtype=np.float64)
    in_gt[ok] = gt.matrix[row_of[ok], col_of[ok]].astype(np.float64)

    def parse(sub, prots, gos, vals):
        d = SC / sub
        d.mkdir(exist_ok=True)
        with open(d / "p.tsv", "w") as fh:
            for p, g, v in zip(prots, gos, vals):
                fh.write(f"{p}\t{g}\t{v:.6f}\n")
        mat = pred_parser(str(d / "p.tsv"), {NS: ont}, gts, "fill", None, 1)[NS].matrix[:, toi]
        return np.asarray(mat.todense()) if hasattr(mat, "todense") else np.asarray(mat)

    O_full = parse(f"orc_{cell}", prot, goa, in_gt)

    # ---- delivered from percut+graft predictions ----
    dp, dg, dv = [], [], []
    with open(cfg["pred"]) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            p, g, v = parts[0], parts[1], parts[2]
            if g in tidx:  # BP toi terms only
                dp.append(p); dg.append(g); dv.append(float(v))
    P_full = parse(f"dep_{cell}", dp, dg, np.array(dv))

    # ---- restrict to proteins with >=1 valid gt BP term ----
    Gv_full = G_full & valid_full
    has = Gv_full.any(axis=1)
    accs = np.array([None] * G_full.shape[0], dtype=object)
    for a_, i in gt.ids.items():
        accs[i] = a_
    accs = accs[has]
    G = G_full[has]; O = O_full[has]; P = P_full[has]; valid = valid_full[has]
    Gv = G & valid
    n = len(accs)
    gt_w = (Gv @ ia_w)

    Pm = P * valid
    Om = O * valid

    f_dep, tau_star, pr_dep, rc_dep = sweep(Pm, Gv)
    f_orc, tau_orc, _, _ = sweep(Om, Gv)
    out["GATE_delivered_f_micro_w"] = round(f_dep, 4)
    out["GATE_anchor"] = cfg["anchor"]
    out["GATE_reproduces"] = bool(abs(f_dep - cfg["anchor"]) < 0.01)
    out["pool_oracle_f_micro_w"] = round(f_orc, 4)
    out["tau_delivered"] = round(tau_star, 3)
    out["tau_oracle"] = round(tau_orc, 3)
    print(f"[{cell}] delivered f={f_dep:.4f} (anchor {cfg['anchor']}) oracle f={f_orc:.4f} n={n}", flush=True)

    # per-protein masses
    hit_d = Pm >= tau_star
    tp_d = (hit_d & Gv) @ ia_w
    fp_d = (hit_d & ~Gv) @ ia_w
    hit_o = Om >= tau_orc
    tp_o = (hit_o & Gv) @ ia_w
    fp_o = (hit_o & ~Gv) @ ia_w
    TP, FP, GW = tp_d.sum(), fp_d.sum(), gt_w.sum()
    f_base = fmicro(TP, FP, GW - TP)

    delivered_f = np.array([fmicro(tp_d[i], fp_d[i], gt_w[i] - tp_d[i]) for i in range(n)])
    oracle_f = np.array([fmicro(tp_o[i], fp_o[i], gt_w[i] - tp_o[i]) for i in range(n)])
    headroom = np.array([fmicro(TP - tp_d[i] + tp_o[i], FP - fp_d[i] + fp_o[i],
                                GW - (TP - tp_d[i] + tp_o[i])) - f_base for i in range(n)])

    reach = tp_o > 0
    out["n_proteins"] = int(n)
    out["global_delivered_f"] = round(f_base, 4)
    out["split"] = {"n_reachable": int(reach.sum()), "n_unreachable": int((~reach).sum()),
                    "pct_unreachable": round(float((~reach).mean()) * 100, 2),
                    "gt_ia_mass_total": round(float(GW), 1),
                    "pct_gt_ia_mass_unreachable": round(float(gt_w[~reach].sum() / GW) * 100, 2)}
    fr = sweep(Pm[reach], Gv[reach]); fo = sweep(Om[reach], Gv[reach])
    out["reachable_only"] = {"deployed_f": round(fr[0], 4), "oracle_f": round(fo[0], 4),
                             "capture_ratio": round(fr[0] / fo[0], 4) if fo[0] > 0 else None}

    # ---- RECALL vs ORDERING IA-mass split ----
    miss = (~hit_d) & Gv  # missed true cells (all proteins)
    in_pool = O > 0  # term reachable by the pool
    miss_ia = (miss * ia_w).sum()
    ordering_ia = (miss & in_pool) * ia_w  # in pool but mis-scored -> RANKING failure
    recall_ia = (miss & ~in_pool) * ia_w  # not in pool at all -> GENERATION/RECALL failure
    out["recall_vs_ordering"] = {
        "total_missed_true_ia": round(float(miss_ia), 1),
        "ordering_fail_ia_in_pool": round(float(ordering_ia.sum()), 1),
        "recall_fail_ia_not_in_pool": round(float(recall_ia.sum()), 1),
        "pct_ordering": round(float(ordering_ia.sum() / miss_ia) * 100, 2) if miss_ia > 0 else None,
        "pct_recall": round(float(recall_ia.sum() / miss_ia) * 100, 2) if miss_ia > 0 else None,
    }

    # ---- concentration ----
    o_ = np.argsort(-headroom)
    cum = np.cumsum(headroom[o_]); tot = cum[-1]
    conc = {"sum_marginal_headrooms": round(float(tot), 4),
            "global_gap": round(float(f_orc - f_dep), 4),
            "n_positive": int((headroom > 1e-9).sum()), "n_negative": int((headroom < -1e-9).sum())}
    for q in (5, 10, 20, 30, 50):
        k = max(1, int(round(n * q / 100)))
        conc[f"top_{q}pct_carry"] = round(float(cum[k - 1] / tot), 4) if tot > 0 else None
    k20 = max(1, int(round(n * 0.2)))
    conc["CONTROL_random_top20pct"] = round(
        float(np.cumsum(headroom[np.random.default_rng(42).permutation(n)])[k20 - 1] / tot), 4) if tot > 0 else None
    out["concentration"] = conc

    # ---- covariates per protein ----
    pool_idx = {}
    for i, p in enumerate(prot):
        pool_idx.setdefault(p, []).append(i)
    med_d, psize, precall, lenq, taxid, nk_known, best_rel = [], [], [], [], [], [], []
    RANK = {"same": 0, "close": 1, "intermediate": 2, "distant": 3, "root-only": 4}
    recs = []
    for i in range(n):
        a_ = accs[i]
        ix = pool_idx.get(a_, [])
        md = float(np.nanmedian(dist[ix])) if ix and not np.all(np.isnan(dist[ix])) else np.nan
        rr = [RANK[x] for x in taxrel[ix] if x in RANK]
        br = min(rr) if rr else 9
        pr_ = float(tp_o[i] / gt_w[i]) if gt_w[i] > 0 else 0.0
        kc = float(knownc[ix[0]]) if ix else np.nan
        med_d.append(md); psize.append(len(ix)); precall.append(pr_)
        lenq.append(LEN.get(a_, np.nan)); taxid.append(TAX.get(a_, "NA"))
        nk_known.append(kc); best_rel.append(br)
        recs.append({"protein": a_, "taxid": TAX.get(a_, "NA"),
                     "delivered_f": round(float(delivered_f[i]), 4),
                     "oracle_f": round(float(oracle_f[i]), 4),
                     "headroom": round(float(headroom[i]), 6),
                     "reachable": bool(reach[i]),
                     "tp_ia": round(float(tp_d[i]), 3), "fn_ia": round(float(gt_w[i] - tp_d[i]), 3),
                     "gt_ia": round(float(gt_w[i]), 3), "oracle_tp_ia": round(float(tp_o[i]), 3),
                     "pool_recall_ia": round(pr_, 4), "pool_size": len(ix),
                     "length": LEN.get(a_, None),
                     "median_candidate_distance": None if np.isnan(md) else round(md, 4),
                     "closest_taxonomic_relation": {v: k for k, v in RANK.items()}.get(br, "no-donor"),
                     "t0_known_count": None if np.isnan(kc) else int(kc)})
    json.dump(recs, open(W / f"bp_atlas_perprotein_{cell}.json", "w"))

    med_d = np.array(med_d); psize = np.array(psize, float); precall = np.array(precall)
    lenq = np.array(lenq, float); nk_known = np.array(nk_known, float); best_rel = np.array(best_rel, float)

    def cut(name, v, edges, labels, restrict_reach=True):
        rows = []
        b = np.digitize(np.asarray(v, float), edges)
        sel_base = reach if restrict_reach else np.ones(n, bool)
        tot_r = headroom[sel_base].sum()
        for k, lab in enumerate(labels):
            s = (b == k) & sel_base
            if s.sum() == 0:
                continue
            rows.append({"band": lab, "n": int(s.sum()),
                         "headroom_sum": round(float(headroom[s].sum()), 4),
                         "headroom_share": round(float(headroom[s].sum() / tot_r), 4) if tot_r else None,
                         "headroom_per_protein_x1e4": round(float(headroom[s].mean()) * 1e4, 3),
                         "mean_delivered_f": round(float(delivered_f[s].mean()), 4),
                         "mean_oracle_f": round(float(oracle_f[s].mean()), 4),
                         "mean_pool_recall": round(float(precall[s].mean()), 4)})
        out.setdefault("cuts", {})[name] = rows

    cut("protein_length", lenq, [0, 200, 400, 700, 1200], ["<0", "0-200", "200-400", "400-700", "700-1200", ">=1200"])
    cut("median_candidate_distance", med_d, [0, .1, .25, .45, .65], ["neg", "0-.1", ".1-.25", ".25-.45", ".45-.65", ">=.65"])
    cut("t0_known_count", nk_known, [0, 1, 5, 15, 40], ["neg", "0", "1-4", "5-14", "15-39", ">=40"])
    cut("pool_recall_ia", precall, [0, .001, .25, .5, .75], ["neg", "0", "0-.25", ".25-.5", ".5-.75", ">=.75"])
    cut("pool_size", psize, [0, 50, 100, 200, 400], ["neg", "0-49", "50-99", "100-199", "200-399", ">=400"])
    cut("closest_neighbour_taxrel", best_rel, [0, 1, 2, 3, 4, 5], ["same", "close", "intermediate", "distant", "root-only", "no-donor", "x"])

    # ---- TAXON cut (NEW) ----
    txarr = np.array(taxid, dtype=object)
    txc = Counter(txarr[reach])
    tot_r = headroom[reach].sum()
    taxrows = []
    for tx, cnt in txc.most_common():
        s = (txarr == tx) & reach
        taxrows.append({"taxid": tx, "n": int(s.sum()),
                        "headroom_sum": round(float(headroom[s].sum()), 4),
                        "headroom_share": round(float(headroom[s].sum() / tot_r), 4) if tot_r else None,
                        "headroom_per_protein_x1e4": round(float(headroom[s].mean()) * 1e4, 3),
                        "mean_delivered_f": round(float(delivered_f[s].mean()), 4),
                        "mean_oracle_f": round(float(oracle_f[s].mean()), 4),
                        "mean_pool_recall": round(float(precall[s].mean()), 4)})
    out.setdefault("cuts", {})["TAXON"] = taxrows

    # ---- missed-term IA band + DAG depth (reachable proteins) ----
    missR = ((Pm < tau_star) & Gv)[reach]
    in_poolR = (O[reach] > 0)
    cells_all = np.nonzero(missR)
    mi_ia, mi_dep = ia_w[cells_all[1]], depth_toi[cells_all[1]]
    cells_rk = np.nonzero(missR & in_poolR)
    rk_ia, rk_dep = ia_w[cells_rk[1]], depth_toi[cells_rk[1]]
    ia_e, ia_l = [0, 1, 3, 6, 10], ["<0", "0-1", "1-3", "3-6", "6-10", ">10"]
    d_e, d_l = [0, 3, 5, 7, 9], ["<0", "0-2", "3-4", "5-6", "7-8", ">=9"]
    for nm, vals, w_, e, l in [("missed_IA_band_ALL", mi_ia, mi_ia, ia_e, ia_l),
                               ("missed_IA_band_RANKER_FIXABLE", rk_ia, rk_ia, ia_e, ia_l),
                               ("missed_DAG_depth_ALL", mi_dep, mi_ia, d_e, d_l),
                               ("missed_DAG_depth_RANKER_FIXABLE", rk_dep, rk_ia, d_e, d_l)]:
        b = np.digitize(vals, e); tt = w_.sum()
        out.setdefault("cuts", {})[nm] = [
            {"band": l[k], "n_cells": int((b == k).sum()), "fn_ia_mass": round(float(w_[b == k].sum()), 1),
             "share": round(float(w_[b == k].sum() / tt), 4) if tt else None}
            for k in range(len(l)) if (b == k).sum()]

    return out


ALL = {}
for cell, cfg in CELLS.items():
    ALL[cell + "-bpo"] = run_cell(cell, cfg)
json.dump(ALL, open(W / "bp_failure_atlas.json", "w"), indent=1)
print("\nDONE bp_failure_atlas.json", flush=True)
