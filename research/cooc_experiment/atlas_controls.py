"""PKW.3 controls. Three things in atlas_build.py's first read were suspects; this
script checks each one instead of quoting it.

C1. THE TAXON PROXY WAS VACUOUS, NOT UNIFORM. atlas_build banded on
    taxonomic_relation == "same_species". That string does not exist: the real values are
    {same, close, intermediate, distant, root-only, ""}. The cut collapsed to one band and
    a vacuous cut reads exactly like a uniform one. Rebanded here on the real vocabulary.

C2. THE "NEGATIVE" neighbour distances are float noise (min = -3.6e-07), i.e. exact
    neighbour hits, not a sign error. But the axis is still mis-banded: every per-protein
    min distance lands in [0, 0.2], so atlas_build's [0,.2,.4,.6,.8] grid was two bands
    wide and could not have shown a gradient even if one existed. Rebanded on the actual
    per-protein quantiles, which is the only way this axis can testify.

C3. IS THE "POOL ORACLE" ACTUALLY AN ORACLE? atlas_build built it from eval.parquet's
    `label`. cafaeval propagates BOTH gt and predictions (prop=fill), so a pool term with
    label=0 can still sit in the PROPAGATED ground truth (it is an ancestor of a true
    term). If so, `label` understates the pool and 0.6077 is not the pool's ceiling.
    This also predicts atlas_build's 594 proteins with NEGATIVE headroom: a deployed FP
    can propagate into a true ancestor and score TP, which a label-built oracle never
    gets credit for. Measured here against a gt-membership oracle.
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from cafaeval.parser import obo_parser, gt_parser, pred_parser

W = Path("/home/frapercan/Thesis2/storage/cooc_experiment")
DS = Path("/home/frapercan/Thesis2/repositories/protea-reranker-lab/datasets/protst-global-train227-test230")
OBO = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
IA = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv"
NS = "biological_process"
TAU = np.arange(0.001, 1, 0.001)
out = {"what": "PKW.3 controls: taxon vocabulary, distance banding, and is the oracle an oracle"}

rec = json.load(open(W / "atlas_per_protein.json"))
head = np.array([r["headroom"] for r in rec])
reach = np.array([r["reachable"] for r in rec])
prot_order = [r["protein"] for r in rec]

ev = pq.read_table(DS / "eval.parquet",
                   columns=["protein_accession", "go_term_id", "category", "aspect", "label",
                            "taxonomic_relation", "neighbor_min_distance"])
c = np.asarray(ev.column("category").to_pylist())
a = np.asarray(ev.column("aspect").to_pylist())
m = (c == "pk") & (a == "bpo")
prot = np.asarray(ev.column("protein_accession").to_pylist())[m]
go = np.asarray(ev.column("go_term_id").to_pylist())[m]
label = ev.column("label").to_numpy(zero_copy_only=False).astype(np.float64)[m]
taxrel = np.asarray(ev.column("taxonomic_relation").to_pylist())[m]
nmd = ev.column("neighbor_min_distance").to_numpy(zero_copy_only=False).astype(np.float64)[m]

pool_idx = {}
for i, p in enumerate(prot):
    pool_idx.setdefault(p, []).append(i)

# ---------------------------------------------------------------- C1 + C2 cuts
RANK = {"same": 0, "close": 1, "intermediate": 2, "distant": 3, "root-only": 4}


def percut(name, vals, edges, labels):
    rows = []
    v = np.asarray(vals, dtype=np.float64)
    b = np.digitize(v, edges)
    tot = head[reach].sum()
    for k, lab in enumerate(labels):
        s = (b == k) & reach
        if s.sum() == 0:
            continue
        rows.append({"band": lab, "n": int(s.sum()),
                     "headroom_sum": round(float(head[s].sum()), 4),
                     "headroom_share": round(float(head[s].sum() / tot), 4),
                     "headroom_per_protein_x1e4": round(float(head[s].mean()) * 1e4, 3)})
    out.setdefault("cuts_fixed", {})[name] = rows


# C1: the closest taxonomic relation any candidate for this protein came from.
best_rel = []
frac_same = []
for p in prot_order:
    ix = pool_idx.get(p, [])
    rr = [RANK[x] for x in taxrel[ix] if x in RANK] if ix else []
    best_rel.append(min(rr) if rr else 9)
    frac_same.append(float((taxrel[ix] == "same").mean()) if ix else 0.0)
out["C1_taxonomic_relation_vocabulary"] = dict(Counter(taxrel.tolist()).most_common())
out["C1_note"] = ("'' = 43.4% of pool rows: candidates with no KNN donor (classifier / interpro / "
                  "protst channels). This is a NEIGHBOUR-taxonomy axis. The QUERY's own taxon is "
                  "absent from every frozen input (no taxid in eval.parquet, no OX= in "
                  "lafa_queries_7401.fasta, none in the CAFA_forever release files), so the "
                  "taxon axis asked for in the brief CANNOT be cut from frozen data.")
percut("closest_neighbour_taxonomic_relation", best_rel, [0, 1, 2, 3, 4, 5],
       ["same", "close", "intermediate", "distant", "root-only", "no-knn-donor", "x"])
percut("frac_same_taxon_candidates_REAL", frac_same, [0, .001, .1, .25, .5],
       ["neg", "0", "0-0.1", "0.1-0.25", "0.25-0.5", ">0.5"])

# C2: reband neighbour distance on the ACTUAL per-protein min distribution.
pmin = np.array([float(np.nanmin(nmd[pool_idx[p]])) if pool_idx.get(p) and
                 not np.all(np.isnan(nmd[pool_idx[p]])) else np.nan for p in prot_order])
qs = np.nanpercentile(pmin[reach], [20, 40, 60, 80])
out["C2_per_protein_min_distance"] = {
    "min": round(float(np.nanmin(pmin)), 6), "max": round(float(np.nanmax(pmin)), 4),
    "pct_below_0.2": round(float((pmin[reach] < 0.2).mean()) * 100, 2),
    "quintile_edges": [round(float(x), 5) for x in qs],
    "note": "atlas_build banded this on [0,.2,.4,.6,.8]; the whole distribution lives inside "
            "the first two bands, so that grid could not have resolved a gradient.",
}
percut("neighbour_min_distance_QUINTILES", pmin, list(qs), ["Q1_closest", "Q2", "Q3", "Q4", "Q5_farthest"])

# ---------------------------------------------------------------- C3 the oracle audit
ont = obo_parser(OBO, ("is_a", "part_of"), IA, False)[NS]
gts = gt_parser(str(DS / "gt_pk_bp.tsv"), {NS: ont})
gt = gts[NS]
toi = ont.toi_ia
ia_w = ont.ia[toi].astype(np.float64)
G = gt.matrix[:, toi]

# is each pool row's term in the PROPAGATED gt for its protein?
tidx = {t: i["index"] for t, i in ont.terms_dict.items()}
col_of = np.array([tidx.get(g, -1) for g in go])
row_of = np.array([gt.ids.get(p, -1) for p in prot])
ok = (col_of >= 0) & (row_of >= 0)
in_gt = np.zeros(len(go), dtype=bool)
in_gt[ok] = gt.matrix[row_of[ok], col_of[ok]]
out["C3_pool_rows"] = {
    "n_rows": int(len(go)),
    "n_label_1": int((label > 0).sum()),
    "n_in_propagated_gt": int(in_gt.sum()),
    "n_label0_but_in_propagated_gt": int(((label == 0) & in_gt).sum()),
    "n_label1_but_not_in_propagated_gt": int(((label > 0) & ~in_gt).sum()),
}

scratch = W / "atlas_scratch"
(scratch / "orc2").mkdir(parents=True, exist_ok=True)
with open(scratch / "orc2" / "o.tsv", "w") as fh:
    for p, g, v in zip(prot, go, in_gt.astype(np.float64)):
        fh.write(f"{p}\t{g}\t{v:.6f}\n")
O2 = pred_parser(str(scratch / "orc2" / "o.tsv"), {NS: ont}, gts, "fill", 500, 1)[NS].matrix[:, toi]
has_gt = G.any(axis=1)
G2, O2 = G[has_gt], O2[has_gt]


def sweep(pred, g):
    gw = float((g @ ia_w).sum())
    r, cc = np.nonzero(pred)
    s = pred[r, cc]
    last = np.searchsorted(TAU, s, side="right") - 1
    k = last >= 0
    r, cc, last = r[k], cc[k], last[k]
    w = ia_w[cc]
    wtp = w * g[r, cc].astype(np.float64)
    n = len(TAU)
    pa = np.cumsum(np.bincount(last, weights=w, minlength=n)[::-1])[::-1]
    ta = np.cumsum(np.bincount(last, weights=wtp, minlength=n)[::-1])[::-1]
    pr = np.divide(ta, pa, out=np.zeros(n), where=pa > 0)
    rc = ta / gw
    f = np.divide(2 * pr * rc, pr + rc, out=np.zeros(n), where=(pr + rc) > 0)
    i = int(np.argmax(f))
    return round(float(f[i]), 4), round(float(TAU[i]), 3)


f2, t2 = sweep(O2, G2)
reach2 = ((O2 > 0) & G2).any(axis=1)
f2r, t2r = sweep(O2[reach2], G2[reach2])
out["C3_oracle"] = {
    "label_built_oracle_f_micro_w": 0.6077,
    "gt_membership_oracle_f_micro_w": f2, "gt_membership_oracle_tau": t2,
    "delta": round(f2 - 0.6077, 4),
    "n_reachable_label_built": int(reach.sum()),
    "n_reachable_gt_membership": int(reach2.sum()),
    "gt_membership_oracle_reachable_only_f": f2r,
}
json.dump(out, open(W / "atlas_controls.json", "w"), indent=1)
print(json.dumps(out, indent=1), flush=True)
print("DONE", flush=True)
