"""INSTRUMENT 1: the mechanism. Why is cafa_eval not invariant to a monotone rescale?

GATE, written before the number:
  H4 = zero-clamp x prop="fill". Under raw, an in-pool candidate with score <= 0 is stored
  as 0.0 ("not predicted"), so fill OVERWRITES it with max(descendants). Under minmax the
  same cell is a small positive ("predicted"), so fill RESTORES its own low value and blocks
  inheritance (graph.py:320-323).
  KILL A: under prop="max" the two arms must agree at a matched cut. If they diverge, H4 dead.
  KILL B: under prop="fill" raw's set must be a strict SUPERSET of minmax's, and the extra
          cells must be in-pool cells with own raw <= 0 having a descendant above the cut.
  KILL C: prefilter raw>0 then rescale must reproduce raw's f.

No repo code is touched. Everything is driven through the installed cafaeval's own functions.
"""
import json, sys, time
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path

sys.path.insert(0, "/home/frapercan/Thesis2/repositories/PROTEA/.venv/lib/python3.12/site-packages")
from cafaeval.parser import obo_parser, gt_parser, _pred_parser_legacy
from cafaeval.graph import propagate

W = Path("/home/frapercan/Thesis2/storage/cooc_experiment")
DS = Path("/home/frapercan/Thesis2/repositories/protea-reranker-lab/datasets/protst-global-train227-test230")
OBO = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
IA = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv"
NS = "biological_process"
RAW_TAU = 0.393
out = {"gate": "H4 = zero-clamp x prop=fill; KILL A: prop=max must agree at matched cut"}

t0 = time.time()
print("[..] parsing obo (no_orphans=True -> orphans=False)", flush=True)
onts = obo_parser(OBO, ("is_a", "part_of"), IA, False)
ont = onts[NS]
print(f"[ok] obo {time.time()-t0:.0f}s  terms={ont.idxs}  toi={len(ont.toi)}  toi_ia={len(ont.toi_ia)}", flush=True)

gts = gt_parser(str(DS / "gt_pk_bp.tsv"), onts)
gt = gts[NS]
G = gt.matrix
print(f"[ok] gt proteins={len(gt.ids)} propagated_annots={int(G.sum())}", flush=True)

# ---- the frozen pool -------------------------------------------------------------
t = pq.read_table(W / "rerank_out" / "eval_scores.parquet")
cat = np.asarray(t.column("category").to_pylist())
asp = np.asarray(t.column("aspect").to_pylist())
m = (cat == "pk") & (asp == "bpo")
prot = np.asarray(t.column("protein_accession").to_pylist())[m]
go = np.asarray(t.column("go_term_id").to_pylist())[m]
raw = t.column("reranker_score").to_numpy(zero_copy_only=False).astype(np.float64)[m]
lo, hi = float(raw.min()), float(raw.max())
MM_TAU = (RAW_TAU - lo) / (hi - lo)
out["pool"] = {"rows": int(m.sum()), "lo": round(lo, 4), "hi": round(hi, 4),
               "frac_nonpositive": round(float((raw <= 0).mean()), 4),
               "raw_tau": RAW_TAU, "matched_minmax_tau": round(MM_TAU, 6)}
print(f"[ok] pool rows={m.sum()} lo={lo:.4f} hi={hi:.4f} matched mm_tau={MM_TAU:.6f}", flush=True)

# pool coordinates in matrix space (rows kept only if protein in gt and term in ontology)
ti = ont.terms_dict
gid = gt.ids
rows = np.array([gid.get(p, -1) for p in prot], dtype=np.int64)
cols = np.array([ti[g]["index"] if g in ti else -1 for g in go], dtype=np.int64)
keep = (rows >= 0) & (cols >= 0)
p_rows, p_cols, p_raw = rows[keep], cols[keep], raw[keep]
out["pool"]["rows_landing_in_matrix"] = int(keep.sum())
print(f"[ok] pool cells landing in matrix: {keep.sum()}", flush=True)

IA_V = ont.ia
toi_ia = ont.toi_ia
toi_mask = np.zeros(ont.idxs, dtype=bool); toi_mask[toi_ia] = True
# total gt weight over toi_ia == n_gt.sum() in compute_metrics
TOTAL_GT_W = float((G[:, toi_ia] * IA_V[toi_ia]).sum())
out["total_gt_weight"] = round(TOTAL_GT_W, 2)


def build_matrix(scores, prop):
    """Exactly reproduce pred_parser(max_terms=500) -> legacy path -> propagate(prop)."""
    import tempfile, os
    mat = {NS: np.zeros(G.shape, dtype="float")}
    row_nnz = {NS: np.zeros(G.shape[0], dtype=np.int32)}
    ids = {NS: {}}
    ns_dict = {tt: NS for tt in ont.terms_dict}
    ns_dict.update({tt: NS for tt in ont.terms_dict_alt})
    term_index = {NS: {tt: i["index"] for tt, i in ont.terms_dict.items()}}
    with tempfile.TemporaryDirectory() as td:
        f = os.path.join(td, "p.tsv")
        with open(f, "w") as fh:
            for p, g, v in zip(prot, go, scores):
                fh.write(f"{p}\t{g}\t{v:.6f}\n")
        _pred_parser_legacy(f, onts, gts, ns_dict, term_index, ids, mat, row_nnz, {}, 500)
    parsed = mat[NS].copy()
    propagate(mat[NS], ont, ont.order, mode=prop, parallel=1)
    return parsed, mat[NS], len(ids[NS])


def micro_at(matrix, tau):
    """Faithful re-implementation of the module's weighted micro metrics at one cut.
    compute_confusion_matrix_sparse: active iff score >= tau; tp/pred weighted by ia[col];
    fn = total_gt_w - tp_w; normalize() divides tp/fp/fn by a CONSTANT ne, which cancels in
    pr_micro=tp/(tp+fp) and rc_micro=tp/(tp+fn)."""
    sub = matrix[:, toi_ia]
    act = sub >= tau
    w = IA_V[toi_ia]
    pred_w = float((act * w).sum())
    tp_w = float((act * (G[:, toi_ia] != 0) * w).sum())
    pr = tp_w / pred_w if pred_w > 0 else 0.0
    rc = tp_w / TOTAL_GT_W if TOTAL_GT_W > 0 else 0.0
    f = 2 * pr * rc / (pr + rc) if (pr + rc) > 0 else 0.0
    n_prot = int(act.any(axis=1).sum())
    return {"f_micro_w": round(f, 4), "pr_micro_w": round(pr, 4), "rc_micro_w": round(rc, 4),
            "n_cells": int(act.sum()), "pred_w": round(pred_w, 1), "tp_w": round(tp_w, 1),
            "n_proteins_scored": n_prot}


mm = (raw - lo) / (hi - lo)
res = {}
sets = {}
for prop in ("fill", "max"):
    for name, sc, tau in (("raw", raw, RAW_TAU), ("minmax", mm, MM_TAU)):
        t1 = time.time()
        parsed, propd, nids = build_matrix(sc, prop)
        k = f"{name}|{prop}"
        res[k] = {"after_parse": micro_at(parsed, tau), "after_prop": micro_at(propd, tau),
                  "parse_nonzero_cells": int(np.count_nonzero(parsed)),
                  "prop_nonzero_cells": int(np.count_nonzero(propd)),
                  "ids_registered": nids}
        a = propd[:, toi_ia] >= tau
        sets[k] = np.flatnonzero(a.ravel())
        del parsed, propd, a
        print(f"[ok] {k:14s} {time.time()-t1:.0f}s  parse={res[k]['after_parse']['f_micro_w']} "
              f"prop={res[k]['after_prop']['f_micro_w']} cells={res[k]['after_prop']['n_cells']}", flush=True)
out["arms"] = res

# ---- KILL A -----------------------------------------------------------------------
fa_r, fa_m = res["raw|max"]["after_prop"]["f_micro_w"], res["minmax|max"]["after_prop"]["f_micro_w"]
sa_r, sa_m = set(sets["raw|max"].tolist()), set(sets["minmax|max"].tolist())
out["KILL_A_prop_max"] = {"raw_f": fa_r, "minmax_f": fa_m, "sets_identical": sa_r == sa_m,
                          "sym_diff": len(sa_r ^ sa_m),
                          "verdict": "H4 SURVIVES" if sa_r == sa_m else "H4 DEAD"}
print(f"\n[KILL A] prop=max  raw={fa_r} minmax={fa_m} identical={sa_r == sa_m} "
      f"symdiff={len(sa_r ^ sa_m)}", flush=True)

# ---- KILL B -----------------------------------------------------------------------
sf_r, sf_m = set(sets["raw|fill"].tolist()), set(sets["minmax|fill"].tolist())
extra_r = sf_r - sf_m
extra_m = sf_m - sf_r
n_toi = len(toi_ia)
pool_raw_at = {}
for r, c, v in zip(p_rows, p_cols, p_raw):
    if toi_mask[c]:
        cc = int(np.searchsorted(toi_ia, c))
        pool_raw_at[int(r) * n_toi + cc] = max(pool_raw_at.get(int(r) * n_toi + cc, -1e9), float(v))
in_pool = sum(1 for x in extra_r if x in pool_raw_at)
in_pool_nonpos = sum(1 for x in extra_r if pool_raw_at.get(x, 1.0) <= 0)
not_in_pool = len(extra_r) - in_pool
tp_extra = sum(1 for x in extra_r if G[x // n_toi, toi_ia[x % n_toi]])
w_extra = float(sum(IA_V[toi_ia[x % n_toi]] for x in extra_r))
w_extra_tp = float(sum(IA_V[toi_ia[x % n_toi]] for x in extra_r
                       if G[x // n_toi, toi_ia[x % n_toi]]))
out["KILL_B_prop_fill"] = {
    "raw_f": res["raw|fill"]["after_prop"]["f_micro_w"],
    "minmax_f": res["minmax|fill"]["after_prop"]["f_micro_w"],
    "raw_is_superset": len(extra_m) == 0,
    "n_extra_in_raw": len(extra_r), "n_extra_in_minmax": len(extra_m),
    "extra_in_pool": in_pool, "extra_in_pool_with_own_raw_le_0": in_pool_nonpos,
    "extra_not_in_pool": not_in_pool,
    "extra_that_are_TP": tp_extra,
    "extra_TP_rate": round(tp_extra / max(1, len(extra_r)), 4),
    "extra_ia_weight": round(w_extra, 1), "extra_ia_weight_TP": round(w_extra_tp, 1),
    "extra_weighted_precision": round(w_extra_tp / max(1e-9, w_extra), 4),
    "verdict": ("H4 SURVIVES" if len(extra_m) == 0 and in_pool_nonpos > 0.9 * len(extra_r)
                else "H4 DEAD or INCOMPLETE"),
}
print(f"[KILL B] prop=fill raw_superset={len(extra_m)==0} extra_in_raw={len(extra_r)} "
      f"in_pool_nonpos={in_pool_nonpos} not_in_pool={not_in_pool} "
      f"TP_rate={out['KILL_B_prop_fill']['extra_TP_rate']}", flush=True)

# print the actual differing rows, as instructed
inv_p = {v: k for k, v in gt.ids.items()}
tl = ont.terms_list
ex = sorted(extra_r)[:25]
rowsout = []
for x in ex:
    r, cc = x // n_toi, x % n_toi
    c = int(toi_ia[cc])
    rowsout.append({"protein": inv_p[int(r)], "term": tl[c]["id"], "name": tl[c]["name"][:44],
                    "ia": round(float(IA_V[c]), 3),
                    "own_raw_in_pool": (round(pool_raw_at[x], 4) if x in pool_raw_at else None),
                    "in_gt_TP": bool(G[r, c])})
out["KILL_B_sample_rows_only_in_raw"] = rowsout
print("\n--- rows in RAW's set but NOT in MINMAX's, at the matched cut ---", flush=True)
for d in rowsout[:15]:
    print(f"  {d['protein']:12s} {d['term']} ia={d['ia']:6.3f} own_raw="
          f"{str(d['own_raw_in_pool']):>9s} TP={str(d['in_gt_TP']):5s} {d['name']}", flush=True)

json.dump(out, open(W / "instrument_mechanism.json", "w"), indent=1, default=str)
print(f"\nDONE {time.time()-t0:.0f}s", flush=True)
