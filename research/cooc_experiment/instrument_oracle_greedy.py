"""INSTRUMENT 2b: the oracle, tightened. The strict oracle (0.7519) is NOT the max over
orderings, because instrument_oracle.py measured a real loophole: 8861 of 80039 gt weight
(11.1%) is reachable ONLY through the ancestor closure of pool cells that are themselves FPs.
Since "best f_micro_w any ORDERING could achieve" == max over PREFIX SETS == max over arbitrary
SUBSETS of the pool, an oracle that refuses to spend an FP to buy a true ancestor is leaving
score on the table and is therefore a lower bound, not the ceiling.

So: greedily buy FP cells by (gt weight gained)/(fp weight paid), and evaluate f EXACTLY at
each prefix. Greedy is not optimal, so every number here is an ACHIEVABLE LOWER BOUND on the
true ceiling. The unreachable-limited UPPER bound is 2*Rmax/(1+Rmax) with Rmax=0.7132.

GATE: the greedy must start at exactly the strict oracle (f=0.7519, pr=1.0) at prefix 0. If it
does not, the accounting is broken and I say so.
"""
import json, sys, time
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path

sys.path.insert(0, "/home/frapercan/Thesis2/repositories/PROTEA/.venv/lib/python3.12/site-packages")
from cafaeval.parser import obo_parser, gt_parser
from cafaeval.graph import _ancestors_csr

W = Path("/home/frapercan/Thesis2/storage/cooc_experiment")
DS = Path("/home/frapercan/Thesis2/repositories/protea-reranker-lab/datasets/protst-global-train227-test230")
OBO = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
IA = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv"
NS = "biological_process"

t0 = time.time()
onts = obo_parser(OBO, ("is_a", "part_of"), IA, False)
ont = onts[NS]
gts = gt_parser(str(DS / "gt_pk_bp.tsv"), onts)
gt = gts[NS]; G = gt.matrix
IA_V = ont.ia; toi_ia = ont.toi_ia
tmask = np.zeros(ont.idxs, dtype=bool); tmask[toi_ia] = True
TOTAL_GT_W = float((G[:, toi_ia] * IA_V[toi_ia]).sum())

t = pq.read_table(W / "rerank_out" / "eval_scores.parquet")
cat = np.asarray(t.column("category").to_pylist()); asp = np.asarray(t.column("aspect").to_pylist())
m = (cat == "pk") & (asp == "bpo")
prot = np.asarray(t.column("protein_accession").to_pylist())[m]
go = np.asarray(t.column("go_term_id").to_pylist())[m]
ti = ont.terms_dict
rows = np.array([gt.ids.get(p, -1) for p in prot], dtype=np.int64)
cols = np.array([ti[g]["index"] if g in ti else -1 for g in go], dtype=np.int64)
keep = (rows >= 0) & (cols >= 0)
p_rows, p_cols = rows[keep], cols[keep]
indptr, anc = _ancestors_csr(ont)
print(f"[ok] setup {time.time()-t0:.0f}s pool={p_rows.size} total_gt_w={TOTAL_GT_W:.1f}", flush=True)


def expand(r, c):
    n_anc = (indptr[c + 1] - indptr[c]).astype(np.int64)
    tot = int(n_anc.sum())
    base = np.repeat(indptr[c], n_anc)
    goff = np.zeros(r.size + 1, dtype=np.int64); np.cumsum(n_anc, out=goff[1:])
    loc = np.arange(tot, dtype=np.int64) - np.repeat(goff[:-1], n_anc)
    return np.repeat(r, n_anc), anc[base + loc]


is_tp_cell = G[p_rows, p_cols] != 0
# baseline: strict oracle predicted set
er, ec = expand(p_rows[is_tp_cell], p_cols[is_tp_cell])
PRED = np.zeros(G.shape, dtype=bool); PRED[er, ec] = True
GT_B = (G != 0)


def metrics(pred):
    p = pred & tmask[None, :]
    pw = float((p * IA_V[None, :]).sum())
    tw = float((p & GT_B) * IA_V[None, :]).sum() if False else float(((p & GT_B) * IA_V[None, :]).sum())
    pr = tw / pw if pw > 0 else 0.0
    rc = tw / TOTAL_GT_W
    return pr, rc, (2 * pr * rc / (pr + rc) if pr + rc > 0 else 0.0)


pr0, rc0, f0 = metrics(PRED)
out = {"gate": "greedy prefix 0 must equal the strict oracle f=0.7519 pr=1.0",
       "prefix0": {"pr": round(pr0, 4), "rc": round(rc0, 4), "f": round(f0, 4)},
       "total_gt_weight": round(TOTAL_GT_W, 2)}
out["GATE_verdict"] = "OK" if abs(f0 - 0.7519) < 5e-4 and abs(pr0 - 1.0) < 1e-6 else "BROKEN ACCOUNTING"
print(f"[GATE] prefix0 pr={pr0:.4f} rc={rc0:.4f} f={f0:.4f} -> {out['GATE_verdict']}", flush=True)

# --- score every FP pool cell against the strict-oracle baseline ---------------------
fp_idx = np.flatnonzero(~is_tp_cell)
gain = np.zeros(fp_idx.size); cost = np.zeros(fp_idx.size)
CH = 20000
for s in range(0, fp_idx.size, CH):
    sl = fp_idx[s:s + CH]
    r, c = p_rows[sl], p_cols[sl]
    n_anc = (indptr[c + 1] - indptr[c]).astype(np.int64)
    er, ec = expand(r, c)
    grp = np.repeat(np.arange(sl.size), n_anc)
    inb = tmask[ec]
    new = inb & (~PRED[er, ec])
    w = IA_V[ec] * new
    gain[s:s + CH] = np.bincount(grp, weights=w * GT_B[er, ec], minlength=sl.size)
    cost[s:s + CH] = np.bincount(grp, weights=w * (~GT_B[er, ec]), minlength=sl.size)
print(f"[ok] scored {fp_idx.size} FP cells {time.time()-t0:.0f}s; "
      f"with_gain={int((gain>0).sum())} free={int(((gain>0)&(cost<=0)).sum())}", flush=True)

ratio = gain / np.maximum(cost, 1e-9)
order = np.argsort(-ratio, kind="stable")
order = order[gain[order] > 0]
out["n_fp_cells_with_any_gain"] = int(order.size)

# --- walk the greedy order, recomputing f EXACTLY at prefix checkpoints --------------
pts = sorted(set(np.unique(np.linspace(0, order.size, 120).astype(int)).tolist()))
cur = PRED.copy(); traj = []; best = {"f": f0, "k": 0}
prev = 0
for k in pts:
    if k > prev:
        sl = fp_idx[order[prev:k]]
        er, ec = expand(p_rows[sl], p_cols[sl])
        cur[er, ec] = True
        prev = k
    pr, rc, f = metrics(cur)
    traj.append({"k_fp_added": int(k), "pr": round(pr, 4), "rc": round(rc, 4), "f": round(f, 4)})
    if f > best["f"]:
        best = {"f": round(f, 4), "k": int(k), "pr": round(pr, 4), "rc": round(rc, 4)}
out["greedy_trajectory"] = traj
out["greedy_best"] = best
Rmax = 0.7132
out["upper_bound_unreachable_limited"] = round(2 * Rmax / (1 + Rmax), 4)
out["verdict"] = (f"true pool ceiling is in [{best['f']}, {out['upper_bound_unreachable_limited']}]; "
                  f"greedy is achievable so the LEFT edge is a real lower bound")
print("\n--- greedy trajectory (FP cells bought by gain/cost) ---", flush=True)
for d in traj[::6]:
    print(f"  +{d['k_fp_added']:7d} FP  pr={d['pr']:.4f} rc={d['rc']:.4f} f={d['f']:.4f}", flush=True)
print(f"\n[BEST] greedy oracle f={best['f']} at +{best['k']} FP cells "
      f"(strict oracle was {round(f0,4)})", flush=True)
print(f"[BOUND] {out['verdict']}", flush=True)
json.dump(out, open(W / "instrument_oracle_greedy.json", "w"), indent=1, default=str)
print(f"DONE {time.time()-t0:.0f}s", flush=True)
