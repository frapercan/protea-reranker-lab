"""INSTRUMENT 2: the correct oracle. What is the best f_micro_w any ordering of this pool
could achieve, under the SAME cafa_eval call we report with?

The definition is forced by the evaluator, not chosen by us:
  - cafaeval defines TP as g[row,col] where g is the PROPAGATED gt (parser.py:174). So a pool
    term that is an ancestor of a true term IS a TP. The 0.6077 number (direct labels only)
    contradicts the evaluator's own TP rule and is therefore not an upper bound on anything.
  - "ancestor of everything is vacuous" is answered by the metric itself: f_micro_w restricts
    to toi_ia (ia > 0) and weights by ia. The BP root has ia == 0, so vacuous ancestors carry
    ZERO weight in both the tp and the pred sum. Reachability via a shallow ancestor is not
    free -- it is worth exactly its information content.
  - prop="fill" means submitting a cell also submits its whole ancestor closure. Since the gt
    is ancestor-closed, submitting ONLY the true pool cells yields a predicted set that is
    entirely inside the gt => precision is EXACTLY 1. So the best ordering is "all TPs above
    all FPs" and the best cut takes exactly the TPs:
        f_oracle = 2R/(1+R),  R = ia-weight(closure(pool AND gt)) / ia-weight(gt)
    Adding any FP strictly lowers pr and cannot raise rc-from-its-own-cell. The only loophole
    is an FP pool cell whose ANCESTOR is a true term not otherwise reachable; that is measured
    below as the "greedy loophole" and bounded, not assumed away.

GATE: if precision of the strict-oracle submission is not 1.0000 when run through the real
cafa_eval, this whole derivation is wrong and I say so.
"""
import json, os, subprocess, sys, tempfile, time
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
PY = "/home/frapercan/Thesis2/repositories/PROTEA/.venv/bin/python"
NS = "biological_process"
out = {"gate": "strict-oracle precision must be 1.0000 through the real cafa_eval"}

t0 = time.time()
onts = obo_parser(OBO, ("is_a", "part_of"), IA, False)
ont = onts[NS]
gts = gt_parser(str(DS / "gt_pk_bp.tsv"), onts)
gt = gts[NS]
G = gt.matrix
IA_V = ont.ia
toi_ia = ont.toi_ia
toi_mask = np.zeros(ont.idxs, dtype=bool); toi_mask[toi_ia] = True
TOTAL_GT_W = float((G[:, toi_ia] * IA_V[toi_ia]).sum())
print(f"[ok] setup {time.time()-t0:.0f}s gt_prot={len(gt.ids)} total_gt_w={TOTAL_GT_W:.1f} "
      f"terms={ont.idxs} toi_ia={len(toi_ia)}", flush=True)
out["total_gt_weight"] = round(TOTAL_GT_W, 2)
out["n_gt_proteins"] = len(gt.ids)

t = pq.read_table(W / "rerank_out" / "eval_scores.parquet")
cat = np.asarray(t.column("category").to_pylist())
asp = np.asarray(t.column("aspect").to_pylist())
m = (cat == "pk") & (asp == "bpo")
prot = np.asarray(t.column("protein_accession").to_pylist())[m]
go = np.asarray(t.column("go_term_id").to_pylist())[m]
lab = t.column("label").to_numpy(zero_copy_only=False).astype(np.int64)[m]

ti = ont.terms_dict
rows = np.array([gt.ids.get(p, -1) for p in prot], dtype=np.int64)
cols = np.array([ti[g]["index"] if g in ti else -1 for g in go], dtype=np.int64)
keep = (rows >= 0) & (cols >= 0)
p_rows, p_cols, p_lab = rows[keep], cols[keep], lab[keep]
print(f"[ok] pool cells in matrix: {keep.sum()} / {m.sum()}", flush=True)

indptr, anc = _ancestors_csr(ont)


def closure_weight(sel):
    """ia-weight of the gt cells covered by the ancestor closure of the selected pool cells,
    and the closure's own size/weight. Returns (cov_w, closure_cells, closure_w, cov_matrix)."""
    r, c = p_rows[sel], p_cols[sel]
    n_anc = (indptr[c + 1] - indptr[c]).astype(np.int64)
    total = int(n_anc.sum())
    base = np.repeat(indptr[c], n_anc)
    goff = np.zeros(r.size + 1, dtype=np.int64); np.cumsum(n_anc, out=goff[1:])
    loc = np.arange(total, dtype=np.int64) - np.repeat(goff[:-1], n_anc)
    ec = anc[base + loc]
    er = np.repeat(r, n_anc)
    M = np.zeros(G.shape, dtype=bool)
    M[er, ec] = True
    covered = M & (G != 0) & toi_mask[None, :]
    cov_w = float((covered * IA_V[None, :]).sum())
    clo_w = float(((M & toi_mask[None, :]) * IA_V[None, :]).sum())
    return cov_w, int((M & toi_mask[None, :]).sum()), clo_w, covered


def f_from_R(R):
    return 2 * R / (1 + R) if R > 0 else 0.0


gt_cell = np.zeros(p_rows.size, dtype=bool)
gt_cell[:] = G[p_rows, p_cols] != 0

# --- A. STRICT ORACLE: submit exactly the pool cells that are in the propagated gt --------
covA, nA, cloA, coveredA = closure_weight(gt_cell)
RA = covA / TOTAL_GT_W
out["A_strict_oracle"] = {
    "definition": "submit pool cells in the PROPAGATED gt at 1.0; prop=fill closes ancestors",
    "pool_cells_submitted": int(gt_cell.sum()),
    "closure_cells_in_toi_ia": nA,
    "precision_by_construction": round(cloA / max(1e-9, cloA), 4),
    "reachable_gt_weight": round(covA, 2),
    "recall_R": round(RA, 4), "f_micro_w": round(f_from_R(RA), 4),
}
print(f"[A] strict oracle R={RA:.4f} f={f_from_R(RA):.4f}  (closure_w={cloA:.1f} cov_w={covA:.1f})", flush=True)

# --- B. the 0.6077-style oracle: DIRECT labels only ---------------------------------------
covB, nB, cloB, _ = closure_weight(p_lab > 0)
RB = covB / TOTAL_GT_W
out["B_direct_label_oracle"] = {
    "definition": "submit pool cells with label==1 (DIRECT annotations) -- the 0.6077 lineage",
    "pool_cells_submitted": int((p_lab > 0).sum()),
    "recall_R": round(RB, 4), "f_micro_w": round(f_from_R(RB), 4),
    "note": "same closure trick applies; this is a SUBSET of A's submission",
}
print(f"[B] direct-label oracle R={RB:.4f} f={f_from_R(RB):.4f}", flush=True)

# --- C. the loophole: closure of ALL pool cells (submitting FPs to reach true ancestors) ---
covC, nC, cloC, coveredC = closure_weight(np.ones(p_rows.size, dtype=bool))
RC = covC / TOTAL_GT_W
prC = covC / max(1e-9, cloC)
out["C_full_pool_closure"] = {
    "definition": "submit EVERY pool cell at 1.0 -- max reachability, pr < 1. Upper bound on R.",
    "recall_R": round(RC, 4), "precision_if_all_submitted": round(prC, 4),
    "f_if_all_submitted": round(2 * prC * RC / (prC + RC), 4),
    "extra_gt_weight_reachable_only_via_FP_cells": round(covC - covA, 2),
    "loophole_is_real": bool(covC - covA > 1e-6),
}
print(f"[C] full-pool closure R={RC:.4f} (vs strict {RA:.4f}); extra reachable only via FPs "
      f"= {covC-covA:.2f} weight", flush=True)

# --- D. unreachable accounting -------------------------------------------------------------
unreach = (G != 0) & toi_mask[None, :] & (~coveredA)
per_prot_gt = ((G != 0) & toi_mask[None, :]).sum(axis=1)
per_prot_un = unreach.sum(axis=1)
out["D_unreachable"] = {
    "gt_weight_unreachable_strict": round(TOTAL_GT_W - covA, 2),
    "gt_cells_unreachable_strict": int(unreach.sum()),
    "proteins_fully_unreachable": int(((per_prot_gt > 0) & (per_prot_un == per_prot_gt)).sum()),
    "proteins_with_any_unreachable": int((per_prot_un > 0).sum()),
    "proteins_with_gt_in_toi_ia": int((per_prot_gt > 0).sum()),
}
print(f"[D] unreachable weight={TOTAL_GT_W-covA:.1f} cells={int(unreach.sum())} "
      f"fully_unreachable_proteins={out['D_unreachable']['proteins_fully_unreachable']}", flush=True)

# --- GATE: run the strict oracle through the REAL cafa_eval --------------------------------
DRIVER = '''
import json, signal
from cafaeval.evaluation import cafa_eval
signal.signal(signal.SIGTERM, signal.SIG_DFL)
df,dfs=cafa_eval("{obo}","{pd}","{gt}",ia="{ia}",prop="fill",norm="cafa",
    no_orphans=True,max_terms=500,th_step=0.001,n_cpu=1,weighted_only=False)
out={{}}
for k,v in dfs.items(): out[k]=v.reset_index().to_dict(orient="records")
json.dump(out,open("{o}","w"),default=str)
'''
inv = {v: k for k, v in gt.ids.items()}
tl = ont.terms_list
sel = gt_cell
with tempfile.TemporaryDirectory() as td:
    d = Path(td) / "pd"; d.mkdir()
    with (d / "oracle.tsv").open("w") as fh:
        for r, c in zip(p_rows[sel], p_cols[sel]):
            fh.write(f"{inv[int(r)]}\t{tl[int(c)]['id']}\t1.000000\n")
    gtf = Path(td) / "gt.tsv"
    with gtf.open("w"), open(DS / "gt_pk_bp.tsv") as src:
        pass
    import shutil; shutil.copy(DS / "gt_pk_bp.tsv", gtf)
    rj = Path(td) / "r.json"; drv = Path(td) / "d.py"
    drv.write_text(DRIVER.format(obo=OBO, pd=str(d), gt=str(gtf), ia=IA, o=str(rj)))
    r = subprocess.run([PY, str(drv)], capture_output=True, text=True, timeout=7200)
    if r.returncode != 0:
        out["GATE_real_cafa_eval"] = {"FAILED": r.stderr[-500:]}
        print("GATE FAILED", r.stderr[-500:], flush=True)
    else:
        best = None
        for rec in json.loads(rj.read_text()).get("f_micro_w", []):
            if (rec.get("ns") or "") == NS and rec.get("f_micro_w") is not None:
                best = rec
        out["GATE_real_cafa_eval"] = {k: (round(float(best[k]), 4) if isinstance(best.get(k), (int, float)) else best.get(k))
                                      for k in ("f_micro_w", "pr_micro_w", "rc_micro_w", "tau", "cov_max")}
        out["GATE_verdict"] = ("DERIVATION HOLDS" if abs(float(best["pr_micro_w"]) - 1.0) < 1e-4
                               else "DERIVATION WRONG: precision != 1")
        print(f"[GATE] real cafa_eval on strict oracle: {out['GATE_real_cafa_eval']}", flush=True)
        print(f"[GATE] {out['GATE_verdict']}", flush=True)

json.dump(out, open(W / "instrument_oracle.json", "w"), indent=1, default=str)
print(f"DONE {time.time()-t0:.0f}s", flush=True)
