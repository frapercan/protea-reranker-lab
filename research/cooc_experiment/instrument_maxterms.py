"""INSTRUMENT 1c: the residual 26 cells are NOT float noise. Identify them or admit failure.

instrument_quantnoise2.py killed the rounding story: dissenting cells have propagated_raw
~0.5248 while the minmax arm's same cell maps back to ~-0.704. Those are different VALUES, not
a boundary flip. Since prop="max" is monotone-equivariant, the only way the two arms compute a
different max is if the CONTRIBUTING CELL SETS differ.

NEW CLAIM (H5): the residual is max_terms=500 x the same zero-clamp, a THIRD interaction.
  parser.py:227-234, legacy path:
      if max_terms is not None and old == 0.0 and row_nnz[i] > max_terms: continue
      if prob_f > old:  ...; if old == 0.0: row_nnz[i] += 1
  row_nnz counts cells actually WRITTEN. Under raw, non-positive scores are never written, so
  row_nnz grows over only ~19% of the row and the cap never binds. Under minmax every cell is
  positive, so row_nnz reaches the 500 cap on any protein with >500 candidates and the row is
  TRUNCATED -- in FILE ORDER, not score order. That is why hypothesis 3 was both right and
  wrong: the cap binds for only ~5 proteins, so it cannot move 0.088, but it is exactly what
  produces the residual dissent.

GATE: with max_terms=None the prop="max" symmetric difference must be EXACTLY 0. If it is not,
H5 is dead and I report that I could not identify the residual.
"""
import json, sys, tempfile, os
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path
from collections import Counter

sys.path.insert(0, "/home/frapercan/Thesis2/repositories/PROTEA/.venv/lib/python3.12/site-packages")
from cafaeval.parser import obo_parser, gt_parser, _pred_parser_legacy
from cafaeval.graph import propagate

W = Path("/home/frapercan/Thesis2/storage/cooc_experiment")
DS = Path("/home/frapercan/Thesis2/repositories/protea-reranker-lab/datasets/protst-global-train227-test230")
OBO = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
IA = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv"
NS = "biological_process"; RAW_TAU = 0.393

onts = obo_parser(OBO, ("is_a", "part_of"), IA, False); ont = onts[NS]
gts = gt_parser(str(DS / "gt_pk_bp.tsv"), onts); gt = gts[NS]; G = gt.matrix
t = pq.read_table(W / "rerank_out" / "eval_scores.parquet")
cat = np.asarray(t.column("category").to_pylist()); asp = np.asarray(t.column("aspect").to_pylist())
m = (cat == "pk") & (asp == "bpo")
prot = np.asarray(t.column("protein_accession").to_pylist())[m]
go = np.asarray(t.column("go_term_id").to_pylist())[m]
raw = t.column("reranker_score").to_numpy(zero_copy_only=False).astype(np.float64)[m]
lo, hi = float(raw.min()), float(raw.max()); MM_TAU = (RAW_TAU - lo) / (hi - lo)
mm = (raw - lo) / (hi - lo)
toi_ia = ont.toi_ia; n_toi = len(toi_ia)
out = {"gate": "with max_terms=None the prop=max symmetric difference must be EXACTLY 0"}

# how many candidates does each protein have, and how many are positive?
cnt = Counter(prot); pos = Counter(prot[raw > 0])
over = {p: c for p, c in cnt.items() if c > 500}
out["candidates_per_protein"] = {
    "n_proteins": len(cnt), "max_candidates": max(cnt.values()),
    "n_proteins_over_500_candidates": len(over),
    "proteins_over_500": {p: {"candidates": c, "positive_candidates": pos.get(p, 0)}
                          for p, c in sorted(over.items(), key=lambda kv: -kv[1])},
    "max_positive_candidates_any_protein": max(pos.values()) if pos else 0,
}
print(f"[ctx] proteins={len(cnt)} max_cands={max(cnt.values())} over500={len(over)} "
      f"max_POSITIVE_cands={max(pos.values())}", flush=True)
print(f"[ctx] -> under raw the cap counts only positives (max {max(pos.values())} < 500) so it "
      f"NEVER binds; under minmax it counts all (max {max(cnt.values())} > 500) so it DOES.",
      flush=True)


def build(scores, prop, max_terms):
    mat = {NS: np.zeros(G.shape, dtype="float")}
    rn = {NS: np.zeros(G.shape[0], dtype=np.int32)}; ids = {NS: {}}
    nsd = {tt: NS for tt in ont.terms_dict}; nsd.update({tt: NS for tt in ont.terms_dict_alt})
    tix = {NS: {tt: i["index"] for tt, i in ont.terms_dict.items()}}
    with tempfile.TemporaryDirectory() as td:
        f = os.path.join(td, "p.tsv")
        with open(f, "w") as fh:
            for p, g, v in zip(prot, go, scores):
                fh.write(f"{p}\t{g}\t{v:.6f}\n")
        _pred_parser_legacy(f, onts, gts, nsd, tix, ids, mat, rn, {}, max_terms)
    propagate(mat[NS], ont, ont.order, mode=prop, parallel=1)
    return mat[NS], rn[NS].copy()


for mt in (500, None):
    Mr, rnr = build(raw, "max", mt)
    Mm, rnm = build(mm, "max", mt)
    a = set(np.flatnonzero((Mr[:, toi_ia] >= RAW_TAU).ravel()).tolist())
    b = set(np.flatnonzero((Mm[:, toi_ia] >= MM_TAU).ravel()).tolist())
    sd = len(a ^ b)
    key = f"max_terms={mt}"
    out[key] = {"sym_diff_prop_max": sd,
                "rows_hitting_cap_raw": int((rnr > 500).sum()),
                "rows_hitting_cap_minmax": int((rnm > 500).sum()),
                "max_row_nnz_raw": int(rnr.max()), "max_row_nnz_minmax": int(rnm.max())}
    print(f"[{key:16s}] prop=max symdiff={sd}  row_nnz_max raw={rnr.max()} mm={rnm.max()}",
          flush=True)
    del Mr, Mm

sd_none = out["max_terms=None"]["sym_diff_prop_max"]
out["H5_verdict"] = ("H5 CONFIRMED: the residual is max_terms x the clamp; with the cap removed "
                     "prop=max is EXACTLY scale-invariant"
                     if sd_none == 0 else
                     f"H5 DEAD: {sd_none} cells still dissent with no cap -- residual UNIDENTIFIED")
print(f"\n[GATE] max_terms=None symdiff={sd_none} -> {out['H5_verdict']}", flush=True)
json.dump(out, open(W / "instrument_maxterms.json", "w"), indent=1, default=str)
print("DONE", flush=True)
