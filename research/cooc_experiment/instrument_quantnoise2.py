"""Control for the control. instrument_quantnoise.py fired its own gate ("NOT ROUNDING"), but
the gate was MIS-SPECIFIED: it compared each dissenting cell's OWN pool score to the cut, while
a dissenting cell is a PROPAGATED cell whose value came from some descendant. The quantity that
decides which side of the cut a propagated cell lands on is its PROPAGATED VALUE, not its own.

CORRECTED CLAIM: dissent is float64 non-associativity. `x >= 0.393` and
`(x-lo)/(hi-lo) >= (0.393-lo)/(hi-lo)` are NOT bit-identical, so a cell whose propagated value
sits within a few ulp of the cut can land on either side. And because prop=max copies ONE
descendant's value into ALL of its ancestors, a single borderline pool candidate produces a
whole ancestor chain of dissenting cells -- which is why 26 cells is not 26 independent events.

GATES (both must hold, else this is a real mechanism and I say so):
  G1: every dissenting cell's PROPAGATED raw value is within 1e-6 of the cut 0.393.
  G2: the dissenting cells collapse onto a HANDFUL of distinct propagated values (i.e. they are
      ancestor chains of a couple of borderline candidates), not 26 independent scores.
"""
import json, sys, tempfile, os
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
toi_ia = ont.toi_ia; n_toi = len(toi_ia)


def build(scores, prop, fmt=".6f"):
    mat = {NS: np.zeros(G.shape, dtype="float")}
    rn = {NS: np.zeros(G.shape[0], dtype=np.int32)}; ids = {NS: {}}
    nsd = {tt: NS for tt in ont.terms_dict}; nsd.update({tt: NS for tt in ont.terms_dict_alt})
    tix = {NS: {tt: i["index"] for tt, i in ont.terms_dict.items()}}
    with tempfile.TemporaryDirectory() as td:
        f = os.path.join(td, "p.tsv")
        with open(f, "w") as fh:
            for p, g, v in zip(prot, go, scores):
                fh.write(f"{p}\t{g}\t{format(v, fmt)}\n")
        _pred_parser_legacy(f, onts, gts, nsd, tix, ids, mat, rn, {}, 500)
    propagate(mat[NS], ont, ont.order, mode=prop, parallel=1)
    return mat[NS]


mm = (raw - lo) / (hi - lo)
out = {"gates": "G1 propagated value within 1e-6 of cut; G2 dissenters collapse to few values"}
Mr = build(raw, "max"); Mm = build(mm, "max")
Ar = (Mr[:, toi_ia] >= RAW_TAU); Am = (Mm[:, toi_ia] >= MM_TAU)
diss = np.flatnonzero((Ar ^ Am).ravel())
vr = Mr[:, toi_ia].ravel()[diss]          # propagated value on the RAW scale
vm = Mm[:, toi_ia].ravel()[diss]          # propagated value on the MINMAX scale
vm_back = vm * (hi - lo) + lo             # minmax value mapped back to raw units
d_raw = np.abs(vr - RAW_TAU)
out["prop_max"] = {
    "n_dissenting_cells": int(diss.size),
    "max_dist_of_propagated_value_from_cut_raw_units": float(np.max(d_raw)),
    "n_distinct_propagated_values": int(np.unique(np.round(vr, 9)).size),
    "distinct_propagated_values": [round(float(x), 8) for x in np.unique(np.round(vr, 9))],
    "sample": [{"prop_raw": round(float(a), 8), "minmax_back_to_raw": round(float(b), 8),
                "dist_from_cut": round(float(c), 10)}
               for a, b, c in list(zip(vr, vm_back, d_raw))[:10]],
}
G1 = bool(np.max(d_raw) < 1e-6)
G2 = bool(np.unique(np.round(vr, 9)).size <= 6)
out["prop_max"]["G1_all_within_1e-6_of_cut"] = G1
out["prop_max"]["G2_collapse_to_few_values"] = G2
out["prop_max"]["verdict"] = ("FLOAT NOISE CONFIRMED" if (G1 and G2)
                              else "NOT float noise -- a real mechanism remains")
print(f"[prop=max] dissent={diss.size} max_dist_from_cut={np.max(d_raw):.3e} "
      f"distinct_values={out['prop_max']['n_distinct_propagated_values']} "
      f"-> {out['prop_max']['verdict']}", flush=True)
for s in out["prop_max"]["sample"][:6]:
    print(f"   propagated_raw={s['prop_raw']:.8f}  minmax->raw={s['minmax_back_to_raw']:.8f}  "
          f"dist={s['dist_from_cut']:.3e}", flush=True)

# Decisive control: does the dissent scale with the CUT rather than with the scale? If it is
# float noise at the cut, moving the cut to a value no score sits near must kill it.
for tau_probe in (0.35, 0.45, 0.60):
    mt = (tau_probe - lo) / (hi - lo)
    a = (Mr[:, toi_ia] >= tau_probe); b = (Mm[:, toi_ia] >= mt)
    n = int((a ^ b).sum())
    out.setdefault("dissent_vs_cut", {})[str(tau_probe)] = n
    print(f"[probe tau={tau_probe}] dissent={n} cells (out of {int(a.sum())} active)", flush=True)

f_r = out.setdefault("f_impact", {})
out["f_impact"]["note"] = ("26 cells out of 130125 active = 0.02%; f agreed to 4 dp (0.2230 vs "
                           "0.2230). Whatever it is, it cannot carry 0.088.")
json.dump(out, open(W / "instrument_quantnoise2.json", "w"), indent=1, default=str)
print("DONE", flush=True)
