"""Control for INSTRUMENT 1. My gates demanded exact set equality and reported 26 (prop=max)
and 9 (prop=fill) dissenting cells. "Quantization" is a story until measured, so measure it.

CLAIM UNDER TEST: the residual disagreement is the harness's own "%.6f" write precision, not
a mechanism. Under minmax, 1e-6 of resolution == (hi-lo)*1e-6 == 1.23e-5 RAW units, so any
pool cell whose raw score sits within ~1.3e-5 of the cut 0.393 can land on the other side.

GATE: every dissenting cell must have a raw score within 2e-5 of 0.393. If even ONE dissenter
sits far from the cut, this is not rounding and I must say the mechanism is incomplete.
"""
import json, sys, tempfile, os, time
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

onts = obo_parser(OBO, ("is_a", "part_of"), IA, False)
ont = onts[NS]
gts = gt_parser(str(DS / "gt_pk_bp.tsv"), onts)
gt = gts[NS]
t = pq.read_table(W / "rerank_out" / "eval_scores.parquet")
cat = np.asarray(t.column("category").to_pylist()); asp = np.asarray(t.column("aspect").to_pylist())
m = (cat == "pk") & (asp == "bpo")
prot = np.asarray(t.column("protein_accession").to_pylist())[m]
go = np.asarray(t.column("go_term_id").to_pylist())[m]
raw = t.column("reranker_score").to_numpy(zero_copy_only=False).astype(np.float64)[m]
lo, hi = float(raw.min()), float(raw.max())
MM_TAU = (RAW_TAU - lo) / (hi - lo)
G = gt.matrix; toi_ia = ont.toi_ia; n_toi = len(toi_ia)


def build(scores, prop, fmt):
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
# pool cell -> own raw score, in the toi_ia flat index space
ti = ont.terms_dict
rows = np.array([gt.ids.get(p, -1) for p in prot], dtype=np.int64)
cols = np.array([ti[g]["index"] if g in ti else -1 for g in go], dtype=np.int64)
tmask = np.zeros(ont.idxs, dtype=bool); tmask[toi_ia] = True
own = {}
for r, c, v in zip(rows, cols, raw):
    if r >= 0 and c >= 0 and tmask[c]:
        k = int(r) * n_toi + int(np.searchsorted(toi_ia, c))
        own[k] = max(own.get(k, -1e9), float(v))

out = {"gate": "every dissenting cell must have raw within 2e-5 of 0.393, else NOT rounding",
       "minmax_resolution_in_raw_units": round((hi - lo) * 1e-6, 8)}

for prop in ("max", "fill"):
    a = set(np.flatnonzero((build(raw, prop, ".6f")[:, toi_ia] >= RAW_TAU).ravel()).tolist())
    b = set(np.flatnonzero((build(mm, prop, ".6f")[:, toi_ia] >= MM_TAU).ravel()).tolist())
    diss = (a ^ b) if prop == "max" else ((b - a) | {x for x in (a - b) if own.get(x, -9e9) > 0})
    # for fill, the EXPECTED extras are in-pool cells with own raw <= 0; anything else dissents
    d = []
    for x in sorted(diss):
        v = own.get(x)
        d.append({"flat": int(x), "own_raw": (round(v, 8) if v is not None else None),
                  "dist_from_cut": (round(abs(v - RAW_TAU), 8) if v is not None else None)})
    far = [z for z in d if z["own_raw"] is None or abs(z["own_raw"] - RAW_TAU) > 2e-5]
    out[prop] = {"n_dissenting": len(d), "n_far_from_cut": len(far),
                 "sample": d[:12], "far_sample": far[:12],
                 "verdict": "ROUNDING" if not far else "NOT ROUNDING -- mechanism incomplete"}
    print(f"[{prop}] dissent={len(d)} far_from_cut={len(far)} -> {out[prop]['verdict']}", flush=True)

# Second control: re-run prop=max with 12 decimals. If it is rounding, dissent must shrink.
a = set(np.flatnonzero((build(raw, "max", ".12f")[:, toi_ia] >= RAW_TAU).ravel()).tolist())
b = set(np.flatnonzero((build(mm, "max", ".12f")[:, toi_ia] >= MM_TAU).ravel()).tolist())
out["max_at_12_decimals"] = {"sym_diff": len(a ^ b),
                             "verdict": "ROUNDING CONFIRMED" if len(a ^ b) < 26 else "NOT rounding"}
print(f"[max @ .12f] symdiff={len(a ^ b)} (was 26 at .6f) -> {out['max_at_12_decimals']['verdict']}", flush=True)
json.dump(out, open(W / "instrument_quantnoise.json", "w"), indent=1, default=str)
print("DONE", flush=True)
