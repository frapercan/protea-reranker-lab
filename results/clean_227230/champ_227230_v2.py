"""CHAMPION (raw-KNN, 1-distance) f_micro_w on the TEST frame 227->230, read
directly from the platform-exported frozen eval.parquet (deployment-realistic).

Reuses the canonical scoring of farm_exp_15_knn_226_227.py EXACTLY:
  champion score = 1 - cosine distance; GT = eval rows with label > 0;
  cafaeval(prop=fill, norm=cafa, no_orphans, max_terms=500, th_step=0.001) in the
  PROTEA venv; IA = v227 LAFA-aligned; OBO = v227 (Sep 2025) LAFA term universe.

Reports BOTH:
  - per-cell (NK/LK/PK x MFO/BPO/CCO), category-specific threshold (diagnostic)
  - pooled-per-aspect (all categories, ONE threshold per namespace) = the
    LAFA-realistic deployment number = the headline.
"""
import json, subprocess, sys
from pathlib import Path
import pyarrow.parquet as pq
import numpy as np

S = "/tmp/claude-1000/-home-frapercan-Thesis2/afd2c43a-ede7-46dc-94dd-9745808d2194/scratchpad"
EVAL = f"{S}/ds227230/eval.parquet"
OBO = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
IA = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv"
PROTEA_PY = "/home/frapercan/Thesis2/repositories/PROTEA/.venv/bin/python"
ASPECTS = ["mfo", "bpo", "cco"]
CATS = ["nk", "lk", "pk"]
NS = {"bpo": "biological_process", "mfo": "molecular_function", "cco": "cellular_component"}

DRIVER = '''
import json, signal
from cafaeval.evaluation import cafa_eval
signal.signal(signal.SIGTERM, signal.SIG_DFL)
df, dfs_best = cafa_eval("{obo}", "{pred_dir}", "{gt}", ia="{ia}",
    prop="fill", norm="cafa", no_orphans=True, max_terms=500, th_step=0.001,
    n_cpu=1, weighted_only=False)
out = {{}}
for kind, dfb in dfs_best.items():
    out[kind] = dfb.reset_index().to_dict(orient="records")
json.dump(out, open("{out_json}", "w"), default=str)
'''


def cafaeval(pred_dir, gt, asp, td):
    raw = Path(td) / "raw.json"
    drv = Path(td) / "drv.py"
    drv.write_text(DRIVER.format(obo=OBO, pred_dir=pred_dir, gt=gt, ia=IA, out_json=str(raw)))
    p = subprocess.run([PROTEA_PY, str(drv)], capture_output=True, text=True, timeout=1800)
    if p.returncode != 0:
        return {"error": p.stderr[-400:]}
    raw = json.loads(raw.read_text())
    tns = NS[asp]

    def pick(kind, col):
        for rec in raw.get(kind, []):
            if (rec.get("ns") or rec.get("namespace") or "") == tns and rec.get(col) is not None:
                try:
                    return round(float(rec[col]), 4)
                except (TypeError, ValueError):
                    return None
        return None
    return {"f_micro_w": pick("f_micro_w", "f_micro_w"), "f_micro": pick("f_micro", "f_micro"),
            "fmax": pick("f", "f")}


def slice_tsvs(t, mask, cellname, td):
    prots = t.column("protein_accession").to_numpy(zero_copy_only=False)[mask]
    gos = t.column("go_term_id").to_numpy(zero_copy_only=False)[mask]
    dist = t.column("distance").to_numpy(zero_copy_only=False).astype(np.float64)[mask]
    lab = t.column("label").to_numpy(zero_copy_only=False).astype(np.int64)[mask]
    score = 1.0 - dist
    pos = lab > 0
    d = Path(td) / cellname
    pd_ = d / "pred_dir"
    pd_.mkdir(parents=True, exist_ok=True)
    with (pd_ / f"{cellname}.tsv").open("w") as fh:
        for p, g, s in zip(prots, gos, score):
            fh.write(f"{p}\t{g}\t{s:.6f}\n")
    gt = d / "gt.tsv"
    with gt.open("w") as fh:
        for p, g in zip(prots[pos], gos[pos]):
            fh.write(f"{p}\t{g}\n")
    return str(pd_), str(gt), int(mask.sum()), int(pos.sum()), int(len(set(prots.tolist())))


def main():
    import tempfile
    t = pq.read_table(EVAL, columns=["protein_accession", "go_term_id", "distance", "label", "category", "aspect"])
    cat = t.column("category").to_numpy(zero_copy_only=False)
    asp = t.column("aspect").to_numpy(zero_copy_only=False)
    out = {"per_cell": {}, "pooled_per_aspect": {}}
    with tempfile.TemporaryDirectory() as td:
        # pooled-per-aspect (the LAFA headline): all categories, one threshold/ns
        for a in ASPECTS:
            m = (asp == a)
            if m.sum() == 0:
                continue
            pd_, gt, n, npos, nprot = slice_tsvs(t, m, f"POOL_{a}", td)
            r = cafaeval(pd_, gt, a, str(Path(td) / f"POOL_{a}"))
            r.update(n=n, pos=npos, prot=nprot)
            out["pooled_per_aspect"][a] = r
            print(f"  POOLED {a.upper()}  n={n:<7} pos={npos:<5} prot={nprot:<5} f_micro_w={r.get('f_micro_w')}", flush=True)
        # per-cell (diagnostic)
        for c in CATS:
            for a in ASPECTS:
                m = (cat == c) & (asp == a)
                if m.sum() == 0:
                    continue
                cell = f"{c}-{a}"
                pd_, gt, n, npos, nprot = slice_tsvs(t, m, cell, td)
                if npos == 0:
                    out["per_cell"][cell] = {"f_micro_w": None, "n": n, "pos": 0}
                    print(f"  {cell}  n={n:<7} pos=0 (skip)", flush=True)
                    continue
                r = cafaeval(pd_, gt, a, str(Path(td) / cell))
                r.update(n=n, pos=npos, prot=nprot)
                out["per_cell"][cell] = r
                print(f"  {cell}  n={n:<7} pos={npos:<5} prot={nprot:<5} f_micro_w={r.get('f_micro_w')}", flush=True)
    pp = [v["f_micro_w"] for v in out["pooled_per_aspect"].values() if isinstance(v.get("f_micro_w"), float)]
    pc = [v["f_micro_w"] for v in out["per_cell"].values() if isinstance(v.get("f_micro_w"), float)]
    out["mean_pooled_per_aspect"] = round(float(np.mean(pp)), 4) if pp else None
    out["mean_per_cell"] = round(float(np.mean(pc)), 4) if pc else None
    Path(f"{S}/champion_227230_v2.json").write_text(json.dumps(out, indent=1))
    print("\n=== CHAMPION 227->230 ===")
    print("mean POOLED-per-aspect (LAFA headline):", out["mean_pooled_per_aspect"])
    print("mean per-cell (diagnostic):", out["mean_per_cell"])
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
