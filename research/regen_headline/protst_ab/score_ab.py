"""Parametrized copy of results/clean_227230/score_rerank_227230.py. Scores the
RERANKED variant (per-category LightGBM score for ALL categories) 9-cell +
pooled-per-aspect f_micro_w with the SAME cafaeval harness (prop=fill, norm=cafa,
no_orphans, max_terms=500, th_step=0.001, IA/OBO v227).

Usage: python score_ab.py <eval_scores.parquet> <out.json>
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

SCORES = sys.argv[1]
OUTJSON = sys.argv[2]
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


def slice_tsvs(prots, gos, score, lab, mask, cellname, td):
    p = prots[mask]
    g = gos[mask]
    s = score[mask]
    pos = lab[mask] > 0
    d = Path(td) / cellname
    pd_ = d / "pred_dir"
    pd_.mkdir(parents=True, exist_ok=True)
    with (pd_ / f"{cellname}.tsv").open("w") as fh:
        for pp, gg, ss in zip(p, g, s):
            fh.write(f"{pp}\t{gg}\t{ss:.6f}\n")
    gt = d / "gt.tsv"
    with gt.open("w") as fh:
        for pp, gg in zip(p[pos], g[pos]):
            fh.write(f"{pp}\t{gg}\n")
    return str(pd_), str(gt), int(mask.sum()), int(pos.sum()), int(len(set(p.tolist())))


def run_variant(name, prots, gos, score, lab, cat, asp):
    out = {"per_cell": {}, "pooled_per_aspect": {}}
    with tempfile.TemporaryDirectory() as td:
        for a in ASPECTS:
            m = (asp == a)
            if m.sum() == 0:
                continue
            pd_, gt, n, npos, nprot = slice_tsvs(prots, gos, score, lab, m, f"{name}_POOL_{a}", td)
            r = cafaeval(pd_, gt, a, str(Path(td) / f"{name}_POOL_{a}"))
            r.update(n=n, pos=npos, prot=nprot)
            out["pooled_per_aspect"][a] = r
            print(f"  [{name}] POOLED {a.upper()} n={n:<7} pos={npos:<5} f_micro_w={r.get('f_micro_w')}", flush=True)
        for c in CATS:
            for a in ASPECTS:
                m = (cat == c) & (asp == a)
                if m.sum() == 0:
                    continue
                cell = f"{c}-{a}"
                pd_, gt, n, npos, nprot = slice_tsvs(prots, gos, score, lab, m, f"{name}_{cell}", td)
                if npos == 0:
                    out["per_cell"][cell] = {"f_micro_w": None, "n": n, "pos": 0}
                    continue
                r = cafaeval(pd_, gt, a, str(Path(td) / f"{name}_{cell}"))
                r.update(n=n, pos=npos, prot=nprot)
                out["per_cell"][cell] = r
                print(f"  [{name}] {cell} n={n:<7} pos={npos:<5} f_micro_w={r.get('f_micro_w')}", flush=True)
    pp = [v["f_micro_w"] for v in out["pooled_per_aspect"].values() if isinstance(v.get("f_micro_w"), float)]
    pc = [v["f_micro_w"] for v in out["per_cell"].values() if isinstance(v.get("f_micro_w"), float)]
    out["mean_pooled_per_aspect"] = round(float(np.mean(pp)), 4) if pp else None
    out["mean_per_cell"] = round(float(np.mean(pc)), 4) if pc else None
    return out


def main():
    t = pq.read_table(SCORES)
    prots = np.asarray(t.column("protein_accession").to_pylist())
    gos = np.asarray(t.column("go_term_id").to_pylist())
    lab = t.column("label").to_numpy(zero_copy_only=False).astype(np.int64)
    cat = np.asarray(t.column("category").to_pylist())
    asp = np.asarray(t.column("aspect").to_pylist())
    rr = t.column("reranker_score").to_numpy(zero_copy_only=False).astype(np.float64)

    results = {}
    print("=== RERANKED (all categories) ===", flush=True)
    results["reranked"] = run_variant("reranked", prots, gos, rr, lab, cat, asp)
    Path(OUTJSON).write_text(json.dumps(results, indent=1))
    print("\nDONE ->", OUTJSON, flush=True)
    print("reranked mean_pooled", results["reranked"]["mean_pooled_per_aspect"],
          "mean_cell", results["reranked"]["mean_per_cell"])


if __name__ == "__main__":
    main()
