"""Expand the eval pool with co-occurrence candidates and measure the real
f_micro_w cost/benefit on PK-BPO.

Baseline  = the reranked pool as-is (what PROTEA does today).
Expanded  = pool + top-N co-occurrence candidates from each PK protein's t0-known
            terms, scored by co-occurrence strength (rank-percentile), placed on a
            sub-range [0, alpha] below the reranker's ranking so the cafaeval
            threshold sweep can trade them off.

Both arms are scored with the SAME cafaeval harness (prop=fill, norm=cafa,
no_orphans, max_terms=500) on the PK-BPO cell, so the delta is attributable to the
candidate expansion alone.
"""
import collections
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.sparse import coo_matrix

W = Path("/home/frapercan/Thesis2/storage/cooc_experiment")
DS = Path("/home/frapercan/Thesis2/repositories/protea-reranker-lab/datasets/protst-global-train227-test230")
BASE = Path("/home/frapercan/Thesis2/CAFA_forever/data/releases/Sep_2025_Mar_2026")
OBO = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
IA = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv"
PROTEA_PY = "/home/frapercan/Thesis2/repositories/PROTEA/.venv/bin/python"
TOPN = 100
ALPHAS = [0.3, 0.6]

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


def cafaeval(pred_dir, gt, td):
    raw = Path(td) / "raw.json"
    drv = Path(td) / "drv.py"
    drv.write_text(DRIVER.format(obo=OBO, pred_dir=pred_dir, gt=gt, ia=IA, out_json=str(raw)))
    p = subprocess.run([PROTEA_PY, str(drv)], capture_output=True, text=True, timeout=3600)
    if p.returncode != 0:
        return {"error": p.stderr[-300:]}
    data = json.loads(raw.read_text())
    for rec in data.get("f_micro_w", []):
        if (rec.get("ns") or "") == "biological_process" and rec.get("f_micro_w") is not None:
            return {"f_micro_w": round(float(rec["f_micro_w"]), 4)}
    return {"f_micro_w": None}


def write_arm(name, prots, gos, scores, truth_pairs, td):
    d = Path(td) / name
    pd_ = d / "pred_dir"
    pd_.mkdir(parents=True, exist_ok=True)
    with (pd_ / f"{name}.tsv").open("w") as fh:
        for p, g, s in zip(prots, gos, scores):
            fh.write(f"{p}\t{g}\t{s:.6f}\n")
    gt = d / "gt.tsv"
    with gt.open("w") as fh:
        for p, g in truth_pairs:
            fh.write(f"{p}\t{g}\n")
    return str(pd_), str(gt)


def rankpct(x):
    if len(x) == 0:
        return x
    o = np.argsort(x, kind="stable")
    r = np.empty(len(x))
    r[o] = np.arange(len(x))
    return r / max(1, len(x) - 1)


def main():
    # ---- co-occurrence from t0-known terms ----
    known = collections.defaultdict(list)
    known_bp = collections.defaultdict(list)
    tidx, bpidx = {}, {}
    for line in open(BASE / "groundtruth_PK_known.tsv"):
        if line.startswith("EntryID"):
            continue
        p, t, a = line.rstrip("\n").split("\t")
        known[p].append(tidx.setdefault(t, len(tidx)))
        if a == "P":
            known_bp[p].append(bpidx.setdefault(t, len(bpidx)))
    nT, nB = len(tidx), len(bpidx)
    bp_terms = [None] * nB
    for t, i in bpidx.items():
        bp_terms[i] = t
    rows, cols = [], []
    for p in known:
        kb = known_bp[p]
        if not kb:
            continue
        for A in known[p]:
            rows.extend([A] * len(kb))
            cols.extend(kb)
    cooc = coo_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(nT, nB)).tocsr()
    print(f"cooc built: {cooc.nnz:,} nnz", flush=True)

    # ---- reranked eval pool (PK-BPO) ----
    t = pq.read_table(W / "rerank_out" / "eval_scores.parquet")
    cat = np.asarray(t.column("category").to_pylist())
    asp = np.asarray(t.column("aspect").to_pylist())
    m = (cat == "pk") & (asp == "bpo")
    prots = np.asarray(t.column("protein_accession").to_pylist())[m]
    gos = np.asarray(t.column("go_term_id").to_pylist())[m]
    lab = t.column("label").to_numpy(zero_copy_only=False).astype(np.int64)[m]
    rr = t.column("reranker_score").to_numpy(zero_copy_only=False).astype(np.float64)[m]
    rr_n = rankpct(rr)  # normalise the reranker ranking to [0,1]
    pool = collections.defaultdict(set)
    for p, g in zip(prots, gos):
        pool[p].add(g)
    # FULL ground truth (all true PK-BP pairs, not just the ones we already reach),
    # so the expansion's recall gain is credited and its precision cost penalised.
    truth = []
    for line in open(DS / "gt_pk_bp.tsv"):
        p, g = line.rstrip("\n").split("\t")
        truth.append((p, g))
    in_pool = sum(1 for p, g in truth if g in pool.get(p, ()))
    print(f"PK-BPO pool: {m.sum():,} rows | FULL gt: {len(truth):,} pairs "
          f"({in_pool:,} reachable = recall {in_pool/len(truth):.3f})", flush=True)

    results = {}
    with tempfile.TemporaryDirectory() as td:
        pd_, gt = write_arm("baseline", prots, gos, rr_n, truth, td)
        results["baseline"] = cafaeval(pd_, gt, td)
        print(f"  BASELINE (pool only)      f_micro_w={results['baseline'].get('f_micro_w')}", flush=True)

        # ---- expansion ----
        add_p, add_g, add_s = [], [], []
        for p in pool:
            ki = known.get(p)
            if not ki:
                continue
            sc = np.asarray(cooc[ki].sum(axis=0)).ravel()
            for b in known_bp.get(p, ()):
                sc[b] = -1
            top = np.argpartition(-sc, min(TOPN, nB - 1))[:TOPN]
            base = pool[p]
            for i in top:
                if sc[i] <= 0:
                    continue
                g = bp_terms[i]
                if g in base:
                    continue
                add_p.append(p)
                add_g.append(g)
                add_s.append(sc[i])
        add_s = rankpct(np.asarray(add_s, dtype=float))
        print(f"  expansion top-{TOPN}: +{len(add_p):,} candidates", flush=True)

        for alpha in ALPHAS:
            P = np.concatenate([prots, np.asarray(add_p)])
            G = np.concatenate([gos, np.asarray(add_g)])
            S = np.concatenate([rr_n, add_s * alpha])
            pd_, gt = write_arm(f"exp_a{alpha}", P, G, S, truth, td)
            r = cafaeval(pd_, gt, td)
            results[f"expanded_alpha{alpha}"] = r
            base_f = results["baseline"].get("f_micro_w")
            d = (r.get("f_micro_w") - base_f) if (r.get("f_micro_w") and base_f) else None
            print(f"  EXPANDED alpha={alpha}      f_micro_w={r.get('f_micro_w')}  delta={('%+.4f' % d) if d is not None else 'NA'}", flush=True)

    (W / "cooc_fusion_result.json").write_text(json.dumps(results, indent=1))
    print("\nBoth arms scored against the FULL PK-BP ground truth, so the expansion's")
    print("recall gain is credited and its precision cost is penalised. A positive delta")
    print("means the co-occurrence candidates pay for themselves; negative means the")
    print("noise they add costs more than the terms they recover.")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
