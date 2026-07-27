"""INSTRUMENT 3+4: is norm="cafa" the mechanism, and what is the honest scale to submit?

(3) READING normalize() (evaluation.py:567-599): every column except "n" is divided by `ne`,
    a CONSTANT vector (num_annot_prots). norm="cafa" only redirects the `pr` column's
    denominator to metrics["n"]. But f_micro_w is built from pr_micro = tp/(tp+fp) and
    rc_micro = tp/(tp+fn) -- and the constant `ne` CANCELS in both ratios. So f_micro_w
    should be EXACTLY invariant to norm. That is a derivation; this measures it.
    GATE: if f_micro_w changes with norm, my reading of normalize() is wrong.

(4) The scale sweep. prop="fill" + the parser's 0-sentinel means the only thing a monotone
    map actually changes is WHICH ROWS ARE SUBMITTED (score <= 0 == not submitted). That is a
    real, legitimate submitter choice, not a scale trick. So sweep the submission threshold
    explicitly and honestly: prefilter raw > t, then map the survivors onto [0.001, 0.999] so
    the tau grid can actually resolve them. t = -inf is minmax; t = 0 is the deployed anchor.

All arms go through the REAL cafa_eval with the deployed call. Files share one pred_dir so
the obo/gt are parsed once per (prop, norm) combination.
"""
import json, subprocess, sys, tempfile, time
from pathlib import Path
import numpy as np, pyarrow.parquet as pq

W = Path("/home/frapercan/Thesis2/storage/cooc_experiment")
DS = Path("/home/frapercan/Thesis2/repositories/protea-reranker-lab/datasets/protst-global-train227-test230")
OBO = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
IA = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv"
PY = "/home/frapercan/Thesis2/repositories/PROTEA/.venv/bin/python"

DRIVER = '''
import json, signal
from cafaeval.evaluation import cafa_eval
signal.signal(signal.SIGTERM, signal.SIG_DFL)
df,dfs=cafa_eval("{obo}","{pd}","{gt}",ia="{ia}",prop="{prop}",norm="{norm}",
    no_orphans=True,max_terms=500,th_step=0.001,n_cpu=1,weighted_only=False)
out={{}}
for k,v in dfs.items(): out[k]=v.reset_index().to_dict(orient="records")
json.dump(out,open("{o}","w"),default=str)
'''

t = pq.read_table(W / "rerank_out" / "eval_scores.parquet")
cat = np.asarray(t.column("category").to_pylist()); asp = np.asarray(t.column("aspect").to_pylist())
m = (cat == "pk") & (asp == "bpo")
prot = np.asarray(t.column("protein_accession").to_pylist())[m]
go = np.asarray(t.column("go_term_id").to_pylist())[m]
raw = t.column("reranker_score").to_numpy(zero_copy_only=False).astype(np.float64)[m]
lo, hi = float(raw.min()), float(raw.max())


def arms():
    """name -> (mask of submitted rows, scores for those rows)"""
    a = {}
    a["raw_ANCHOR"] = (np.ones(raw.size, bool), raw.copy())
    a["minmax_ALL"] = (np.ones(raw.size, bool), (raw - lo) / (hi - lo))
    # clip at 0 then rescale: identical prefilter to raw, but positives spread over the grid
    cl = np.maximum(raw, 0.0) / hi
    a["clip0_rescale"] = (np.ones(raw.size, bool), cl)
    for th in (-4.0, -3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5):
        s = raw > th
        if s.sum() < 10:
            continue
        v = raw[s]
        vv = 0.001 + 0.998 * (v - v.min()) / max(1e-12, (v.max() - v.min()))
        a[f"prefilter_gt_{th:+.1f}_rescaled"] = (s, vv)
    return a


A = arms()
out = {"pool_rows": int(m.sum()), "lo": round(lo, 4), "hi": round(hi, 4),
       "frac_raw_nonpositive": round(float((raw <= 0).mean()), 4),
       "note": "max_terms=500 kept at the deployed value for every arm"}
for n, (s, v) in A.items():
    out.setdefault("arm_sizes", {})[n] = int(s.sum())


def run(prop, norm, names):
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "pd"; d.mkdir()
        for n in names:
            s, v = A[n]
            with (d / f"{n}.tsv").open("w") as fh:
                for p, g, x in zip(prot[s], go[s], v):
                    fh.write(f"{p}\t{g}\t{x:.6f}\n")
        import shutil
        gtf = Path(td) / "gt.tsv"; shutil.copy(DS / "gt_pk_bp.tsv", gtf)
        rj = Path(td) / "r.json"; drv = Path(td) / "d.py"
        drv.write_text(DRIVER.format(obo=OBO, pd=str(d), gt=str(gtf), ia=IA, o=str(rj),
                                     prop=prop, norm=norm))
        r = subprocess.run([PY, str(drv)], capture_output=True, text=True, timeout=14400)
        if r.returncode != 0:
            print(f"FAILED prop={prop} norm={norm}: {r.stderr[-400:]}", flush=True)
            return {}
        best = {}
        for rec in json.loads(rj.read_text()).get("f_micro_w", []):
            if (rec.get("ns") or "") != "biological_process" or rec.get("f_micro_w") is None:
                continue
            fn = (rec.get("filename") or "").replace(".tsv", "")
            best[fn] = {k: (round(float(rec[k]), 4) if isinstance(rec.get(k), (int, float)) else rec.get(k))
                        for k in ("f_micro_w", "pr_micro_w", "rc_micro_w", "tau", "cov_max")}
        return best


t0 = time.time()
# --- (3) norm invariance --------------------------------------------------------------
out["norm_invariance"] = {}
for norm in ("cafa", "gt", "pred"):
    b = run("fill", norm, ["raw_ANCHOR"])
    out["norm_invariance"][norm] = b.get("raw_ANCHOR")
    print(f"[norm={norm:4s}] raw|fill -> {b.get('raw_ANCHOR')}  ({time.time()-t0:.0f}s)", flush=True)
fs = {k: (v or {}).get("f_micro_w") for k, v in out["norm_invariance"].items()}
out["norm_invariance"]["verdict"] = ("INVARIANT: norm cannot be the mechanism"
                                     if len(set(fs.values())) == 1 else f"NOT invariant: {fs}")
print(f"[3] {out['norm_invariance']['verdict']}\n", flush=True)

# --- (4) the scale / submission-threshold sweep ---------------------------------------
names = list(A.keys())
out["sweep_prop_fill_DEPLOYED"] = run("fill", "cafa", names)
print(f"\n=== prop=fill (the deployed call) ===  ({time.time()-t0:.0f}s)", flush=True)
for n in names:
    r = out["sweep_prop_fill_DEPLOYED"].get(n)
    if r: print(f"  {n:28s} f={r['f_micro_w']:.4f} pr={r['pr_micro_w']:.4f} rc={r['rc_micro_w']:.4f} "
                f"tau={r['tau']} n={out['arm_sizes'][n]}", flush=True)

out["sweep_prop_max_INVARIANT"] = run("max", "cafa", names)
print(f"\n=== prop=max (cafaeval's default) ===  ({time.time()-t0:.0f}s)", flush=True)
for n in names:
    r = out["sweep_prop_max_INVARIANT"].get(n)
    if r: print(f"  {n:28s} f={r['f_micro_w']:.4f} pr={r['pr_micro_w']:.4f} rc={r['rc_micro_w']:.4f} "
                f"tau={r['tau']} n={out['arm_sizes'][n]}", flush=True)

json.dump(out, open(W / "instrument_scale.json", "w"), indent=1, default=str)
print(f"\nDONE {time.time()-t0:.0f}s", flush=True)
