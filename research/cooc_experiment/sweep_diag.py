"""Why do three monotone maps give 0.1255 while raw gives 0.2131?

raw's optimum sits at tau=0.393, i.e. score > 0.393, NOT at the sign boundary. Under
minmax that same cut is tau ~= 0.722, which IS on the 0.001..0.999 grid. So the sweep
should find f=0.2131 and report it. It does not. Either the sweep is not returning its
maximum, or the prediction SET at a matched cut is not what I think it is.

This dumps the FULL sweep (df, not the best rows dfs) for raw and minmax, and lines them
up at MATCHED operating points via the exact affine map between the two scores. If f
matches cut-for-cut, the bug is in how the best row is selected. If it does not, the two
maps produce genuinely different prediction sets and the culprit is the parse.
"""
import json, subprocess, tempfile
from pathlib import Path
import numpy as np, pyarrow.parquet as pq
W=Path("/home/frapercan/Thesis2/storage/cooc_experiment")
DS=Path("/home/frapercan/Thesis2/repositories/protea-reranker-lab/datasets/protst-global-train227-test230")
OBO="/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
IA="/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv"
PY_="/home/frapercan/Thesis2/repositories/PROTEA/.venv/bin/python"
t=pq.read_table(W/"rerank_out"/"eval_scores.parquet")
c=np.asarray(t.column("category").to_pylist()); a=np.asarray(t.column("aspect").to_pylist())
m=(c=="pk")&(a=="bpo")
prot=np.asarray(t.column("protein_accession").to_pylist())[m]
go=np.asarray(t.column("go_term_id").to_pylist())[m]
rr=t.column("reranker_score").to_numpy(zero_copy_only=False).astype(np.float64)[m]
lo,hi=rr.min(),rr.max()
DRIVER='''
import json,signal
from cafaeval.evaluation import cafa_eval
signal.signal(signal.SIGTERM,signal.SIG_DFL)
df,dfs=cafa_eval("{obo}","{pd}","{gt}",ia="{ia}",prop="fill",norm="cafa",
    no_orphans=True,max_terms=500,th_step=0.001,n_cpu=1,weighted_only=False)
d=df.reset_index()
d=d[[col for col in d.columns if col in ("ns","tau","pr_micro_w","rc_micro_w","f_micro_w","cov_max","cov")]]
d=d[d["ns"]=="biological_process"]
json.dump(d.to_dict(orient="records"),open("{o}","w"),default=str)
'''
TRUTH=[tuple(l.rstrip("\n").split("\t")) for l in open(DS/"gt_pk_bp.tsv")]
def sweep(name,s):
    with tempfile.TemporaryDirectory() as td:
        d=Path(td)/"pd"; d.mkdir()
        with (d/f"{name}.tsv").open("w") as fh:
            for p,g,v in zip(prot,go,s): fh.write(f"{p}\t{g}\t{v:.6f}\n")
        gt=Path(td)/"gt.tsv"
        with gt.open("w") as fh:
            for p,g in TRUTH: fh.write(f"{p}\t{g}\n")
        raw=Path(td)/"r.json"; drv=Path(td)/"d.py"
        drv.write_text(DRIVER.format(obo=OBO,pd=str(d),gt=str(gt),ia=IA,o=str(raw)))
        r=subprocess.run([PY_,str(drv)],capture_output=True,text=True,timeout=5400)
        if r.returncode!=0: print(f"FAILED {name}: {r.stderr[-300:]}",flush=True); return []
        return json.loads(raw.read_text())
A=sweep("raw",rr)
B=sweep("minmax",(rr-lo)/(hi-lo))
def best(rows):
    v=[(float(x["f_micro_w"]),float(x["tau"])) for x in rows if x.get("f_micro_w") not in (None,"nan")]
    return max(v) if v else (None,None)
fa,ta=best(A); fb,tb=best(B)
print(f"raw    sweep: {len(A)} taus, MAX f={fa} at tau={ta}")
print(f"minmax sweep: {len(B)} taus, MAX f={fb} at tau={tb}")
print("\n=== matched operating points: minmax tau = (raw_tau - lo)/(hi - lo) ===")
print(f"  score range: [{lo:.4f}, {hi:.4f}]")
Bm={round(float(x['tau']),3):x for x in B}
print(f"\n{'raw tau':>8} {'raw f':>8} | {'-> minmax tau':>13} {'minmax f':>9}  {'MATCH?':>7}")
for x in A:
    try: rt=float(x["tau"]); rf=float(x["f_micro_w"])
    except: continue
    if rt not in (0.1,0.2,0.3,0.393,0.4,0.5,0.6,0.8): continue
    mt=round((rt-lo)/(hi-lo),3)
    y=Bm.get(mt)
    mf=float(y["f_micro_w"]) if y and y.get("f_micro_w") not in (None,"nan") else None
    ok="yes" if mf is not None and abs(mf-rf)<0.005 else "NO"
    print(f"{rt:>8.3f} {rf:>8.4f} | {mt:>13.3f} {str(round(mf,4)) if mf is not None else 'n/a':>9}  {ok:>7}")
json.dump({"raw_max":[fa,ta],"minmax_max":[fb,tb],"raw_sweep":A[:5],"minmax_sweep":B[:5]},
          open(W/"sweep_diag.json","w"),indent=1)
print("\nIf MATCH is NO, the two maps yield DIFFERENT prediction sets at the same cut,")
print("and the culprit is the parse, not the sweep.")
print("DONE",flush=True)
