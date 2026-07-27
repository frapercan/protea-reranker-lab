"""Is 0.2131 the reranker, or an accidental prefilter?

cafa_eval writes into a zero-initialised matrix with `if prob_f > old` (parser.py:230),
so a NEGATIVE score is never stored. Our lambdarank margins run [-8.48, 3.81] and 80.9%
of pk-bpo rows are <= 0: 498,648 candidates are annihilated at parse. The deployed anchor
therefore contains a free hard prefilter, "keep only what the reranker scored > 0", that
nobody chose.

This also explains rankpct at last: it is monotone, but monotone is NOT invariant here.
It moves rows ACROSS the zero boundary, from discarded to kept, re-inflating negative-margin
junk. The metric is not just a threshold sweep; it clamps first.

ARMS. Same booster, same rows, same gt, same harness. Only the score MAP varies, and every
map is strictly monotone, so the ORDER is identical in all four.

  raw        as deployed. 80.9% of rows never reach the metric.
  sigmoid    1/(1+e^-x). All rows survive; the sign boundary lands at tau=0.5, so the
             sweep CAN still choose exactly the raw arm's cut, plus every other.
  minmax     (x-min)/(max-min). All rows survive, spacing preserved.
  rankpct    the known artefact, for the ledger.

READ IT: sigmoid/minmax explore a SUPERSET of raw's operating points, so they should be
>= raw. If they are materially ABOVE, the clamp is COSTING us and the deployed number
understates the reranker. If they are equal, the clamp happens to sit at the optimum.
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
DRIVER='''
import json,signal
from cafaeval.evaluation import cafa_eval
signal.signal(signal.SIGTERM,signal.SIG_DFL)
df,dfs=cafa_eval("{obo}","{pd}","{gt}",ia="{ia}",prop="fill",norm="cafa",
    no_orphans=True,max_terms=500,th_step=0.001,n_cpu=1,weighted_only=False)
out={{}}
for k,v in dfs.items(): out[k]=v.reset_index().to_dict(orient="records")
json.dump(out,open("{o}","w"),default=str)
'''
TRUTH=[tuple(l.rstrip("\n").split("\t")) for l in open(DS/"gt_pk_bp.tsv")]
def score(name,s):
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
        if r.returncode!=0: print(f"  FAILED {name}: {r.stderr[-200:]}",flush=True); return None
        for rec in json.loads(raw.read_text()).get("f_micro_w",[]):
            if (rec.get("ns") or "")=="biological_process" and rec.get("f_micro_w") is not None:
                return {k:round(float(rec[k]),4) for k in ("f_micro_w","pr_micro_w","rc_micro_w","tau","cov_max")}
def rankpct(x):
    o=np.argsort(x,kind="stable"); r=np.empty(len(x)); r[o]=np.arange(len(x)); return r/max(1,len(x)-1)
res={"rows":len(rr),"frac_le_zero_dropped_by_raw":round(float((rr<=0).mean()),4)}
arms=[("raw",rr),("sigmoid",1.0/(1.0+np.exp(-rr))),
      ("minmax",(rr-rr.min())/(rr.max()-rr.min())),("rankpct",rankpct(rr))]
for n,s in arms:
    res[n]=score(n,s); print(f"  {n:9s} {res[n]}",flush=True)
    json.dump(res,open(W/"clamp_test.json","w"),indent=1)
g=lambda k:(res.get(k) or {}).get("f_micro_w")
print("\n=== Is the anchor the reranker, or the clamp? ===",flush=True)
for n,_ in arms: print(f"  {n:9s} = {g(n)}",flush=True)
if g("raw") and g("sigmoid"):
    print(f"  -> sigmoid - raw = {g('sigmoid')-g('raw'):+.4f}",flush=True)
    print("     ABOVE  => the clamp is COSTING us; the deployed number understates the reranker.",flush=True)
    print("     EQUAL  => the sign boundary happens to sit at the optimum.",flush=True)
print("DONE",flush=True)
