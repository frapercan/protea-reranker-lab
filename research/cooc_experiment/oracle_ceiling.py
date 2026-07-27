"""Oracle ceiling of the CURRENT PK-BPO pool: score candidates by their true label
and cafaeval. This is the best f_micro_w any ranker could extract from the pool we
already retrieve. Compares against the real reranker baseline on the same harness."""
import json, subprocess, tempfile, collections
from pathlib import Path
import numpy as np, pyarrow.parquet as pq
W=Path("/home/frapercan/Thesis2/storage/cooc_experiment")
DS=Path("/home/frapercan/Thesis2/repositories/protea-reranker-lab/datasets/protst-global-train227-test230")
OBO="/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
IA="/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv"
PY="/home/frapercan/Thesis2/repositories/PROTEA/.venv/bin/python"
DRIVER='''
import json, signal
from cafaeval.evaluation import cafa_eval
signal.signal(signal.SIGTERM, signal.SIG_DFL)
df, dfs_best = cafa_eval("{obo}", "{pred_dir}", "{gt}", ia="{ia}",
    prop="fill", norm="cafa", no_orphans=True, max_terms=500, th_step=0.001,
    n_cpu=1, weighted_only=False)
out={{}}
for kind, dfb in dfs_best.items(): out[kind]=dfb.reset_index().to_dict(orient="records")
json.dump(out, open("{out_json}","w"), default=str)
'''
def cafaeval(pred_dir, gt, td):
    raw=Path(td)/"raw.json"; drv=Path(td)/"drv.py"
    drv.write_text(DRIVER.format(obo=OBO,pred_dir=pred_dir,gt=gt,ia=IA,out_json=str(raw)))
    p=subprocess.run([PY,str(drv)],capture_output=True,text=True,timeout=3600)
    if p.returncode!=0: return {"error":p.stderr[-300:]}
    d=json.loads(raw.read_text())
    for rec in d.get("f_micro_w",[]):
        if (rec.get("ns") or "")=="biological_process" and rec.get("f_micro_w") is not None:
            return {"f_micro_w":round(float(rec["f_micro_w"]),4)}
    return {"f_micro_w":None}
def write(name,P,G,S,truth,td):
    d=Path(td)/name; pd_=d/"pred_dir"; pd_.mkdir(parents=True,exist_ok=True)
    with (pd_/f"{name}.tsv").open("w") as fh:
        for p,g,s in zip(P,G,S): fh.write(f"{p}\t{g}\t{s:.6f}\n")
    gt=d/"gt.tsv"
    with gt.open("w") as fh:
        for p,g in truth: fh.write(f"{p}\t{g}\n")
    return str(pd_),str(gt)
t=pq.read_table(W/"rerank_out"/"eval_scores.parquet")
cat=np.asarray(t.column("category").to_pylist()); asp=np.asarray(t.column("aspect").to_pylist())
m=(cat=="pk")&(asp=="bpo")
P=np.asarray(t.column("protein_accession").to_pylist())[m]
G=np.asarray(t.column("go_term_id").to_pylist())[m]
lab=t.column("label").to_numpy(zero_copy_only=False).astype(np.int64)[m]
rr=t.column("reranker_score").to_numpy(zero_copy_only=False).astype(np.float64)[m]
o=np.argsort(rr,kind="stable"); rrn=np.empty(len(rr)); rrn[o]=np.arange(len(rr)); rrn/=max(1,len(rr)-1)
truth=[tuple(l.rstrip("\n").split("\t")) for l in open(DS/"gt_pk_bp.tsv")]
pool=collections.defaultdict(set)
for p,g in zip(P,G): pool[p].add(g)
reach=sum(1 for p,g in truth if g in pool.get(p,()))
print(f"pool rows={m.sum():,} | full gt={len(truth):,} | reachable={reach:,} (recall {reach/len(truth):.3f})",flush=True)
res={}
with tempfile.TemporaryDirectory() as td:
    pd_,gt=write("reranker",P,G,rrn,truth,td); res["reranker_baseline"]=cafaeval(pd_,gt,td)
    print(f"  RERANKER (what we deliver) f_micro_w={res['reranker_baseline'].get('f_micro_w')}",flush=True)
    pd_,gt=write("oracle",P,G,lab.astype(float),truth,td); res["oracle_pool_ceiling"]=cafaeval(pd_,gt,td)
    print(f"  ORACLE   (perfect ranking of the SAME pool) f_micro_w={res['oracle_pool_ceiling'].get('f_micro_w')}",flush=True)
(W/"oracle_ceiling_result.json").write_text(json.dumps(res,indent=1))
b=res["reranker_baseline"].get("f_micro_w"); c=res["oracle_pool_ceiling"].get("f_micro_w")
if b and c: print(f"\n  => ranking headroom in the pool we ALREADY have: {b} -> {c}  (capture {100*b/c:.1f}% of the ceiling)")
print("DONE",flush=True)
