"""What ranking quality would PK-BPO need? Measures the reranker's AUC on the pool,
then calibrates the AUC -> f_micro_w curve by blending the oracle label with noise to
hit target AUCs. Tells us the AUC target required to clear the TransFew gap."""
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
df,dfs=cafa_eval("{obo}","{pred_dir}","{gt}",ia="{ia}",prop="fill",norm="cafa",
    no_orphans=True,max_terms=500,th_step=0.001,n_cpu=1,weighted_only=False)
out={{}}
for k,v in dfs.items(): out[k]=v.reset_index().to_dict(orient="records")
json.dump(out,open("{out_json}","w"),default=str)
'''
def cafaeval(pd_,gt,td):
    raw=Path(td)/"r.json"; drv=Path(td)/"d.py"
    drv.write_text(DRIVER.format(obo=OBO,pred_dir=pd_,gt=gt,ia=IA,out_json=str(raw)))
    p=subprocess.run([PY,str(drv)],capture_output=True,text=True,timeout=3600)
    if p.returncode!=0: return None
    d=json.loads(raw.read_text())
    for rec in d.get("f_micro_w",[]):
        if (rec.get("ns") or "")=="biological_process" and rec.get("f_micro_w") is not None:
            return round(float(rec["f_micro_w"]),4)
    return None
def write(name,P,G,S,truth,td):
    d=Path(td)/name; pd_=d/"pred_dir"; pd_.mkdir(parents=True,exist_ok=True)
    with (pd_/f"{name}.tsv").open("w") as fh:
        for p,g,s in zip(P,G,S): fh.write(f"{p}\t{g}\t{s:.6f}\n")
    gt=d/"gt.tsv"
    with gt.open("w") as fh:
        for p,g in truth: fh.write(f"{p}\t{g}\n")
    return str(pd_),str(gt)
def auc(x,y):
    o=np.argsort(x,kind="stable"); r=np.empty(len(x)); r[o]=np.arange(1,len(x)+1)
    npos=y.sum(); return (r[y==1].sum()-npos*(npos+1)/2)/(npos*(len(y)-npos))
t=pq.read_table(W/"rerank_out"/"eval_scores.parquet")
cat=np.asarray(t.column("category").to_pylist()); asp=np.asarray(t.column("aspect").to_pylist())
m=(cat=="pk")&(asp=="bpo")
P=np.asarray(t.column("protein_accession").to_pylist())[m]
G=np.asarray(t.column("go_term_id").to_pylist())[m]
y=(t.column("label").to_numpy(zero_copy_only=False).astype(np.int64)[m]>0).astype(np.int8)
rr=t.column("reranker_score").to_numpy(zero_copy_only=False).astype(np.float64)[m]
truth=[tuple(l.rstrip("\n").split("\t")) for l in open(DS/"gt_pk_bp.tsv")]
print(f"PK-BPO: {m.sum():,} cands, pos_rate={y.mean():.4f}",flush=True)
print(f"  RERANKER AUC on the pool = {auc(rr,y):.4f}",flush=True)
rng=np.random.default_rng(42)
res={"reranker_auc":round(float(auc(rr,y)),4)}
with tempfile.TemporaryDirectory() as td:
    # blend oracle with noise to hit target AUCs
    for w in (0.0,0.15,0.3,0.5,0.75,1.0):
        s=w*y.astype(float)+(1-w)*rng.random(len(y))
        a=auc(s,y)
        pd_,gt=write(f"a{w}",P,G,s,truth,td)
        f=cafaeval(pd_,gt,td)
        res[f"blend_{w}"]={"auc":round(float(a),4),"f_micro_w":f}
        print(f"  blend w={w:<4} -> AUC={a:.4f}  f_micro_w={f}",flush=True)
(W/"auc_to_f.json").write_text(json.dumps(res,indent=1))
print("DONE",flush=True)
