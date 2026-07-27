"""Does the binary-objective AUC gain (0.7903 -> 0.8227) convert to f_micro_w on
PK-BPO? Trains the binary PK-BPO scorer, predicts the eval pool, and runs the SAME
cafaeval harness against the FULL ground truth used for every other arm today.
No new data: same 72 features, same rows. Only the objective changes."""
import json, subprocess, tempfile
from pathlib import Path
import numpy as np, pyarrow.parquet as pq, lightgbm as lgb
W=Path("/home/frapercan/Thesis2/storage/cooc_experiment")
DS=Path("/home/frapercan/Thesis2/repositories/protea-reranker-lab/datasets/protst-global-train227-test230")
OBO="/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
IA="/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv"
PY="/home/frapercan/Thesis2/repositories/PROTEA/.venv/bin/python"
VAL="v225-v227"
EX={"protein_accession","go_term_id","label","category","snapshot_pair","qualifier",
    "evidence_code","taxonomic_relation","aspect"}
feats=[c for c in pq.ParquetFile(DS/"train.parquet").schema_arrow.names if c not in EX]
def load(p):
    cols=feats+["category","aspect","snapshot_pair","protein_accession","go_term_id","label"]
    t=pq.read_table(p,columns=list(dict.fromkeys(cols)))
    cat=np.asarray(t.column("category").to_pylist()); asp=np.asarray(t.column("aspect").to_pylist())
    m=(cat=="pk")&(asp=="bpo")
    X=np.empty((int(m.sum()),len(feats)),dtype=np.float32)
    for j,c in enumerate(feats): X[:,j]=t.column(c).to_numpy(zero_copy_only=False).astype(np.float32)[m]
    md={k:np.asarray(t.column(k).to_pylist())[m] for k in ("snapshot_pair","protein_accession","go_term_id")}
    md["label"]=(t.column("label").to_numpy(zero_copy_only=False).astype(np.int64)[m]>0).astype(np.int32)
    return X,md
Xtr,mtr=load(DS/"train.parquet"); Xev,mev=load(DS/"eval.parquet")
iv=mtr["snapshot_pair"]==VAL; tr=~iv
P={"objective":"binary","metric":["auc"],"learning_rate":0.05,"num_leaves":63,
   "min_data_in_leaf":100,"feature_fraction":0.9,"bagging_fraction":0.9,"bagging_freq":5,
   "seed":42,"verbose":-1,"num_threads":12,"max_bin":63,"two_round":True,"force_col_wise":True}
b=lgb.train(P,lgb.Dataset(Xtr[tr],label=mtr["label"][tr],feature_name=feats),
            num_boost_round=2000,
            valid_sets=[lgb.Dataset(Xtr[iv],label=mtr["label"][iv],feature_name=feats)],
            callbacks=[lgb.early_stopping(50,verbose=False)])
s=b.predict(Xev,num_iteration=b.best_iteration)
print(f"binary scorer trained, best_iter={b.best_iteration}",flush=True)
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
truth=[tuple(l.rstrip("\n").split("\t")) for l in open(DS/"gt_pk_bp.tsv")]
with tempfile.TemporaryDirectory() as td:
    d=Path(td)/"pred_dir"; d.mkdir(parents=True)
    with (d/"binary.tsv").open("w") as fh:
        for p,g,v in zip(mev["protein_accession"],mev["go_term_id"],s): fh.write(f"{p}\t{g}\t{v:.6f}\n")
    gt=Path(td)/"gt.tsv"
    with gt.open("w") as fh:
        for p,g in truth: fh.write(f"{p}\t{g}\n")
    raw=Path(td)/"r.json"; drv=Path(td)/"d.py"
    drv.write_text(DRIVER.format(obo=OBO,pred_dir=str(d),gt=str(gt),ia=IA,out_json=str(raw)))
    r=subprocess.run([PY,str(drv)],capture_output=True,text=True,timeout=3600)
    if r.returncode!=0: print("cafaeval FAILED:",r.stderr[-400:]); raise SystemExit(1)
    data=json.loads(raw.read_text()); f=None
    for rec in data.get("f_micro_w",[]):
        if (rec.get("ns") or "")=="biological_process" and rec.get("f_micro_w") is not None:
            f=round(float(rec["f_micro_w"]),4)
    print(f"\n=== PK-BPO f_micro_w ===")
    print(f"  lambdarank baseline (AUC 0.7903) = 0.1255")
    print(f"  BINARY objective    (AUC 0.8227) = {f}")
    if f: print(f"  delta = {f-0.1255:+.4f}")
    print(f"  (TransFew on the public board = 0.2943; our board number = 0.2181)")
    json.dump({"binary_f_micro_w":f,"baseline":0.1255},open(W/"binary_to_f.json","w"),indent=1)
print("DONE",flush=True)
