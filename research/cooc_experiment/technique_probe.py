"""Technique probe on PK-BPO: no new data, no structures, no text. Same features.
Tests two hypotheses against the current reranker (AUC 0.7903):
  A) per-(category x aspect) model: train on PK-BPO rows only, instead of pk with
     MFO/BPO/CCO pooled.
  B) objective mismatch: lambdarank optimises WITHIN-protein order, but f_micro_w
     sweeps ONE GLOBAL threshold across proteins, so cross-protein comparability is
     what actually matters. Try binary logloss (a calibrated global scorer).
Reports AUC on the PK-BPO eval subset for each.
"""
import json
from pathlib import Path
import numpy as np, pyarrow.parquet as pq, lightgbm as lgb, pandas as pd

W=Path("/home/frapercan/Thesis2/storage/cooc_experiment")
DS=Path("/home/frapercan/Thesis2/repositories/protea-reranker-lab/datasets/protst-global-train227-test230")
VAL_PAIR="v225-v227"
EXCLUDE={"protein_accession","go_term_id","label","category","snapshot_pair",
         "qualifier","evidence_code","taxonomic_relation","aspect"}
feats=[c for c in pq.ParquetFile(DS/"train.parquet").schema_arrow.names if c not in EXCLUDE]
print(f"n_features={len(feats)}",flush=True)

def load(path):
    cols=feats+["category","aspect","snapshot_pair","protein_accession","label"]
    t=pq.read_table(path,columns=list(dict.fromkeys(cols)))
    cat=np.asarray(t.column("category").to_pylist()); asp=np.asarray(t.column("aspect").to_pylist())
    m=(cat=="pk")&(asp=="bpo")
    X=np.empty((int(m.sum()),len(feats)),dtype=np.float32)
    for j,c in enumerate(feats):
        X[:,j]=t.column(c).to_numpy(zero_copy_only=False).astype(np.float32)[m]
    meta={k:np.asarray(t.column(k).to_pylist())[m] for k in ("snapshot_pair","protein_accession")}
    meta["label"]=(t.column("label").to_numpy(zero_copy_only=False).astype(np.int64)[m]>0).astype(np.int32)
    return X,meta
print("loading PK-BPO train...",flush=True)
Xtr,mtr=load(DS/"train.parquet")
print(f"  PK-BPO train rows={len(mtr['label']):,} pos={mtr['label'].mean():.4f}",flush=True)
print("loading PK-BPO eval...",flush=True)
Xev,mev=load(DS/"eval.parquet")
print(f"  PK-BPO eval rows={len(mev['label']):,}",flush=True)

is_val=mtr["snapshot_pair"]==VAL_PAIR; tr=~is_val
def groups(snap,prot):
    key=pd.factorize(pd.Series(snap)+"|"+pd.Series(prot))[0]
    order=np.argsort(key,kind="stable")
    _,sizes=np.unique(key[order],return_counts=True)
    return order,sizes.astype(np.int32)
def auc(x,y):
    o=np.argsort(x,kind="stable"); r=np.empty(len(x)); r[o]=np.arange(1,len(x)+1)
    npos=y.sum(); return (r[y==1].sum()-npos*(npos+1)/2)/(npos*(len(y)-npos))
BASE={"learning_rate":0.05,"num_leaves":63,"min_data_in_leaf":100,"feature_fraction":0.9,
      "bagging_fraction":0.9,"bagging_freq":5,"seed":42,"verbose":-1,"num_threads":12,
      "max_bin":63,"two_round":True,"force_col_wise":True}
res={"current_reranker_auc":0.7903}

# --- A: per-(cat x aspect) lambdarank on PK-BPO only ---
o_tr,g_tr=groups(mtr["snapshot_pair"][tr],mtr["protein_accession"][tr])
o_va,g_va=groups(mtr["snapshot_pair"][is_val],mtr["protein_accession"][is_val])
p=dict(BASE); p.update({"objective":"lambdarank","metric":["ndcg"],"ndcg_eval_at":[5,10],"label_gain":[0,1]})
dtr=lgb.Dataset(Xtr[tr][o_tr],label=mtr["label"][tr][o_tr],group=g_tr,feature_name=feats)
dva=lgb.Dataset(Xtr[is_val][o_va],label=mtr["label"][is_val][o_va],group=g_va,reference=dtr,feature_name=feats)
b=lgb.train(p,dtr,num_boost_round=2000,valid_sets=[dva],valid_names=["val"],
            callbacks=[lgb.early_stopping(50,verbose=False)])
a=auc(b.predict(Xev,num_iteration=b.best_iteration),mev["label"])
res["A_lambdarank_pkbpo_only"]={"auc":float(a),"best_iter":int(b.best_iteration or 0)}
print(f"\n  A) lambdarank, PK-BPO-only model : AUC={a:.4f} (best_iter={b.best_iteration})",flush=True)

# --- B: binary logloss on PK-BPO only (globally calibrated scorer) ---
p=dict(BASE); p.update({"objective":"binary","metric":["auc"]})
dtr=lgb.Dataset(Xtr[tr],label=mtr["label"][tr],feature_name=feats)
dva=lgb.Dataset(Xtr[is_val],label=mtr["label"][is_val],reference=dtr,feature_name=feats)
b2=lgb.train(p,dtr,num_boost_round=2000,valid_sets=[dva],valid_names=["val"],
             callbacks=[lgb.early_stopping(50,verbose=False)])
a2=auc(b2.predict(Xev,num_iteration=b2.best_iteration),mev["label"])
res["B_binary_pkbpo_only"]={"auc":float(a2),"best_iter":int(b2.best_iteration or 0)}
print(f"  B) binary logloss, PK-BPO-only   : AUC={a2:.4f} (best_iter={b2.best_iteration})",flush=True)
print(f"\n  baseline (pooled-aspect lambdarank): AUC=0.7903")
print(f"  target: ~0.834 would be worth ~+0.11 f_micro_w")
json.dump(res,open(W/"technique_probe.json","w"),indent=1)
print("DONE",flush=True)
