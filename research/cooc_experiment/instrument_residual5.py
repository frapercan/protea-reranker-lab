"""Final control on the residual. max_terms=None cut dissent 26 -> 5. Earlier, widening the
harness write format .6f -> .12f cut it 26 -> 21. Those two account for 21 + 5 = 26 with no
overlap, which predicts: max_terms=None AND .12f => symdiff EXACTLY 0.

GATE: if symdiff is not 0, some part of the residual is still unidentified and I say so.
"""
import json, sys, tempfile, os
import numpy as np, pyarrow.parquet as pq
from pathlib import Path
sys.path.insert(0, "/home/frapercan/Thesis2/repositories/PROTEA/.venv/lib/python3.12/site-packages")
from cafaeval.parser import obo_parser, gt_parser, _pred_parser_legacy
from cafaeval.graph import propagate
W = Path("/home/frapercan/Thesis2/storage/cooc_experiment")
DS = Path("/home/frapercan/Thesis2/repositories/protea-reranker-lab/datasets/protst-global-train227-test230")
OBO="/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
IA="/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv"
NS="biological_process"; RAW_TAU=0.393
onts=obo_parser(OBO,("is_a","part_of"),IA,False); ont=onts[NS]
gts=gt_parser(str(DS/"gt_pk_bp.tsv"),onts); gt=gts[NS]; G=gt.matrix
t=pq.read_table(W/"rerank_out"/"eval_scores.parquet")
cat=np.asarray(t.column("category").to_pylist()); asp=np.asarray(t.column("aspect").to_pylist())
m=(cat=="pk")&(asp=="bpo")
prot=np.asarray(t.column("protein_accession").to_pylist())[m]
go=np.asarray(t.column("go_term_id").to_pylist())[m]
raw=t.column("reranker_score").to_numpy(zero_copy_only=False).astype(np.float64)[m]
lo,hi=float(raw.min()),float(raw.max()); MM=(RAW_TAU-lo)/(hi-lo); mm=(raw-lo)/(hi-lo)
toi=ont.toi_ia
def build(sc,mt,fmt):
    mat={NS:np.zeros(G.shape,dtype="float")}; rn={NS:np.zeros(G.shape[0],dtype=np.int32)}; ids={NS:{}}
    nsd={x:NS for x in ont.terms_dict}; nsd.update({x:NS for x in ont.terms_dict_alt})
    tix={NS:{x:i["index"] for x,i in ont.terms_dict.items()}}
    with tempfile.TemporaryDirectory() as td:
        f=os.path.join(td,"p.tsv")
        with open(f,"w") as fh:
            for p,g,v in zip(prot,go,sc): fh.write(f"{p}\t{g}\t{format(v,fmt)}\n")
        _pred_parser_legacy(f,onts,gts,nsd,tix,ids,mat,rn,{},mt)
    propagate(mat[NS],ont,ont.order,mode="max",parallel=1)
    return mat[NS]
out={}
for mt,fmt in ((500,".6f"),(None,".6f"),(500,".12f"),(None,".12f"),(None,".15g")):
    a=set(np.flatnonzero((build(raw,mt,fmt)[:,toi]>=RAW_TAU).ravel()).tolist())
    b=set(np.flatnonzero((build(mm,mt,fmt)[:,toi]>=MM).ravel()).tolist())
    k=f"max_terms={mt},fmt={fmt}"; out[k]=len(a^b)
    print(f"[{k:26s}] prop=max symdiff={len(a^b)}",flush=True)
z=out["max_terms=None,fmt=.12f"]
out["verdict"]=("FULLY IDENTIFIED: dissent = max_terms x clamp (21) + harness %.6f write "
  "precision (5). With both removed prop=max is EXACTLY scale-invariant."
  if z==0 else f"UNIDENTIFIED: {z} cells still dissent")
print("\n[GATE]",out["verdict"],flush=True)
json.dump(out,open(W/"instrument_residual5.json","w"),indent=1)
