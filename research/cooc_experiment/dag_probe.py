"""Probe: does GO-DAG structure (semantic proximity between a candidate BP term and
the protein's t0-known BP terms) carry ranking signal on PK-BPO that our current
features lack? Measures the feature's AUC alone, and whether it is decorrelated from
the reranker score (i.e. whether it could ADD to the ensemble)."""
import collections, json
from pathlib import Path
import numpy as np, pyarrow.parquet as pq
from scipy.sparse import lil_matrix, csr_matrix

W=Path("/home/frapercan/Thesis2/storage/cooc_experiment")
DS=Path("/home/frapercan/Thesis2/repositories/protea-reranker-lab/datasets/protst-global-train227-test230")
BASE=Path("/home/frapercan/Thesis2/CAFA_forever/data/releases/Sep_2025_Mar_2026")
OBO="/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"

# --- parse OBO: BP terms + is_a/part_of parents ---
parents=collections.defaultdict(set); ns={}
cur=None
for line in open(OBO):
    line=line.rstrip("\n")
    if line=="[Term]": cur=None; continue
    if line.startswith("id: GO:"): cur=line[4:]; continue
    if cur is None: continue
    if line.startswith("namespace: "): ns[cur]=line[11:]
    elif line.startswith("is_a: GO:"): parents[cur].add(line[6:16])
    elif line.startswith("relationship: part_of GO:"): parents[cur].add(line[22:32])
bp={t for t,v in ns.items() if v=="biological_process"}
print(f"OBO: {len(ns):,} terms, {len(bp):,} BP",flush=True)

# --- ancestor closure (BP only) ---
anc_cache={}
def anc(t):
    if t in anc_cache: return anc_cache[t]
    out=set(); stack=[t]
    while stack:
        x=stack.pop()
        for p in parents.get(x,()):
            if p not in out: out.add(p); stack.append(p)
    anc_cache[t]=out; return out
vocab={t:i for i,t in enumerate(sorted(bp))}
nV=len(vocab)
A=lil_matrix((nV,nV),dtype=np.int8)
for t,i in vocab.items():
    A[i,i]=1
    for a in anc(t):
        j=vocab.get(a)
        if j is not None: A[i,j]=1
A=A.tocsr()
asz=np.asarray(A.sum(axis=1)).ravel()
print(f"ancestor matrix: {A.shape}, nnz={A.nnz:,}",flush=True)

# --- known BP terms per PK protein ---
known_bp=collections.defaultdict(list)
for line in open(BASE/"groundtruth_PK_known.tsv"):
    if line.startswith("EntryID"): continue
    p,t,a=line.rstrip("\n").split("\t")
    if a=="P" and t in vocab: known_bp[p].append(vocab[t])

# --- PK-BPO candidates + labels + reranker score ---
t=pq.read_table(W/"rerank_out"/"eval_scores.parquet")
cat=np.asarray(t.column("category").to_pylist()); asp=np.asarray(t.column("aspect").to_pylist())
m=(cat=="pk")&(asp=="bpo")
P=np.asarray(t.column("protein_accession").to_pylist())[m]
G=np.asarray(t.column("go_term_id").to_pylist())[m]
y=(t.column("label").to_numpy(zero_copy_only=False).astype(np.int64)[m]>0).astype(np.int8)
rr=t.column("reranker_score").to_numpy(zero_copy_only=False).astype(np.float64)[m]

# --- DAG feature: max Jaccard(anc(cand), anc(known_k)) over the protein's known BP terms ---
feat=np.zeros(len(P),dtype=np.float32)
byprot=collections.defaultdict(list)
for i,p in enumerate(P): byprot[p].append(i)
done=0
for p,idxs in byprot.items():
    ks=known_bp.get(p)
    if not ks: continue
    u=np.zeros(nV,dtype=bool)
    for k in ks: u |= (A[k].toarray().ravel()>0)
    usz=u.sum()
    for i in idxs:
        j=vocab.get(G[i])
        if j is None: continue
        av=A[j].toarray().ravel()>0
        inter=np.count_nonzero(av & u)
        union=av.sum()+usz-inter
        feat[i]= inter/union if union else 0.0
    done+=1
    if done%500==0: print(f"  {done}/{len(byprot)} proteins",flush=True)
def auc(x,yy):
    o=np.argsort(x,kind="stable"); r=np.empty(len(x)); r[o]=np.arange(1,len(x)+1)
    npos=yy.sum(); return (r[yy==1].sum()-npos*(npos+1)/2)/(npos*(len(yy)-npos))
a_dag=auc(feat,y); a_rr=auc(rr,y)
corr=np.corrcoef(feat,rr)[0,1]
print(f"\n=== DAG-proximity feature on PK-BPO ===")
print(f"  AUC(dag_jaccard)   = {a_dag:.4f}")
print(f"  AUC(reranker)      = {a_rr:.4f}")
print(f"  corr(dag, reranker)= {corr:.4f}   (low corr => could ADD to the ensemble)")
# quick check: does a naive blend beat the reranker alone?
def rank(x):
    o=np.argsort(x,kind="stable"); r=np.empty(len(x)); r[o]=np.arange(len(x)); return r/max(1,len(x)-1)
for w in (0.1,0.2,0.3):
    b=(1-w)*rank(rr)+w*rank(feat)
    print(f"  blend {1-w:.1f}*reranker + {w:.1f}*dag -> AUC={auc(b,y):.4f}")
json.dump({"auc_dag":float(a_dag),"auc_reranker":float(a_rr),"corr":float(corr)},
          open(W/"dag_probe.json","w"),indent=1)
print("DONE",flush=True)
