"""Extract t0 propagated experimental labels for eval proteins (label-aware input channel).
For a frame: eval protein list + annotation_set + frame vocab -> per-protein label index list.
Saves <out>.npz with ev_acc, lab_rows, lab_cols (into the frame vocab)."""
import sys, numpy as np, psycopg
from collections import defaultdict
DB="postgresql://protea:protea@localhost:5432/protea"
EXP=('EXP','IDA','IMP','IPI','IGI','IEP','TAS','IC','HTP','HDA','HMP','HGI','HEP')
OBO="/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
ANNSET=sys.argv[1]; FRAME_NPZ=sys.argv[2]; OUT=sys.argv[3]
d=np.load(FRAME_NPZ,allow_pickle=True)
vocab=[str(x) for x in d["vocab"]]; tidx={t:i for i,t in enumerate(vocab)}
ev_acc=[str(x) for x in d["ev_acc"]]
par=defaultdict(set); cur=None
for line in open(OBO):
    line=line.strip()
    if line=="[Term]": cur=None
    elif line.startswith("id: GO:"): cur=line[4:]
    elif line.startswith("is_a:") and cur: par[cur].add(line.split()[1])
    elif line.startswith("relationship: part_of") and cur:
        p=line.split()
        if len(p)>=3: par[cur].add(p[2])
ac={}
def anc(t):
    if t in ac: return ac[t]
    o=set(); st=list(par.get(t,()))
    while st:
        a=st.pop()
        if a in o: continue
        o.add(a); st.extend(par.get(a,()))
    ac[t]=o; return o
conn=psycopg.connect(DB); c=conn.cursor()
evset=set(ev_acc)
c.execute("select a.protein_accession,g.go_id from protein_go_annotation a join go_term g on g.id=a.go_term_id where a.annotation_set_id=%s and a.evidence_code=any(%s) and coalesce(a.qualifier,'') not like '%%NOT%%' and g.aspect in ('F','P','C')",(ANNSET,list(EXP)))
leaf=defaultdict(set)
for acc,go in c:
    if acc in evset: leaf[acc].add(go)
conn.close()
accidx={a:i for i,a in enumerate(ev_acc)}
rows=[]; cols=[]
for acc,ls in leaf.items():
    s=set(ls)
    for t in ls: s|=anc(t)
    i=accidx[acc]
    for t in s:
        j=tidx.get(t)
        if j is not None: rows.append(i); cols.append(j)
print(f"eval proteins with t0 labels={len(leaf)} nnz={len(rows)}",flush=True)
np.savez_compressed(OUT,ev_acc=np.array(ev_acc),lab_rows=np.array(rows,np.int32),lab_cols=np.array(cols,np.int32))
print(f"saved {OUT}",flush=True)
