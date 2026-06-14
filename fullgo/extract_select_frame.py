"""SELECT-frame data for classifier validation: v220 labels + 6-PLM embeddings.
Train = v220 experimental proteins; eval = SELECT GT proteins. Saves /tmp/sel_data.npz."""
import numpy as np, psycopg
from collections import defaultdict
DB="postgresql://protea:protea@localhost:5432/protea"
V220="1559d9f7-195d-4892-af16-8b58f7fc9942"
EXP=('EXP','IDA','IMP','IPI','IGI','IEP','TAS','IC','HTP','HDA','HMP','HGI','HEP')
PLMS=["08234f06-ba76-4d7d-aaec-ae601096b4fa","55e43f1c-1a3b-4b1d-88c0-26b433f5f673",
      "238f79b1-3068-4c6f-9013-5cc52b4f662b","c2e9dda3-e505-4170-b50d-435a451761ac",
      "2bf1e753-022f-44b8-a131-9a90acb4024e","084943c6-fec1-441d-bdc5-63b0268ada1b"]
OBO="/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
TOI=set(l.strip() for l in open("/home/frapercan/Thesis2/CAFA_forever/data/releases/Sep_2025_Mar_2026/groundtruth_terms_of_interest.txt") if l.startswith("GO:"))
SEL_EVAL=sorted(set(l.strip() for l in open("/tmp/select_gt_prots.txt") if l.strip()))
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
conn=psycopg.connect(DB); cur=conn.cursor()
cur.execute("select a.protein_accession,g.go_id from protein_go_annotation a join go_term g on g.id=a.go_term_id where a.annotation_set_id=%s and a.evidence_code=any(%s) and coalesce(a.qualifier,'') not like '%%NOT%%' and g.aspect in ('F','P','C')",(V220,list(EXP)))
leaf=defaultdict(set)
for acc,go in cur: leaf[acc].add(go)
prop={}; vc=defaultdict(int)
for acc,ls in leaf.items():
    s=set(ls)
    for t in ls: s|=anc(t)
    s&=TOI
    if s:
        prop[acc]=s
        for t in s: vc[t]+=1
vocab=sorted(vc); tidx={t:i for i,t in enumerate(vocab)}
tr_acc=sorted(prop); print(f"v220 train={len(tr_acc)} vocab={len(vocab)} eval={len(SEL_EVAL)}",flush=True)
need=sorted(set(tr_acc)|set(SEL_EVAL))
# fetch 6-PLM concat
embcat={a:[] for a in need}; dims=[]
for cfg in PLMS:
    emb={}; B=5000
    for i in range(0,len(need),B):
        ch=need[i:i+B]
        cur.execute("select p.accession,e.embedding::text from protein p join sequence s on s.id=p.sequence_id join sequence_embedding e on e.sequence_id=s.id where e.embedding_config_id=%s and p.accession=any(%s)",(cfg,ch))
        for a,v in cur:
            if a in embcat and not (len(embcat[a])>len(dims)): emb[a]=np.fromstring(v.strip("[]"),sep=",",dtype=np.float32)
    dim=len(next(iter(emb.values()))); dims.append(dim)
    for a in need: embcat[a].append(emb.get(a,np.zeros(dim,np.float32)))
    print(f"{cfg[:8]} dim {dim} n {len(emb)}",flush=True)
conn.close()
D=sum(dims)
def mat(accs):
    M=np.zeros((len(accs),D),np.float32)
    for i,a in enumerate(accs):
        M[i]=np.concatenate(embcat[a])
    return M
Xtr=mat(tr_acc); Xev=mat(SEL_EVAL)
rows=[]; cols=[]
for i,a in enumerate(tr_acc):
    for t in prop[a]:
        j=tidx.get(t)
        if j is not None: rows.append(i); cols.append(j)
print("Xtr",Xtr.shape,"Xev",Xev.shape,"nnz",len(rows),flush=True)
np.savez_compressed("/tmp/sel_data.npz",Xtr=Xtr,rows=np.array(rows,np.int32),cols=np.array(cols,np.int32),tr_acc=np.array(tr_acc),Xev=Xev,ev_acc=np.array(SEL_EVAL),vocab=np.array(vocab))
print("saved /tmp/sel_data.npz",flush=True)
