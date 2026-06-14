import numpy as np, psycopg
DB="postgresql://protea:protea@localhost:5432/protea"
EXTRA=["c2e9dda3-e505-4170-b50d-435a451761ac",  # ESM2-650M 1280
       "2bf1e753-022f-44b8-a131-9a90acb4024e",  # ESMC-600M 1152
       "084943c6-fec1-441d-bdc5-63b0268ada1b"]  # ProtT5 1024
d=np.load("/tmp/m0v2_data.npz",allow_pickle=True)
tr_acc=[str(x) for x in d["tr_acc"]]; ev_acc=[str(x) for x in d["ev_acc"]]
need=sorted(set(tr_acc)|set(ev_acc))
conn=psycopg.connect(DB); cur=conn.cursor(); emaps=[]
for cfg in EXTRA:
    emb={}; B=5000
    for i in range(0,len(need),B):
        ch=need[i:i+B]
        cur.execute("select p.accession,e.embedding::text from protein p join sequence s on s.id=p.sequence_id join sequence_embedding e on e.sequence_id=s.id where e.embedding_config_id=%s and p.accession=any(%s)",(cfg,ch))
        for a,v in cur:
            if a not in emb: emb[a]=np.fromstring(v.strip("[]"),sep=",",dtype=np.float32)
    dim=len(next(iter(emb.values()))); print(f"{cfg[:8]} dim {dim} n {len(emb)}",flush=True); emaps.append((emb,dim))
conn.close()
def stack(accs,base):
    parts=[base]
    for emb,dim in emaps:
        M=np.zeros((len(accs),dim),np.float32)
        for i,a in enumerate(accs):
            v=emb.get(a)
            if v is not None and len(v)==dim: M[i]=v
        parts.append(M)
    return np.hstack(parts).astype(np.float32)
Xtr=stack(tr_acc,d["Xtr"]); Xev=stack(ev_acc,d["Xev"])
print("Xtr",Xtr.shape,"Xev",Xev.shape,flush=True)
np.savez_compressed("/tmp/m0v3_data.npz",Xtr=Xtr,rows=d["rows"],cols=d["cols"],tr_acc=d["tr_acc"],Xev=Xev,ev_acc=d["ev_acc"],vocab=d["vocab"])
print("saved /tmp/m0v3_data.npz",flush=True)
