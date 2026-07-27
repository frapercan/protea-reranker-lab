import sys, time, collections; sys.path.insert(0,"/home/frapercan/Thesis2/storage/joint_model")
import numpy as np, pyarrow.parquet as pq, torch, torch.nn as nn, scipy.sparse as sp
from trueframe import load_obo
t0=time.time(); DEV="cuda"; D=512
ROOT="/home/frapercan/Thesis2"; SC=f"{ROOT}/repositories/protea-reranker-lab/results/sparse_classifier"
par,ns,alt=load_obo(); BP={t for t,n in ns.items() if n=="biological_process"}
g=np.load(f"{SC}/go_text_emb.npz",allow_pickle=True)
tids=[alt.get(x,x) for x in g["go_ids"].tolist()]; te=g["emb"].astype(np.float32)
vocab=[];TEXT=[];seen=set()
for x,e in zip(tids,te):
    if x in BP and x not in seen: seen.add(x);vocab.append(x);TEXT.append(e)
TEXT=np.vstack(TEXT); vidx={x:i for i,x in enumerate(vocab)}; N=len(vocab)
print("N",N,"TEXT norm mean",np.linalg.norm(TEXT,axis=1).mean(),"std",TEXT.std())
# adjacency
rows,cols=[],[]
for x in vocab:
    i=vidx[x]
    for p in par.get(x,()):
        p=alt.get(p,p)
        if p in vidx: rows+=[i,vidx[p]];cols+=[vidx[p],i]
A=sp.coo_matrix((np.ones(len(rows),np.float32),(rows,cols)),shape=(N,N)); A=(A>0).astype(np.float32)+sp.eye(N,dtype=np.float32)
d=np.asarray(A.sum(1)).ravel(); Dm=sp.diags(1/np.sqrt(np.maximum(d,1e-8))); Ah=(Dm@A@Dm).tocoo()
Ah=torch.sparse_coo_tensor(np.vstack([Ah.row,Ah.col]),Ah.data.astype(np.float32),(N,N)).coalesce().to(DEV)
# codes + targets (train only, small sample)
clf=np.load(f"{SC}/clf_protein_codes.npz",allow_pickle=True)
code={a:c.astype(np.float32) for a,c in zip(clf["accs"].tolist(),clf["codes"])}
t=pq.read_table(f"{ROOT}/repositories/protea-reranker-lab/datasets/protst-global-train227-test230/train.parquet",
    columns=["protein_accession","go_term_id","label","aspect","snapshot_pair"])
asp=np.asarray(t.column("aspect").to_pylist());lab=t.column("label").to_numpy(zero_copy_only=False)
spp=np.asarray(t.column("snapshot_pair").to_pylist());Pn=np.asarray(t.column("protein_accession").to_pylist())
Gn=np.array([alt.get(x,x) for x in np.asarray(t.column("go_term_id").to_pylist())])
m=(asp=="bpo")&(lab>0)&(spp!="v225-v227")
pos=collections.defaultdict(set)
for p,gg in zip(Pn[m],Gn[m]):
    if gg in vidx and p in code: pos[p].add(vidx[gg])
prots=[p for p in pos if pos[p]][:4000]
print("sample train proteins",len(prots),"avg pos",np.mean([len(pos[p]) for p in prots]),"t",int(time.time()-t0))
X=torch.tensor(np.vstack([code[p] for p in prots]),dtype=torch.float32)
TEXT_t=torch.tensor(TEXT,device=DEV)
def asl(logits,targets,gn=4.,gp=1.,clip=.05,eps=1e-8):
    xs=torch.sigmoid(logits);xn=(1-xs+clip).clamp(max=1)
    los=targets*torch.log(xs.clamp(min=eps))+(1-targets)*torch.log(xn.clamp(min=eps))
    pt=xs*targets+(1-xs)*(1-targets);gg=gp*targets+gn*(1-targets)
    return -(los*torch.pow(1-pt,gg)).sum()/logits.shape[0]
class M(nn.Module):
    def __init__(s):
        super().__init__()
        s.pt=nn.Sequential(nn.Linear(2048,D),nn.LayerNorm(D),nn.GELU(),nn.Dropout(0.3),nn.Linear(D,D))
        s.txt=nn.Linear(768,D);s.g1=nn.Linear(D,D);s.g2=nn.Linear(D,D)
        s.ln_g=nn.LayerNorm(D);s.ln_f=nn.LayerNorm(D)
        s.scale=nn.Parameter(torch.tensor(1/np.sqrt(D)));s.bias=nn.Parameter(torch.zeros(N))
    def labels(s):
        h=s.txt(TEXT_t);h1=torch.relu(torch.sparse.mm(Ah,s.g1(h)));h2=torch.sparse.mm(Ah,s.g2(h1));return s.ln_g(h+h2)
    def forward(s,x,L):
        h=s.pt(x);att=torch.softmax((h@L.t())*s.scale,dim=1);ctx=att@L;hf=s.ln_f(h+ctx);return (hf@L.t())+s.bias
torch.manual_seed(42)
m=M().to(DEV);opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-5)
BS=128;order=np.arange(len(prots))
for step in range(200):
    bi=np.random.choice(order,BS,replace=False)
    xb=X[bi].to(DEV)
    yb=torch.zeros(BS,N,device=DEV)
    for r,ii in enumerate(bi):
        yb[r,list(pos[prots[ii]])]=1.
    L=m.labels();lg=m(xb,L);loss=asl(lg,yb)
    opt.zero_grad();loss.backward()
    gnorm=float(torch.sqrt(sum((p.grad**2).sum() for p in m.parameters() if p.grad is not None)))
    bgrad=float(m.bias.grad.abs().mean())
    opt.step()
    if step%25==0: print(f"step {step:3d} loss {float(loss):.2f} gnorm {gnorm:.3f} bias|grad| {bgrad:.4f} bias.std {float(m.bias.std()):.3f}")
print("t",int(time.time()-t0))
