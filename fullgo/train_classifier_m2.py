"""M2 label-semantics (hybrid): base independent head + label-similarity head
(protein_rep @ anc2vec[vocab].T). Tests whether GO-DAG label semantics lift LK.
Usage: train_classifier_m2.py DATA PRED [--seed N]
Run once per seed (e.g. base/7/137) then combine with seed_average.py."""
import sys, numpy as np, torch, torch.nn as nn
from scipy.sparse import csr_matrix
DATA=sys.argv[1]; PRED=sys.argv[2]
if "--seed" in sys.argv:
    _s=int(sys.argv[sys.argv.index("--seed")+1]); np.random.seed(_s); torch.manual_seed(_s)
ANC="/home/frapercan/Thesis2/worktrees/protea-deploy/artifacts/anc2vec/anc2vec_2020-10.npz"
d=np.load(DATA,allow_pickle=True)
Xtr=torch.tensor(d["Xtr"],dtype=torch.float32); rows,cols=d["rows"],d["cols"]
vocab=[str(x) for x in d["vocab"]]; V=len(vocab); N=Xtr.shape[0]
Y=csr_matrix((np.ones(len(rows),np.float32),(rows,cols)),shape=(N,V))
Xev=torch.tensor(d["Xev"],dtype=torch.float32); ev_acc=[str(x) for x in d["ev_acc"]]
dev="cuda" if torch.cuda.is_available() else "cpu"
mu=Xtr.mean(0,keepdim=True); sd=Xtr.std(0,keepdim=True)+1e-6
Xn=(Xtr-mu)/sd; Xevn=(Xev-mu)/sd
# anc2vec label matrix for vocab (L2-normalized; zero for missing)
a=np.load(ANC,allow_pickle=True); aid={str(g):i for i,g in enumerate(a["go_ids"])}; AE=a["embeddings"]
L=np.zeros((V,AE.shape[1]),np.float32)
miss=0
for i,t in enumerate(vocab):
    j=aid.get(t)
    if j is not None: L[i]=AE[j]
    else: miss+=1
L=L/ (np.linalg.norm(L,axis=1,keepdims=True)+1e-8)
Lt=torch.tensor(L,dtype=torch.float32,device=dev)
print(f"N={N} V={V} dim={Xtr.shape[1]} anc2vec_dim={AE.shape[1]} miss={miss}",flush=True)

def asl(lg,t,gn=4.0,gp=1.0,clip=0.05,eps=1e-8):
    p=torch.sigmoid(lg); pm=(p-clip).clamp(min=0)
    return -((t*torch.log(p.clamp(min=eps))*(1-p)**gp)+((1-t)*torch.log((1-pm).clamp(min=eps))*(pm**gn))).mean()

class Hybrid(nn.Module):
    def __init__(s,di,h,V,ld):
        super().__init__()
        s.trunk=nn.Sequential(nn.Linear(di,h),nn.LayerNorm(h),nn.GELU(),nn.Dropout(0.2),
                              nn.Linear(h,h),nn.LayerNorm(h),nn.GELU(),nn.Dropout(0.2))
        s.indep=nn.Linear(h,V)
        s.proj=nn.Linear(h,ld)        # protein rep in label space
        s.scale=nn.Parameter(torch.tensor(1.0))
    def forward(s,x,Lt):
        h=s.trunk(x)
        return s.indep(h) + s.scale*(s.proj(h) @ Lt.t())

model=Hybrid(Xtr.shape[1],1024,V,AE.shape[1]).to(dev)
opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-5)
idx=np.arange(N); BS=512
for ep in range(30):
    model.train(); np.random.shuffle(idx)
    for i in range(0,N,BS):
        b=idx[i:i+BS]; xb=Xn[b].to(dev)
        yb=torch.tensor(Y[b].toarray(),dtype=torch.float32,device=dev)
        opt.zero_grad(); loss=asl(model(xb,Lt),yb); loss.backward(); opt.step()
    if ep%10==0 or ep==29: print(f"epoch {ep} loss {loss.item():.4f} scale {model.scale.item():.3f}",flush=True)
model.eval()
with torch.no_grad(), open(PRED,"w") as w:
    n=0
    for i in range(0,Xevn.shape[0],512):
        xb=Xevn[i:i+512].to(dev); sc=torch.sigmoid(model(xb,Lt)).cpu().numpy()
        for r in range(sc.shape[0]):
            acc=ev_acc[i+r]
            for j in np.argsort(-sc[r])[:100]:
                if sc[r,j]>=0.01: w.write(f"{acc}\t{vocab[j]}\t{sc[r,j]:.6f}\n"); n+=1
print(f"wrote {n} -> {PRED}",flush=True)
