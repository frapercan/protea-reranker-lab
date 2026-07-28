"""Seed-averaged M2 for anc2vec + gotext bases (robustness confirmation).

Pulls the 8320-d frame once, trains each basis over SEEDS, averages the full
sigmoid output across seeds, writes averaged top-100 pred TSVs. Then merge_aspect
+ eval give the seed-averaged aspect-aware 9-cell. Usage: m2_seedavg.py OUTDIR
"""
import sys, os, numpy as np, psycopg, torch, torch.nn as nn
from scipy.sparse import csr_matrix

OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
SEEDS = [0, 7, 23, 91, 137]
DB = "postgresql://protea:protea@localhost:5432/protea"
ORDER = [("55e43f1c-1a3b-4b1d-88c0-26b433f5f673", 2560), ("238f79b1-3068-4c6f-9013-5cc52b4f662b", 1536),
         ("c2e9dda3-e505-4170-b50d-435a451761ac", 1280), ("2bf1e753-022f-44b8-a131-9a90acb4024e", 1152),
         ("084943c6-fec1-441d-bdc5-63b0268ada1b", 1024)]
BASES = {"anc2vec": ("/home/frapercan/Thesis2/worktrees/protea-deploy/artifacts/anc2vec/anc2vec_2020-10.npz", "go_ids", "embeddings"),
         "gotext": ("/home/frapercan/Thesis2/repositories/protea-reranker-lab/results/sparse_classifier/go_text_emb.npz", "go_ids", "emb")}

d = np.load("/tmp/m0_data.npz", allow_pickle=True)
tr_acc = [str(x) for x in d["tr_acc"]]; ev_acc = [str(x) for x in d["ev_acc"]]
vocab = [str(x) for x in d["vocab"]]; V = len(vocab); rows, cols = d["rows"], d["cols"]; N = len(tr_acc)
need = sorted(set(tr_acc) | set(ev_acc))
conn = psycopg.connect(DB); cur = conn.cursor()
tr_parts = [d["Xtr"].astype(np.float32)]; ev_parts = [d["Xev"].astype(np.float32)]
for cfg, dim in ORDER:
    emb = {}; B = 5000
    for i in range(0, len(need), B):
        ch = need[i:i+B]
        cur.execute("select p.accession,e.embedding::text from protein p join sequence s on s.id=p.sequence_id "
                    "join sequence_embedding e on e.sequence_id=s.id where e.embedding_config_id=%s and p.accession=any(%s)", (cfg, ch))
        for a, v in cur:
            if a not in emb: emb[a] = np.fromstring(v.strip("[]"), sep=",", dtype=np.float32)
    Mtr = np.zeros((len(tr_acc), dim), np.float32)
    for i, a in enumerate(tr_acc):
        v = emb.get(a);  Mtr[i] = v if (v is not None and len(v) == dim) else 0
    Mev = np.zeros((len(ev_acc), dim), np.float32)
    for i, a in enumerate(ev_acc):
        v = emb.get(a);  Mev[i] = v if (v is not None and len(v) == dim) else 0
    tr_parts.append(Mtr); ev_parts.append(Mev); del emb
    print(f"pulled {cfg[:8]}", flush=True)
conn.close()
Xtr = torch.tensor(np.hstack(tr_parts), dtype=torch.float32); del tr_parts
Xev = torch.tensor(np.hstack(ev_parts), dtype=torch.float32); del ev_parts
Y = csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(N, V))
dev = "cuda" if torch.cuda.is_available() else "cpu"
mu = Xtr.mean(0, keepdim=True); sd = Xtr.std(0, keepdim=True) + 1e-6
Xn = (Xtr - mu) / sd; Xevn = (Xev - mu) / sd
print(f"frame ready {tuple(Xtr.shape)} V={V}", flush=True)


def asl(lg, t, gn=4.0, gp=1.0, clip=0.05, eps=1e-8):
    p = torch.sigmoid(lg); pm = (p - clip).clamp(min=0)
    return -((t*torch.log(p.clamp(min=eps))*(1-p)**gp)+((1-t)*torch.log((1-pm).clamp(min=eps))*(pm**gn))).mean()


class Hybrid(nn.Module):
    def __init__(s, di, h, V, ld):
        super().__init__()
        s.trunk = nn.Sequential(nn.Linear(di, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(0.2),
                                nn.Linear(h, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(0.2))
        s.indep = nn.Linear(h, V); s.proj = nn.Linear(h, ld); s.scale = nn.Parameter(torch.tensor(1.0))
    def forward(s, x, Lt): h = s.trunk(x); return s.indep(h) + s.scale*(s.proj(h) @ Lt.t())


def build_L(basis):
    bpath, kid, kemb = BASES[basis]; a = np.load(bpath, allow_pickle=True)
    aid = {str(g): i for i, g in enumerate(a[kid])}; AE = a[kemb]
    L = np.zeros((V, AE.shape[1]), np.float32)
    for i, t in enumerate(vocab):
        j = aid.get(t)
        if j is not None: L[i] = AE[j]
    L = L / (np.linalg.norm(L, axis=1, keepdims=True) + 1e-8)
    return torch.tensor(L, dtype=torch.float32, device=dev), AE.shape[1]


for basis in ("anc2vec", "gotext"):
    Lt, ld = build_L(basis)
    acc = np.zeros((Xevn.shape[0], V), np.float32)
    for sd_ in SEEDS:
        np.random.seed(sd_); torch.manual_seed(sd_)
        model = Hybrid(Xtr.shape[1], 1024, V, ld).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
        idx = np.arange(N); BS = 512
        for ep in range(30):
            model.train(); np.random.shuffle(idx)
            for i in range(0, N, BS):
                b = idx[i:i+BS]; xb = Xn[b].to(dev)
                yb = torch.tensor(Y[b].toarray(), dtype=torch.float32, device=dev)
                opt.zero_grad(); asl(model(xb, Lt), yb).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            for i in range(0, Xevn.shape[0], 512):
                xb = Xevn[i:i+512].to(dev)
                acc[i:i+512] += torch.sigmoid(model(xb, Lt)).cpu().numpy()
        print(f"[{basis}] seed {sd_} done", flush=True)
    acc /= len(SEEDS)
    pred = os.path.join(OUT, f"pred_{basis}_seedavg.tsv"); n = 0
    with open(pred, "w") as w:
        for r in range(acc.shape[0]):
            a = ev_acc[r]
            for j in np.argsort(-acc[r])[:100]:
                if acc[r, j] >= 0.01: w.write(f"{a}\t{vocab[j]}\t{acc[r,j]:.6f}\n"); n += 1
    print(f"[{basis}] seedavg wrote {n} -> {pred}", flush=True)
print("DONE", flush=True)
