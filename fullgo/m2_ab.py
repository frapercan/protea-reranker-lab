"""anc2vec-vs-sparse M2 head-to-head, all in memory (no fragile 2.9GB save).

Loads the base Ankh-base frame (/tmp/m0_data.npz), pulls the 5 extra PLMs from
the DB to build the 8320-d frame in RAM, then trains the M2 hybrid twice (label
basis = anc2vec-2020 vs sparse functional codes) and writes only the small
prediction TSVs. Usage: m2_ab.py OUTDIR [--seed N]
"""
import sys, os, numpy as np, psycopg, torch, torch.nn as nn
from scipy.sparse import csr_matrix

OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
SEED = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 0
DB = "postgresql://protea:protea@localhost:5432/protea"
ORDER = [("55e43f1c-1a3b-4b1d-88c0-26b433f5f673", 2560), ("238f79b1-3068-4c6f-9013-5cc52b4f662b", 1536),
         ("c2e9dda3-e505-4170-b50d-435a451761ac", 1280), ("2bf1e753-022f-44b8-a131-9a90acb4024e", 1152),
         ("084943c6-fec1-441d-bdc5-63b0268ada1b", 1024)]
BASES = {"anc2vec": ("/home/frapercan/Thesis2/worktrees/protea-deploy/artifacts/anc2vec/anc2vec_2020-10.npz", "go_ids", "embeddings"),
         "sparse": ("/home/frapercan/Thesis2/repositories/protea-reranker-lab/results/sparse_classifier/go_sparse_codes.npz", "go_ids", "codes"),
         "gotext": ("/home/frapercan/Thesis2/repositories/protea-reranker-lab/results/sparse_classifier/go_text_emb.npz", "go_ids", "emb")}
RUN_BASES = sys.argv[sys.argv.index("--bases") + 1].split(",") if "--bases" in sys.argv else ["anc2vec", "sparse"]

d = np.load("/tmp/m0_data.npz", allow_pickle=True)
tr_acc = [str(x) for x in d["tr_acc"]]; ev_acc = [str(x) for x in d["ev_acc"]]
vocab = [str(x) for x in d["vocab"]]; V = len(vocab)
rows, cols = d["rows"], d["cols"]; N = len(tr_acc)
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
            if a not in emb:
                emb[a] = np.fromstring(v.strip("[]"), sep=",", dtype=np.float32)
    Mtr = np.zeros((len(tr_acc), dim), np.float32)
    for i, a in enumerate(tr_acc):
        v = emb.get(a)
        if v is not None and len(v) == dim: Mtr[i] = v
    Mev = np.zeros((len(ev_acc), dim), np.float32)
    for i, a in enumerate(ev_acc):
        v = emb.get(a)
        if v is not None and len(v) == dim: Mev[i] = v
    tr_parts.append(Mtr); ev_parts.append(Mev); del emb
    print(f"pulled {cfg[:8]}", flush=True)
conn.close()
Xtr = torch.tensor(np.hstack(tr_parts), dtype=torch.float32); del tr_parts
Xev = torch.tensor(np.hstack(ev_parts), dtype=torch.float32); del ev_parts
Y = csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(N, V))
dev = "cuda" if torch.cuda.is_available() else "cpu"
mu = Xtr.mean(0, keepdim=True); sd = Xtr.std(0, keepdim=True) + 1e-6
Xn = (Xtr - mu) / sd; Xevn = (Xev - mu) / sd
print(f"frame ready Xtr {tuple(Xtr.shape)} Xev {tuple(Xev.shape)} V={V}", flush=True)


def asl(lg, t, gn=4.0, gp=1.0, clip=0.05, eps=1e-8):
    p = torch.sigmoid(lg); pm = (p - clip).clamp(min=0)
    return -((t*torch.log(p.clamp(min=eps))*(1-p)**gp)+((1-t)*torch.log((1-pm).clamp(min=eps))*(pm**gn))).mean()


class Hybrid(nn.Module):
    def __init__(s, di, h, V, ld):
        super().__init__()
        s.trunk = nn.Sequential(nn.Linear(di, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(0.2),
                                nn.Linear(h, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(0.2))
        s.indep = nn.Linear(h, V); s.proj = nn.Linear(h, ld); s.scale = nn.Parameter(torch.tensor(1.0))
    def forward(s, x, Lt):
        h = s.trunk(x); return s.indep(h) + s.scale*(s.proj(h) @ Lt.t())


def build_L(basis):
    bpath, kid, kemb = BASES[basis]; a = np.load(bpath, allow_pickle=True)
    aid = {str(g): i for i, g in enumerate(a[kid])}; AE = a[kemb]
    L = np.zeros((V, AE.shape[1]), np.float32); miss = 0
    for i, t in enumerate(vocab):
        j = aid.get(t)
        if j is not None: L[i] = AE[j]
        else: miss += 1
    L = L / (np.linalg.norm(L, axis=1, keepdims=True) + 1e-8)
    return torch.tensor(L, dtype=torch.float32, device=dev), AE.shape[1], miss


def train_eval(basis):
    np.random.seed(SEED); torch.manual_seed(SEED)
    Lt, ld, miss = build_L(basis)
    print(f"[{basis}] label_dim={ld} miss={miss}", flush=True)
    model = Hybrid(Xtr.shape[1], 1024, V, ld).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    idx = np.arange(N); BS = 512
    for ep in range(30):
        model.train(); np.random.shuffle(idx)
        for i in range(0, N, BS):
            b = idx[i:i+BS]; xb = Xn[b].to(dev)
            yb = torch.tensor(Y[b].toarray(), dtype=torch.float32, device=dev)
            opt.zero_grad(); loss = asl(model(xb, Lt), yb); loss.backward(); opt.step()
        if ep % 10 == 0 or ep == 29:
            print(f"[{basis}] epoch {ep} loss {loss.item():.4f} scale {model.scale.item():.3f}", flush=True)
    model.eval(); pred = os.path.join(OUT, f"pred_{basis}.tsv"); n = 0
    with torch.no_grad(), open(pred, "w") as w:
        for i in range(0, Xevn.shape[0], 512):
            xb = Xevn[i:i+512].to(dev); sc = torch.sigmoid(model(xb, Lt)).cpu().numpy()
            for r in range(sc.shape[0]):
                acc = ev_acc[i+r]
                for j in np.argsort(-sc[r])[:100]:
                    if sc[r, j] >= 0.01: w.write(f"{acc}\t{vocab[j]}\t{sc[r,j]:.6f}\n"); n += 1
    print(f"[{basis}] wrote {n} -> {pred}", flush=True)


for basis in RUN_BASES:
    train_eval(basis)
print("DONE", flush=True)
