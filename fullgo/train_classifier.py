"""Full-label classifier with ASL (asymmetric loss) + optional per-protein calibration.
Usage: m0_train_asl.py DATA PRED [--cal]
"""
import sys
import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import csr_matrix

DATA = sys.argv[1] if len(sys.argv) > 1 else "/tmp/m0v3_data.npz"
PRED = sys.argv[2] if len(sys.argv) > 2 else "/tmp/m0_asl_pred.tsv"
CAL = "--cal" in sys.argv

d = np.load(DATA, allow_pickle=True)
Xtr = torch.tensor(d["Xtr"], dtype=torch.float32)
rows, cols = d["rows"], d["cols"]
vocab = [str(x) for x in d["vocab"]]
V = len(vocab); N = Xtr.shape[0]
Y = csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(N, V))
Xev = torch.tensor(d["Xev"], dtype=torch.float32)
ev_acc = [str(x) for x in d["ev_acc"]]
dev = "cuda" if torch.cuda.is_available() else "cpu"
mu = Xtr.mean(0, keepdim=True); sd = Xtr.std(0, keepdim=True) + 1e-6
Xtr = (Xtr - mu) / sd; Xevn = (Xev - mu) / sd
print(f"N={N} V={V} dim={Xtr.shape[1]} cal={CAL}", flush=True)


def asl_loss(logits, targets, gamma_neg=4.0, gamma_pos=1.0, clip=0.05, eps=1e-8):
    p = torch.sigmoid(logits)
    pm = (p - clip).clamp(min=0)  # shifted prob for negatives
    los_pos = targets * torch.log(p.clamp(min=eps)) * (1 - p) ** gamma_pos
    los_neg = (1 - targets) * torch.log((1 - pm).clamp(min=eps)) * (pm ** gamma_neg)
    return -(los_pos + los_neg).mean()


class MLP(nn.Module):
    def __init__(s, di, h, o):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(di, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(0.2),
                              nn.Linear(h, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(0.2),
                              nn.Linear(h, o))
    def forward(s, x): return s.net(x)


model = MLP(Xtr.shape[1], 1024, V).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
idx = np.arange(N); BS = 512; EPOCHS = 30
for ep in range(EPOCHS):
    model.train(); np.random.shuffle(idx); tot = 0.0
    for i in range(0, N, BS):
        b = idx[i:i + BS]
        xb = Xtr[b].to(dev)
        yb = torch.tensor(Y[b].toarray(), dtype=torch.float32, device=dev)
        opt.zero_grad(); out = model(xb); loss = asl_loss(out, yb)
        loss.backward(); opt.step(); tot += loss.item() * len(b)
    if ep % 5 == 0 or ep == EPOCHS - 1:
        print(f"epoch {ep} loss {tot/N:.4f}", flush=True)

model.eval()
with torch.no_grad(), open(PRED, "w") as w:
    n_out = 0
    for i in range(0, Xevn.shape[0], 512):
        xb = Xevn[i:i + 512].to(dev)
        sc = torch.sigmoid(model(xb)).cpu().numpy()
        for r in range(sc.shape[0]):
            row = sc[r]
            if CAL:  # per-protein min-max calibration
                lo, hi = row.min(), row.max()
                row = (row - lo) / (hi - lo + 1e-8)
            acc = ev_acc[i + r]
            top = np.argsort(-row)[:120]
            for j in top:
                if row[j] >= 0.01:
                    w.write(f"{acc}\t{vocab[j]}\t{row[j]:.6f}\n"); n_out += 1
print(f"wrote {n_out} -> {PRED}", flush=True)
