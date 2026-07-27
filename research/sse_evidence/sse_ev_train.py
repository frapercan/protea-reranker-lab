"""Train evidence-arm SSE (K=1) per aspect. ARM in {full,noiea,exp}. Fixed vocab = base SSE meta terms
and M_T (so arms are directly comparable; only the positive-label set changes = clean de-circularisation
ablation). Same model/axioms/geometry as storage/sse_full/sse_full_train.py. ESM2-3B mean input; rows =
base eligibility (eval held out). Saves models/{arm}_{asp}_0.th and {arm}_{asp}_meta.npz.
Run under PROTEA/.venv (torch+CUDA).  usage: sse_ev_train.py <arm> [asp1,asp2,...]
"""
import sys, json, time, collections
from pathlib import Path
import numpy as np, scipy.sparse as sp
import torch as th
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score

ARM = sys.argv[1]
ASPS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["mfo", "bpo", "cco"]
t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.0f}s] [{ARM}] {m}", flush=True)
ROOT = Path("/home/frapercan/Thesis2")
GF = ROOT / "storage/cooc_experiment/generator_frames"
OBO = ROOT / "protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
NORM = ROOT / "storage/deepgose_kwta/data/go_2025-07-22.norm"
BASE = ROOT / "storage/sse_full"
OUT = ROOT / "storage/sse_evidence"; LAB = OUT / "labels"
MDIR = OUT / "models"; MDIR.mkdir(exist_ok=True, parents=True)
DEV = "cuda:0"
N, PK, PM, TEMP = 2048, 128, 64, 0.3
EPOCHS = 14; BATCH = 512; POS_W = 20.0; MARGIN = 0.3; BETA_MARG = 1.0
GAMMA_AX = 0.8; HARDNEG = 20; AX_BATCH = 4096; LR = 1e-3
ASPECTS = {"mfo": "molecular_function", "bpo": "biological_process", "cco": "cellular_component"}

par = collections.defaultdict(set); ns = {}; alt = {}; cur = None
for line in open(OBO):
    line = line.rstrip("\n")
    if line == "[Term]": cur = None
    elif line.startswith("id: GO:"): cur = line[4:]
    elif cur and line.startswith("namespace: "): ns[cur] = line[11:]
    elif cur and line.startswith("is_a: GO:"): par[cur].add(line[6:].split(" ! ")[0].strip())
    elif cur and line.startswith("relationship: part_of GO:"):
        par[cur].add(line[len("relationship: part_of "):].split(" ! ")[0].strip())
    elif cur and line.startswith("alt_id: GO:"): alt[line[8:]] = cur

EMB = np.load(GF / "raw_esm2_3b.npy", mmap_mode="r")

def soft_kwta_scalar(a, k):
    thr = a.topk(k, dim=1).values[:, -1:].detach(); return th.sigmoid((a - thr) / TEMP)
def hard_kwta_scalar(a, k):
    idx = a.topk(k, dim=1).indices; m = th.zeros_like(a); m.scatter_(1, idx, 1.0); return m
def _thr_perrow(a, ks):
    srt = a.sort(dim=1, descending=True).values
    gi = (ks - 1).clamp(0, a.shape[1] - 1).unsqueeze(1); return srt.gather(1, gi).detach()
def soft_kwta_perrow(a, ks): return th.sigmoid((a - _thr_perrow(a, ks)) / TEMP)
def hard_kwta_perrow(a, ks): return (a >= _thr_perrow(a, ks)).float()

class SSE(nn.Module):
    def __init__(self, in_dim, n_terms, n_rels, MTt):
        super().__init__(); self.MTt = MTt
        self.prot = nn.Sequential(nn.Linear(in_dim, 1024), nn.LayerNorm(1024), nn.GELU(),
                                  nn.Dropout(0.1), nn.Linear(1024, N))
        self.pln = nn.LayerNorm(N)
        self.term = nn.Embedding(n_terms, N); nn.init.normal_(self.term.weight, std=0.02)
        self.tln = nn.LayerNorm(N)
        self.rel = nn.Parameter(th.zeros(n_rels, N))
    def prot_gate(self, x, hard=False):
        a = self.pln(self.prot(x)); return hard_kwta_scalar(a, PK) if hard else soft_kwta_scalar(a, PK)
    def term_gate(self, hard=False):
        a = self.tln(self.term.weight); return hard_kwta_perrow(a, self.MTt) if hard else soft_kwta_perrow(a, self.MTt)
    def coverage(self, gp, gt): return (gp @ gt.t()) / (gt.sum(1) + 1e-6)

RES = {"arm": ARM, "aspects": {}}
for asp in ASPS:
    nsname = ASPECTS[asp]
    meta = np.load(BASE / f"{asp}_meta.npz", allow_pickle=True)
    terms = [alt.get(g, g) for g in meta["terms"].tolist()]
    term_idx = {g: j for j, g in enumerate(terms)}; n_terms = len(terms)
    M_T = meta["M_T"]
    sub = np.load(LAB / f"{asp}_rows.npy")
    Y = sp.load_npz(LAB / f"{asp}_{ARM}.npz").toarray().astype(np.float32)
    Xsub = np.ascontiguousarray(EMB[sub].astype(np.float32))
    mu = Xsub.mean(0); sd = Xsub.std(0) + 1e-6; Xsub = (Xsub - mu) / sd
    log(f"{asp}: rows {len(sub)} terms {n_terms} pos/prot {Y.sum(1).mean():.1f}")

    # axioms restricted to base terms
    relations = {}
    def ridx(r):
        if r not in relations: relations[r] = len(relations)
        return relations[r]
    nf1 = []; nf4 = []
    for line in open(NORM):
        s = line.strip().replace("_", ":")
        if " SubClassOf " not in s: continue
        left, right = s.split(" SubClassOf ")
        if len(left) == 10 and len(right) == 10:
            C = alt.get(left, left); D = alt.get(right, right)
            if C in term_idx and D in term_idx and C != D: nf1.append((term_idx[C], term_idx[D]))
        elif " some " in right:
            r, d = right.split(" some ")
            C = alt.get(left, left); D = alt.get(d, d)
            if C in term_idx and D in term_idx: nf4.append((term_idx[C], ridx(r), term_idx[D]))
    n_rels = max(1, len(relations))
    NF1 = th.LongTensor(nf1).to(DEV) if nf1 else th.zeros((0, 2), dtype=th.long, device=DEV)
    NF4 = th.LongTensor(nf4).to(DEV) if nf4 else th.zeros((0, 3), dtype=th.long, device=DEV)
    MTt = th.tensor(M_T, device=DEV)

    th.manual_seed(0); np.random.seed(0)
    net = SSE(Xsub.shape[1], n_terms, n_rels, MTt).to(DEV)
    opt = th.optim.Adam(net.parameters(), lr=LR)
    idx = np.arange(len(sub)); np.random.default_rng(0).shuffle(idx)
    nval = int(0.1 * len(idx)); vi = idx[:nval]; ti = idx[nval:]
    Xt = th.tensor(Xsub); Yt = th.tensor(Y)
    Xv = Xt[vi].to(DEV); Yv = Yt[vi]
    best_auc = 0.0; best_state = None
    for ep in range(EPOCHS):
        net.train(); order = np.random.permutation(ti); tot = 0.0; nb = 0
        for s in range(0, len(order), BATCH):
            b = order[s:s + BATCH]
            xb = Xt[b].to(DEV); yb = Yt[b].to(DEV)
            gp = net.prot_gate(xb); gt = net.term_gate()
            S = net.coverage(gp, gt).clamp(1e-6, 1 - 1e-6)
            w = th.where(yb > 0, POS_W, 1.0)
            loss_bce = F.binary_cross_entropy(S, yb, weight=w)
            neg = S.masked_fill(yb > 0, -1.0)
            hn = neg.topk(min(HARDNEG, S.shape[1]), dim=1).values
            pos_mean = (S * yb).sum(1) / (yb.sum(1) + 1e-6)
            loss_marg = th.relu(MARGIN - pos_mean.unsqueeze(1) + hn).mean()
            ax = th.zeros((), device=DEV)
            if len(NF1):
                sel = NF1[th.randint(0, len(NF1), (min(AX_BATCH, len(NF1)),), device=DEV)]
                gC = gt[sel[:, 0]]; gD = gt[sel[:, 1]].detach(); ax = ax + th.relu(gD - gC).sum(1).mean() / PM
            if len(NF4):
                sel = NF4[th.randint(0, len(NF4), (min(AX_BATCH, len(NF4)),), device=DEV)]
                gC = gt[sel[:, 0]]; mask = th.sigmoid(net.rel[sel[:, 1]]); gD = gt[sel[:, 2]].detach()
                ax = ax + th.relu(gD * mask - gC).sum(1).mean() / PM
            loss = loss_bce + BETA_MARG * loss_marg + GAMMA_AX * ax
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item(); nb += 1
        net.eval()
        with th.no_grad():
            gt = net.term_gate(); preds = []
            for s in range(0, len(Xv), 1024):
                gp = net.prot_gate(Xv[s:s + 1024]); preds.append(net.coverage(gp, gt).cpu().numpy())
            P = np.concatenate(preds).ravel(); Ytrue = Yv.numpy().ravel()
            pos = np.where(Ytrue > 0)[0]; ne = np.where(Ytrue == 0)[0]
            if len(pos) == 0: auc = 0.5
            else:
                negs = np.random.default_rng(0).choice(ne, min(len(pos) * 20, len(ne)), replace=False)
                selq = np.concatenate([pos, negs]); auc = roc_auc_score(Ytrue[selq], P[selq])
        if auc > best_auc: best_auc = auc; best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        log(f"  {asp} ep {ep}: loss {tot/nb:.4f} val_auc {auc:.4f} (best {best_auc:.4f})")
    th.save(best_state, MDIR / f"{ARM}_{asp}_0.th")
    np.savez_compressed(MDIR / f"{ARM}_{asp}_meta.npz", terms=np.array(terms), M_T=M_T, mu=mu, sd=sd,
                        n_rels=np.array([n_rels]))
    RES["aspects"][asp] = {"n_rows": int(len(sub)), "n_terms": n_terms, "pos_per_prot": float(Y.sum(1).mean()),
                           "nf1": len(nf1), "nf4": len(nf4), "val_auc": float(best_auc)}
    log(f"{asp}: best_val_auc {best_auc:.4f}")
    json.dump(RES, open(OUT / f"train_{ARM}.json", "w"), indent=1)

log(f"SSE EV TRAIN DONE arm={ARM}")
print(json.dumps(RES, indent=1))
