"""ProtEx verification -- STAGE 3: train the exemplar-conditioned verifier + score the blind pool.

Model input per (protein, candidate-term):
    [ p_emb(1024) , term_feat(anc2vec 200 + IA 1) , POS-pool(1024) , NEG-pool(1024) , contrast(5) ]
POS/NEG pools are mean ProtT5 embeddings of the K positive / K hard-negative exemplars (deep-sets =
permutation-invariant set encoder). The POS-vs-NEG contrast is the load-bearing ProtEx signal.

Two models are trained under the IDENTICAL temporal gate:
    with_neg : the full ProtEx verifier
    no_neg   : ablation -- NEG pool + NEG contrast removed (isolates negative-exemplar conditioning)

Temporal gate: train on snapshot pairs <= v225, early-stop on v225-v227 (pooled true-vs-false AUC),
then score the BLIND v227-v230 pool. cafa_eval is not called here; this stage only emits per-row
scores for the evaluator.
"""
import json, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn

t0 = time.time()
def log(m): print(f"[{time.time()-t0:.0f}s] {m}", flush=True)
W = Path("/home/frapercan/Thesis2/storage/protex")
T0D = Path("/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025")
AZ = "/home/frapercan/Thesis2/repositories/PROTEA/artifacts/anc2vec/anc2vec_2020-10.npz"
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0); np.random.seed(0)
K = 8

# ---- ontology alt + IA + anc2vec, build candidate-term feature matrix ----
alt = {}; cur = None
for line in open(str(T0D / "go-basic.obo")):
    line = line.rstrip("\n")
    if line == "[Term]": cur = None
    elif line.startswith("id: GO:"): cur = line[4:]
    elif cur and line.startswith("alt_id: GO:"): alt[line[8:]] = cur
IA = {}
for line in open(str(T0D / "IA.tsv")):
    p = line.rstrip("\n").split("\t")
    if len(p) >= 2:
        try: IA[p[0]] = float(p[1])
        except ValueError: pass
az = np.load(AZ, allow_pickle=True)
AV = az["embeddings"].astype(np.float32); apos = {alt.get(g, g): i for i, g in enumerate(az["go_ids"].tolist())}
cand_terms = json.load(open(W / "cand_terms.json"))
TF = np.zeros((len(cand_terms), AV.shape[1] + 1), np.float32)
for i, g in enumerate(cand_terms):
    ai = apos.get(g)
    if ai is not None: TF[i, :AV.shape[1]] = AV[ai]
    TF[i, -1] = IA.get(g, 0.0)
TFg = torch.from_numpy(TF).to(dev)
TDIM = TF.shape[1]
log(f"term features {TF.shape}")

# ---- resident reference matrix (fp16, GPU) ----
E = torch.from_numpy(np.load(W / "ref_emb_norm_fp16.npy")).to(dev)   # (R,1024) fp16 normalised
DE = E.shape[1]
log(f"ref matrix on {dev}: {tuple(E.shape)}")

def load_split(name):
    z = np.load(W / f"exemplars_{name}.npz", allow_pickle=True)
    return {k: z[k] for k in z.files}
TR = load_split("train"); VA = load_split("val")

def pools(d, idx, dev_=dev):
    """Gather p_emb, pos_pool, neg_pool, term_feat, contrast for rows `idx` of split dict d."""
    pr = torch.from_numpy(d["p_ref"][idx].astype(np.int64)).to(dev_)
    pe = E[pr.clamp(min=0)].float()
    pe[pr < 0] = 0.0
    def pool(ref):
        r = torch.from_numpy(ref[idx].astype(np.int64)).to(dev_)         # (b,K)
        m = (r >= 0).float().unsqueeze(-1)
        g = E[r.clamp(min=0)].float() * m
        s = g.sum(1); c = m.sum(1).clamp(min=1)
        return s / c
    pos = pool(d["pos_ref"]); neg = pool(d["neg_ref"])
    tf = TFg[torch.from_numpy(d["term_ci"][idx].astype(np.int64)).clamp(min=0).to(dev_)]
    ps = torch.from_numpy(d["pos_sim"][idx].astype(np.float32)).to(dev_)
    nsim = torch.from_numpy(d["neg_sim"][idx].astype(np.float32)).to(dev_)
    npos = torch.from_numpy(d["n_pos"][idx].astype(np.float32)).to(dev_) / K
    nneg = torch.from_numpy(d["n_neg"][idx].astype(np.float32)).to(dev_) / K
    contrast = torch.stack([ps, nsim, ps - nsim, npos, nneg], 1)
    return pe, tf, pos, neg, contrast

def build_x(pe, tf, pos, neg, contrast, use_neg):
    if use_neg:
        return torch.cat([pe, tf, pos, neg, contrast], 1)
    return torch.cat([pe, tf, pos, contrast[:, [0, 3]]], 1)   # keep pos_sim + n_pos only

class Verifier(nn.Module):
    def __init__(self, din):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(din), nn.Linear(din, 1024), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(1024, 512), nn.GELU(), nn.Dropout(0.3), nn.Linear(512, 1))
    def forward(self, x): return self.net(x).squeeze(-1)

def auc(y, s):
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(len(s))
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0: return float("nan")
    return float((r[y == 1].sum() - n1 * (n1 - 1) / 2) / (n1 * n0))

def train_variant(use_neg):
    tag = "with_neg" if use_neg else "no_neg"
    din = DE + TDIM + (2 * DE + 5 if use_neg else DE + 2)
    net = Verifier(din).to(dev)
    yb = TR["label"].astype(np.float32)
    pw = float((yb == 0).sum() / max(1, (yb > 0).sum()))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw, device=dev))
    ntr = len(yb); B = 8192
    vidx = np.arange(len(VA["label"])); vy = VA["label"].astype(np.float32)
    best = (-1, None, -1); patience = 0
    log(f"[{tag}] din={din} pos_weight={pw:.2f} ntr={ntr:,}")
    for ep in range(40):
        net.train(); perm = np.random.permutation(ntr)
        for s in range(0, ntr, B):
            idx = perm[s:s + B]
            pe, tf, pos, neg, cn = pools(TR, idx)
            x = build_x(pe, tf, pos, neg, cn, use_neg)
            y = torch.from_numpy(yb[idx]).to(dev)
            opt.zero_grad(); l = lossf(net(x), y); l.backward(); opt.step()
        # val AUC
        net.eval(); vs = np.zeros(len(vidx), np.float32)
        with torch.no_grad():
            for s in range(0, len(vidx), B):
                idx = vidx[s:s + B]
                pe, tf, pos, neg, cn = pools(VA, idx)
                vs[s:s + len(idx)] = torch.sigmoid(net(build_x(pe, tf, pos, neg, cn, use_neg))).cpu().numpy()
        va = auc(vy, vs)
        log(f"[{tag}] epoch {ep} val_auc={va:.4f}")
        if va > best[0]:
            best = (va, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}, ep); patience = 0
        else:
            patience += 1
            if patience >= 6: break
    net.load_state_dict(best[1]); net.eval()
    torch.save({"state_dict": best[1], "din": din, "use_neg": use_neg, "val_auc": best[0], "best_epoch": best[2]},
               W / f"verifier_{tag}.pt")
    log(f"[{tag}] BEST val_auc={best[0]:.4f} @ep{best[2]}")
    return net, use_neg, best[0]

# ---- score the blind test pool with both variants ----
TE = load_split("test"); nte = len(TE["label"]); B = 16384
scores = {}
val_aucs = {}
for use_neg in (True, False):
    net, un, va = train_variant(use_neg)
    tag = "with_neg" if un else "no_neg"; val_aucs[tag] = va
    ss = np.zeros(nte, np.float32)
    with torch.no_grad():
        for s in range(0, nte, B):
            idx = np.arange(s, min(s + B, nte))
            pe, tf, pos, neg, cn = pools(TE, idx)
            ss[s:s + len(idx)] = torch.sigmoid(net(build_x(pe, tf, pos, neg, cn, un))).cpu().numpy()
    scores[tag] = ss
np.savez(W / "test_scores.npz", with_neg=scores["with_neg"], no_neg=scores["no_neg"],
         protein=TE["protein"], term=TE["term"], cell=TE["cell"],
         reranker_score=TE["reranker_score"], n_pos=TE["n_pos"], n_neg=TE["n_neg"])
json.dump(val_aucs, open(W / "train_val_aucs.json", "w"), indent=1)
log(f"DONE train+score. val_aucs={val_aucs}")
