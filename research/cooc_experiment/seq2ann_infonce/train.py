"""Stage 1: genuine contrastive InfoNCE seq<->annotation dual-encoder, two arms.
Frozen pretrained encoders (d8979601 2048-d protein codes; BioBERT 768-d GO-text) + trainable
MLP projection heads + learnable temperature. Full-BP-vocab multi-positive InfoNCE: every rare/deep
term competes as a hard negative every step; positives IA-weighted so the deep tail is not drowned.
Trains arm A (EXP-only) and arm B (EXP+IEA) with identical architecture/schedule.
NO PLM fine-tune. Saves checkpoints, training log, and eval-protein retrieval scores per arm.
"""
import json, time, sys
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

W = Path("/home/frapercan/Thesis2/storage/cooc_experiment/seq2ann_infonce")
SC = Path("/home/frapercan/Thesis2/repositories/protea-reranker-lab/results/sparse_classifier")
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0); np.random.seed(0)
D = 256; EPOCHS = 60; BATCH = 2048; LR = 1e-3; N_VAL = 3000
t0 = time.time()

voc = np.load(W / "vocab.npz", allow_pickle=True)
vocab = voc["vocab"]; vocab_emb = torch.from_numpy(voc["vocab_emb"]).to(dev)     # (Nt,768)
vocab_ia = torch.from_numpy(voc["vocab_ia"]).to(dev)                              # (Nt,)
Nt = len(vocab)
pos = json.load(open(W / "positives.json"))
tc = np.load(W / "train_codes.npz", allow_pickle=True)
code_accs = tc["accs"].tolist(); code_mat = torch.from_numpy(tc["codes"].astype(np.float32)).to(dev)
acc2crow = {a: i for i, a in enumerate(code_accs)}
print(f"[{time.time()-t0:.0f}s] vocab {Nt:,}, train-code prots {len(code_accs):,} on {dev}", flush=True)

# fixed val protein slice (shared across arms): sample from arm B proteins
rng = np.random.default_rng(42)
allB = sorted(pos["armB"].keys())
val_set = set(rng.choice(allB, size=min(N_VAL, len(allB)), replace=False).tolist())

class Tower(nn.Module):
    def __init__(self, din):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, 1024), nn.GELU(), nn.Dropout(0.1),
                                 nn.Linear(1024, D))
    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)

def build_arm(arm_key):
    prots = [p for p in pos[arm_key] if p in acc2crow and p not in val_set]
    rows = torch.tensor([acc2crow[p] for p in prots])
    # ragged positives -> lists
    plist = [torch.tensor(pos[arm_key][p], dtype=torch.long) for p in prots]
    return prots, rows, plist

def val_recall(seq_tower, ann_tower, arm_key, ks=(5, 20, 50)):
    seq_tower.eval(); ann_tower.eval()
    with torch.no_grad():
        T = ann_tower(vocab_emb)                                   # (Nt,D)
        vp = [p for p in val_set if p in acc2crow and p in pos[arm_key]]
        if not vp: return {}
        X = code_mat[torch.tensor([acc2crow[p] for p in vp])]
        S = seq_tower(X)                                           # (V,D)
        sim = S @ T.t()                                            # (V,Nt)
        topk = torch.topk(sim, max(ks), dim=1).indices.cpu().numpy()
        rec = {k: [] for k in ks}; recw = {k: [] for k in ks}
        ia = vocab_ia.cpu().numpy()
        for i, p in enumerate(vp):
            tp = set(pos[arm_key][p]); denom = len(tp)
            wtot = sum(ia[j] for j in tp) or 1.0
            for k in ks:
                hit = [j for j in topk[i, :k] if j in tp]
                rec[k].append(len(hit) / denom)
                recw[k].append(sum(ia[j] for j in hit) / wtot)
        return {"recall@k": {k: round(float(np.mean(rec[k])), 4) for k in ks},
                "ia_wrecall@k": {k: round(float(np.mean(recw[k])), 4) for k in ks}}

def train_arm(arm_key):
    prots, rows, plist = build_arm(arm_key)
    rows = rows.to(dev)
    N = len(prots)
    seq_tower = Tower(2048).to(dev); ann_tower = Tower(768).to(dev)
    logit_scale = nn.Parameter(torch.tensor(np.log(1 / 0.07), dtype=torch.float32, device=dev))
    opt = torch.optim.AdamW(list(seq_tower.parameters()) + list(ann_tower.parameters()) + [logit_scale], lr=LR, weight_decay=1e-4)
    log = []
    print(f"\n=== ARM {arm_key}: {N:,} train prots ===", flush=True)
    for ep in range(EPOCHS):
        seq_tower.train(); ann_tower.train()
        perm = torch.randperm(N)
        ep_loss = 0.0; nb = 0
        for s in range(0, N, BATCH):
            bidx = perm[s:s + BATCH]
            X = code_mat[rows[bidx]]                               # (B,2048)
            S = seq_tower(X)                                       # (B,D)
            T = ann_tower(vocab_emb)                               # (Nt,D)
            ls = logit_scale.clamp(max=np.log(100.0)).exp()
            logits = ls * (S @ T.t())                             # (B,Nt)
            lse = torch.logsumexp(logits, dim=1)                  # (B,)
            # flatten positives of this batch
            bi = []; ti = []; wi = []
            for local, gi in enumerate(bidx.tolist()):
                pl = plist[gi]
                if len(pl) == 0: continue
                w = 1.0 + vocab_ia[pl.to(dev)]                    # IA-weight; deep tail dominates
                w = w / w.sum()
                bi.append(torch.full((len(pl),), local, dtype=torch.long))
                ti.append(pl); wi.append(w.cpu())
            if not bi: continue
            bi = torch.cat(bi).to(dev); ti = torch.cat(ti).to(dev); wi = torch.cat(wi).to(dev)
            pos_logit = logits[bi, ti]
            perpair = wi * (lse[bi] - pos_logit)                 # weighted CE per positive
            loss = perpair.sum() / bidx.numel()                  # mean over batch proteins
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += float(loss); nb += 1
        if ep % 5 == 0 or ep == EPOCHS - 1:
            vr = val_recall(seq_tower, ann_tower, arm_key)
            rec = {"epoch": ep, "loss": round(ep_loss / max(nb, 1), 4),
                   "logit_scale": round(float(ls), 2), **vr}
            log.append(rec); print(f"  {rec}", flush=True)
    torch.save({"seq": seq_tower.state_dict(), "ann": ann_tower.state_dict(),
                "logit_scale": logit_scale.detach().cpu()}, W / f"ckpt_{arm_key}.pt")
    json.dump(log, open(W / f"trainlog_{arm_key}.json", "w"), indent=1)
    return seq_tower, ann_tower

def export_eval(arm_key, seq_tower, ann_tower, topk=500):
    seq_tower.eval(); ann_tower.eval()
    ev = np.load(SC / "eval_protein_codes.npz", allow_pickle=True)
    accs = ev["accs"]; codes = torch.from_numpy(ev["codes"].astype(np.float32)).to(dev)
    with torch.no_grad():
        T = ann_tower(vocab_emb)                                  # (Nt,D)
        out_p = []; out_t = []; out_s = []
        for s in range(0, len(accs), 1024):
            S = seq_tower(codes[s:s + 1024])
            sim = (S @ T.t())                                     # cosine (normalized)
            vals, idx = torch.topk(sim, topk, dim=1)
            vals = vals.cpu().numpy(); idx = idx.cpu().numpy()
            for bi in range(idx.shape[0]):
                a = accs[s + bi]
                out_p.append(np.full(topk, a)); out_t.append(vocab[idx[bi]]); out_s.append(vals[bi])
    np.savez(W / f"eval_scores_{arm_key}.npz",
             prot=np.concatenate(out_p), term=np.concatenate(out_t), score=np.concatenate(out_s).astype(np.float32))
    print(f"[{time.time()-t0:.0f}s] exported eval scores {arm_key}: {len(accs)} prots x {topk}", flush=True)

for arm in ["armA", "armB"]:
    st, at = train_arm(arm)
    export_eval(arm, st, at)
print(f"[{time.time()-t0:.0f}s] ALL DONE", flush=True)
