"""Phase 1: LEARN the k-WTA GO ontology encoder end-to-end with SEPARABILITY hard negatives.

Distinct from the two-tower (fixed SVD->kWTA recipe, no learning, no hard negatives) and from
the annotation-RAG (dense, no k-WTA). Here an MLP encoder maps term-feature blocks to a k-WTA
sparse GO code (top-k active, STE gradient), trained so:
  POSITIVES      co-occurring term pairs (share proteins in the t0 corpus)  -> OVERLAP
  HARD NEGATIVES text/ontology-close but NEVER co-annotated                 -> DISJOINT
This directly optimizes the true/false-at-tail distinction that is the BP wall.

Ablation over term-feature blocks: coann-only / coann+struct / coann+struct+text.
Deterministic (seed 0). Saves model + code matrix per variant + intrinsic-quality report.
GPU, but small. Read-only w.r.t. repos; writes only under storage/kwta_go_encoder/.
"""
import json, time, sys
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

t0 = time.time()
def log(m): print(f"[{time.time()-t0:.0f}s] {m}", flush=True)

OUT = Path("/home/frapercan/Thesis2/storage/kwta_go_encoder")
dev = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0
D_CODE, K_WTA = 2048, 128           # mirror the protein encoder (~128 active / 2048)
EPOCHS, BATCH, LR = 40, 512, 1e-3
STEPS_PER_EPOCH = 400
N_HARD, N_RAND = 8, 8               # hard + random negatives per anchor (plus in-batch)
TEMP = 0.15
MARGIN = 0.2

torch.manual_seed(SEED); np.random.seed(SEED)
tf = np.load(OUT / "term_features.npz", allow_pickle=True)
vocab = tf["vocab"]; N = len(vocab)
blocks = {"coann": tf["coann"], "struct": tf["struct"], "text": tf["text"]}
knn_idx = tf["knn_idx"]
cg = np.load(OUT / "cooc_graph.npz", allow_pickle=True)
indptr, indices, cdata = cg["indptr"], cg["indices"], cg["data"]
# per-term co-occurrence neighbours + sampling prob; and a set for the hard-neg mask
neigh = [indices[indptr[i]:indptr[i+1]] for i in range(N)]
cooc_set = [set(x.tolist()) for x in neigh]
anchors = np.array([i for i in range(N) if len(neigh[i]) > 0], np.int64)
# --- precompute padded matrices for VECTORIZED batch sampling (speed) ---
POS_MAX = 64          # top co-occurring neighbours (by count) per anchor
HARD_MAX = 16         # text-close never-co-annotated candidates per anchor
pos_pad = np.full((N, POS_MAX), -1, np.int64)
pos_len = np.zeros(N, np.int64)
hard_pad = np.full((N, HARD_MAX), -1, np.int64)
hard_len = np.zeros(N, np.int64)
rng0 = np.random.default_rng(SEED)
for i in range(N):
    nb = neigh[i]
    if len(nb):
        d = cdata[indptr[i]:indptr[i+1]]
        top = nb[np.argsort(-d)[:POS_MAX]]
        pos_pad[i, :len(top)] = top; pos_len[i] = len(top)
    hn = [int(x) for x in knn_idx[i] if int(x) not in cooc_set[i] and int(x) != i][:HARD_MAX]
    if hn:
        hard_pad[i, :len(hn)] = hn; hard_len[i] = len(hn)
log(f"vocab {N:,}; anchors with co-occurrence {len(anchors):,}; padded sampling matrices built")

def sample_pos(ai, g):
    ln = pos_len[ai]; col = (g.integers(0, 1 << 30, len(ai)) % np.maximum(ln, 1))
    return pos_pad[ai, col]

def sample_negs(ai, k, g):
    """k text-close hard negs per anchor; fall back to random where the pool is short."""
    ln = np.maximum(hard_len[ai], 1)
    cols = g.integers(0, 1 << 30, (len(ai), k)) % ln[:, None]
    out = hard_pad[ai[:, None], cols]
    bad = out < 0
    if bad.any(): out[bad] = g.integers(0, N, int(bad.sum()))
    return out

# held-out positive pairs for early-stop / intrinsic AUC (leakage-safe: relation split, all t0)
rng = np.random.default_rng(SEED)
val_anchors = rng.choice(anchors, size=min(2000, len(anchors)), replace=False)
val_set = set(val_anchors.tolist())
vj = sample_pos(val_anchors, rng)
val_pos = np.stack([val_anchors, vj], 1)
vh_mask = hard_len[val_anchors] > 0
vhj = sample_negs(val_anchors[vh_mask], 1, rng).ravel()
val_hard = np.stack([val_anchors[vh_mask], vhj], 1)
train_anchors = np.array([a for a in anchors.tolist() if a not in val_set], np.int64)
log(f"val_pos {len(val_pos):,} val_hard {len(val_hard):,} train_anchors {len(train_anchors):,}")


class KWTA(torch.autograd.Function):
    """Hard top-k in forward; straight-through gradient to the selected units only."""
    @staticmethod
    def forward(ctx, x, k):
        thr = torch.topk(x, k, dim=1).values[:, -1:].detach()
        mask = (x >= thr).to(x.dtype)
        ctx.save_for_backward(mask)
        return x * mask
    @staticmethod
    def backward(ctx, g):
        (mask,) = ctx.saved_tensors
        return g * mask, None


def kwta(x, k): return KWTA.apply(x, k)


class Encoder(nn.Module):
    def __init__(self, din):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, 1024), nn.GELU(), nn.Dropout(0.1),
                                 nn.Linear(1024, 1024), nn.GELU(),
                                 nn.Linear(1024, D_CODE))
    def encode(self, x, hard=True):
        h = F.relu(self.net(x))                 # nonneg pre-activation
        c = kwta(h, K_WTA) if hard else h
        return F.normalize(c, dim=-1)           # unit-norm nonneg -> overlap in [0,1]
    def forward(self, x): return self.encode(x)


def run_variant(use_blocks, tag):
    torch.manual_seed(SEED); np.random.seed(SEED)
    X = np.hstack([blocks[b] for b in use_blocks]).astype(np.float32)
    Xt = torch.from_numpy(X).to(dev)
    enc = Encoder(X.shape[1]).to(dev)
    opt = torch.optim.Adam(enc.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS * STEPS_PER_EPOCH)
    log(f"[{tag}] input dim {X.shape[1]} blocks={use_blocks}")

    def val_auc():
        enc.eval()
        with torch.no_grad():
            Call = enc.encode(Xt, hard=True)
            po = (Call[val_pos[:, 0]] * Call[val_pos[:, 1]]).sum(1).cpu().numpy()
            ho = (Call[val_hard[:, 0]] * Call[val_hard[:, 1]]).sum(1).cpu().numpy()
        # AUC: does POSITIVE overlap exceed HARD-NEG overlap? (separability of the objective)
        s = np.concatenate([po, ho]); y = np.concatenate([np.ones(len(po)), np.zeros(len(ho))])
        order = np.argsort(s); r = np.empty(len(s)); r[order] = np.arange(len(s))
        n1 = y.sum(); n0 = len(y) - n1
        auc = (r[y == 1].sum() - n1 * (n1 - 1) / 2) / (n1 * n0)
        enc.train()
        return float(auc), float(po.mean()), float(ho.mean())

    best = {"auc": -1}
    rgen = np.random.default_rng(SEED)
    for ep in range(EPOCHS):
        for _ in range(STEPS_PER_EPOCH):
            ai = rgen.choice(train_anchors, BATCH)
            pj = sample_pos(ai, rgen)
            hj = sample_negs(ai, N_HARD, rgen)          # text-close, never co-annotated
            rj = rgen.integers(0, N, (BATCH, N_RAND))
            idx = np.concatenate([ai, pj, hj.ravel(), rj.ravel()])
            C = enc.encode(Xt[torch.from_numpy(idx).to(dev)], hard=True)
            a = C[:BATCH]; p = C[BATCH:2*BATCH]
            hn = C[2*BATCH:2*BATCH+BATCH*N_HARD].view(BATCH, N_HARD, D_CODE)
            rn = C[2*BATCH+BATCH*N_HARD:].view(BATCH, N_RAND, D_CODE)
            s_pos = (a * p).sum(1, keepdim=True)                         # (B,1)
            s_hard = torch.einsum("bd,bkd->bk", a, hn)                   # (B,H)
            s_rand = torch.einsum("bd,bkd->bk", a, rn)                   # (B,R)
            s_in = a @ p.t()                                             # in-batch (B,B)
            eye = torch.eye(BATCH, device=dev).bool()
            s_in = s_in.masked_fill(eye, -1e9)
            logits = torch.cat([s_pos, s_hard, s_rand, s_in], dim=1) / TEMP
            target = torch.zeros(BATCH, dtype=torch.long, device=dev)
            loss = F.cross_entropy(logits, target)
            loss = loss + F.relu(s_hard - s_pos + MARGIN).mean()        # explicit separation margin
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        auc, pm, hm = val_auc()
        if auc > best["auc"]:
            best = {"auc": auc, "pos_ovl": pm, "hard_ovl": hm, "ep": ep,
                    "state": {k: v.detach().cpu().clone() for k, v in enc.state_dict().items()}}
        log(f"[{tag}] ep{ep:02d} loss {loss.item():.3f} valAUC {auc:.4f} pos_ovl {pm:.3f} hard_ovl {hm:.3f}")

    enc.load_state_dict(best["state"])
    enc.eval()
    with torch.no_grad():
        codes = enc.encode(Xt, hard=True).cpu().numpy().astype(np.float32)
    torch.save({"state_dict": best["state"], "blocks": use_blocks, "in_dim": int(X.shape[1]),
                "D_CODE": D_CODE, "K_WTA": K_WTA, "seed": SEED, "tag": tag,
                "best_epoch": best["ep"]}, OUT / f"encoder_{tag}.pt")
    np.savez(OUT / f"go_codes_{tag}.npz", go_ids=vocab, codes=codes, K_WTA=K_WTA, D_CODE=D_CODE)
    nnz = (codes != 0).sum(1)
    log(f"[{tag}] SAVED codes {codes.shape} nnz_med {int(np.median(nnz))} best_ep {best['ep']} "
        f"valAUC {best['auc']:.4f}")
    return {"tag": tag, "blocks": use_blocks, "in_dim": int(X.shape[1]),
            "val_sep_auc": round(best["auc"], 4), "pos_overlap": round(best["pos_ovl"], 4),
            "hard_overlap": round(best["hard_ovl"], 4), "best_epoch": best["ep"],
            "nnz_med": int(np.median(nnz))}


VARIANTS = [(["coann"], "coann"),
            (["coann", "struct"], "coann_struct"),
            (["coann", "struct", "text"], "coann_struct_text")]
only = sys.argv[1] if len(sys.argv) > 1 else None
report = {}
for blk, tag in VARIANTS:
    if only and tag != only: continue
    report[tag] = run_variant(blk, tag)
    json.dump(report, open(OUT / "phase1_ablation.json", "w"), indent=1)
log(f"DONE ablation {json.dumps(report, indent=1)}")
