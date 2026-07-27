"""Phase 2: LEARN the sequence -> GO-code projection into the FROZEN Phase-1 code space.

Head: protein Ankh/d8979601 code (2048, k-WTA) -> MLP -> ReLU -> k-WTA(128) -> L2, landing in
the SAME 2048-d GO-code space as the Phase-1 term codes. Objective: a protein's projected code
SPARSE-OVERLAPS the codes of its TRUE t0 terms, not false ones. Full-BP-vocab multi-positive
InfoNCE (every non-true term competes as a negative; the highest-overlap non-true terms are the
hard negatives), IA-weighted so the deep tail is not drowned.

TEMPORAL discipline: every positive is a t0 (<= v227) corpus annotation; the frozen corpus carries
no per-annotation dates, so early-stopping uses a held-out slice of TRAIN proteins (leakage-safe:
the eval targets' NOVEL post-t0 terms, the actual test, are never seen). Deterministic (seed 0).

Arg1 = which Phase-1 GO-code variant to project onto (default coann_struct_text).
Read-only w.r.t. repos; writes only under storage/kwta_go_encoder/.
"""
import json, time, sys, collections
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, pyarrow.parquet as pq

t0 = time.time()
def log(m): print(f"[{time.time()-t0:.0f}s] {m}", flush=True)

OUT = Path("/home/frapercan/Thesis2/storage/kwta_go_encoder")
FROZEN = Path("/home/frapercan/Thesis2/storage/protea-frozen-v227-2025-09-04")
SC = Path("/home/frapercan/Thesis2/repositories/protea-reranker-lab/results/sparse_classifier")
T0D = Path("/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025")
OBO = str(T0D / "go-basic.obo"); IA_F = str(T0D / "IA.tsv")
dev = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0; D_CODE, K_WTA = 2048, 128
EPOCHS, BATCH, LR, TEMP = 30, 256, 1e-3, 0.15
N_VAL = 3000
VARIANT = sys.argv[1] if len(sys.argv) > 1 else "coann_struct_text"
torch.manual_seed(SEED); np.random.seed(SEED)

# ---- ontology closure + IA ----
par = collections.defaultdict(set); ns = {}; alt = {}; cur = None
for line in open(OBO):
    line = line.rstrip("\n")
    if line == "[Term]": cur = None
    elif line.startswith("id: GO:"): cur = line[4:]
    elif cur and line.startswith("namespace: "): ns[cur] = line[11:]
    elif cur and line.startswith("is_a: GO:"): par[cur].add(line[6:].split(" ! ")[0].strip())
    elif cur and line.startswith("relationship: part_of GO:"): par[cur].add(line[len("relationship: part_of "):].split(" ! ")[0].strip())
    elif cur and line.startswith("alt_id: GO:"): alt[line[8:]] = cur
BP = {t for t, n in ns.items() if n == "biological_process"}
AC = {}
def anc(t):
    if t in AC: return AC[t]
    o, st = set(), [t]
    while st:
        x = st.pop()
        for p in par.get(x, ()):
            if p not in o: o.add(p); st.append(p)
    AC[t] = o; return o
IA = {}
for line in open(IA_F):
    p = line.rstrip("\n").split("\t")
    if len(p) >= 2:
        try: IA[p[0]] = float(p[1])
        except ValueError: pass

# ---- frozen Phase-1 GO codes ----
gc = np.load(OUT / f"go_codes_{VARIANT}.npz", allow_pickle=True)
vocab = np.array([alt.get(g, g) for g in gc["go_ids"].tolist()])
term2idx = {g: i for i, g in enumerate(vocab)}; Nt = len(vocab)
Gcodes = torch.from_numpy(gc["codes"].astype(np.float32)).to(dev)     # (Nt, D) frozen
D_CODE = Gcodes.shape[1]; K_WTA = min(K_WTA, D_CODE)                  # match the code space
vocab_ia = torch.tensor([IA.get(g, 0.0) for g in vocab], dtype=torch.float32, device=dev)
log(f"variant {VARIANT}: frozen GO codes {tuple(Gcodes.shape)}")

# ---- train protein codes + t0 true BP terms (all evidence, closure, in vocab) ----
tr = np.load(SC / "clf_protein_codes.npz", allow_pickle=True)
tr_accs = tr["accs"].tolist(); tr_codes = tr["codes"].astype(np.float32)
acc2row = {a: i for i, a in enumerate(tr_accs)}
meta = pq.read_table(FROZEN / "go_term_metadata.parquet")
id2go = {i: g for i, g in zip(meta.column("go_term_id").to_pylist(), meta.column("go_id").to_pylist())}
ra = pq.read_table(FROZEN / "reference_annotations.parquet")
acc = ra.column("accession").to_pylist(); gid = ra.column("go_term_id").to_pylist()
direct = collections.defaultdict(set)
for a, gi in zip(acc, gid):
    g = id2go.get(gi)
    if g is None: continue
    g = alt.get(g, g)
    if g in BP: direct[a].add(g)
positives = {}
for a in tr_accs:
    if a not in direct: continue
    cl = set()
    for g in direct[a]: cl.add(g); cl |= (anc(g) & BP)
    idx = sorted({term2idx[g] for g in cl if g in term2idx})
    if idx: positives[a] = idx
prots = [a for a in tr_accs if a in positives]
log(f"train proteins with t0 BP positives: {len(prots):,} (of {len(tr_accs):,})")

rng = np.random.default_rng(SEED)
val_prots = set(rng.choice(prots, size=min(N_VAL, len(prots)), replace=False).tolist())
train_prots = [a for a in prots if a not in val_prots]
Xcode = torch.from_numpy(tr_codes).to(dev)


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2048, 2048), nn.GELU(), nn.Dropout(0.1),
                                 nn.Linear(2048, 2048), nn.GELU(), nn.Linear(2048, D_CODE))
    def forward(self, x):
        h = F.relu(self.net(x))
        thr = torch.topk(h, K_WTA, dim=1).values[:, -1:]
        h = h * (h >= thr).to(h.dtype)          # hard k-WTA (STE not needed: selection ~ ReLU mag)
        return F.normalize(h, dim=-1)


head = Head().to(dev)
opt = torch.optim.Adam(head.parameters(), lr=LR)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS * (len(train_prots) // BATCH + 1))

def build_target(batch_prots):
    """multi-positive IA-weighted target rows over the full vocab."""
    Y = torch.zeros(len(batch_prots), Nt, device=dev)
    for r, a in enumerate(batch_prots):
        idx = positives[a]
        Y[r, idx] = vocab_ia[idx].clamp(min=0.1)     # IA weight, floor so common terms still count
    Y = Y / Y.sum(1, keepdim=True).clamp(min=1e-6)
    return Y

@torch.no_grad()
def val_recall(ks=(5, 20, 50)):
    head.eval()
    vp = [a for a in val_prots if a in acc2row]
    rows = torch.tensor([acc2row[a] for a in vp])
    rec = {k: [] for k in ks}
    for s in range(0, len(vp), 512):
        S = head(Xcode[rows[s:s+512].to(dev)])
        sim = S @ Gcodes.t()
        topk = torch.topk(sim, max(ks), dim=1).indices.cpu().numpy()
        for r, a in enumerate(vp[s:s+512]):
            ps = set(positives[a])
            for k in ks:
                rec[k].append(len(ps & set(topk[r, :k].tolist())) / min(k, len(ps)))
    head.train()
    return {k: round(float(np.mean(v)), 4) for k, v in rec.items()}

best = {"r20": -1}
for ep in range(EPOCHS):
    order = rng.permutation(len(train_prots))
    losses = []
    for s in range(0, len(order), BATCH):
        bp = [train_prots[i] for i in order[s:s+BATCH]]
        rows = torch.tensor([acc2row[a] for a in bp])
        S = head(Xcode[rows.to(dev)])
        sim = (S @ Gcodes.t()) / TEMP                # (B, Nt) overlap
        logp = F.log_softmax(sim, dim=1)
        Y = build_target(bp)
        loss = -(Y * logp).sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        losses.append(loss.item())
    rec = val_recall()
    if rec[20] > best["r20"]:
        best = {"r20": rec[20], "rec": rec, "ep": ep,
                "state": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}}
    log(f"ep{ep:02d} loss {np.mean(losses):.3f} val_recall {rec}")

head.load_state_dict(best["state"]); head.eval()
torch.save({"state_dict": best["state"], "variant": VARIANT, "D_CODE": D_CODE, "K_WTA": K_WTA,
            "seed": SEED, "best_epoch": best["ep"]}, OUT / f"seq_head_{VARIANT}.pt")

# project eval proteins + emit codes
ev = np.load(SC / "eval_protein_codes.npz", allow_pickle=True)
ev_accs = ev["accs"]; ev_codes = torch.from_numpy(ev["codes"].astype(np.float32)).to(dev)
with torch.no_grad():
    Pev = torch.vstack([head(ev_codes[s:s+512]) for s in range(0, len(ev_accs), 512)]).cpu().numpy()
np.savez(OUT / f"eval_proj_codes_{VARIANT}.npz", accs=ev_accs, codes=Pev.astype(np.float32),
         K_WTA=K_WTA, D_CODE=D_CODE)
rep = {"variant": VARIANT, "best_epoch": best["ep"], "val_recall": best["rec"],
       "train_prots": len(train_prots), "val_prots": len(val_prots), "Nt": Nt}
json.dump(rep, open(OUT / f"phase2_{VARIANT}.json", "w"), indent=1)
log(f"DONE phase2 {VARIANT}: {rep}; wrote eval_proj_codes_{VARIANT}.npz")
