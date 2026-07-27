"""Frozen-rep Sparse Semantic-Entailment (SSE) over the DEPLOYED champion k-WTA codes.

The protein representation is FROZEN: the champion learned k-WTA codes (config d8979601, 2048-d,
exactly 128 active dims per protein). We learn ONLY the per-term sparse codes z_t and the relation
masks against those frozen supports. supp(x_p) = the 128 non-zero dims of the champion code.

  coverage(p,t) = |supp(x_p) cap supp(z_t)| / |supp(z_t)|            (sparse set containment)
  term tower    = embedding(n_terms x 2048) -> soft k-WTA with per-term cardinality m_t by depth
  axioms (t0 GO EL normal forms go_2025-07-22): nf1 subsumption supp(z_D) superset supp(z_C),
                nf4 existential supp(R_r . z_D) subset supp(z_C); parent gate DETACHED (POC fix)
  K seeds trained (K=1 primary / accept-all; K=5 MIN kept only as a control).

TEMPORAL GATE / LEAKAGE: train codes = clf_protein_codes_big.npz (554,378 v227 curated corpus,
d8979601). Labels = clf_labels_big.json (v227=t0 leaf GO) propagated per aspect. The 7,401 sealed
LAFA eval proteins are HELD OUT of training entirely (no protein appears in the term-code fit).
Eval window v227->v230 is blind by construction. Frozen on-disk data only; NO live DB.
"""
import json, time, os, collections
from pathlib import Path
import numpy as np, scipy.sparse as sp
import torch as th
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)

ROOT = Path("/home/frapercan/Thesis2")
RR   = ROOT / "repositories/protea-reranker-lab/results/sparse_classifier"
OBO  = ROOT / "protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
IA_F = ROOT / "protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv"
NORM = ROOT / "storage/deepgose_kwta/data/go_2025-07-22.norm"
GTDIR= RR / "lafa_gt"
OUT  = ROOT / "storage/sse_kwta_gen"; OUT.mkdir(exist_ok=True)
MDIR = OUT / "models"; MDIR.mkdir(exist_ok=True)
DEV  = "cuda:0"

N        = 2048
M_MIN, M_MAX = 16, 112
MIN_COUNT= int(os.environ.get("SSE_MINCOUNT", 25))
K_SEEDS  = int(os.environ.get("SSE_SEEDS", 5))
EPOCHS   = int(os.environ.get("SSE_EPOCHS", 12))
BATCH    = 1024
TEMP     = 0.3
POS_W    = 20.0
MARGIN   = 0.3
BETA_MARG= 1.0
GAMMA_AX = 0.8
HARDNEG  = 20
AX_BATCH = 4096
LR       = 1e-3
ASPECTS  = {"bpo": "biological_process", "mfo": "molecular_function", "cco": "cellular_component"}
ASPLET   = {"bpo": "P", "mfo": "F", "cco": "C"}

st = {"stage": "start", "aspects": {}}
def save_status():
    json.dump(st, open(OUT / "train_status.json", "w"), indent=1, default=float)

# ---------------- obo: namespaces, ancestors, alt, IA ----------------
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
NSSET = {a: {t for t, n in ns.items() if n == full} for a, full in ASPECTS.items()}
_anc = {}
def anc(t, aspset):
    t = alt.get(t, t)
    key = (t, id(aspset))
    if key in _anc: return _anc[key]
    seen, stk = set(), [t]
    while stk:
        x = stk.pop()
        if x in seen: continue
        seen.add(x)
        for p in par.get(x, ()): stk.append(p)
    r = frozenset(x for x in seen if x in aspset); _anc[key] = r; return r
IA = {}
for line in open(IA_F):
    p = line.split("\t")
    if len(p) >= 2:
        try: IA[p[0].strip()] = float(p[1])
        except ValueError: pass

# ---------------- frozen champion k-WTA codes (train corpus), hold eval out ----------------
Z = np.load(RR / "clf_protein_codes_big.npz", allow_pickle=True)
accs = Z["accs"].tolist(); CODES = Z["codes"]        # (554378, 2048) fp16, d8979601 champion k-WTA
acc_idx = {a: i for i, a in enumerate(accs)}
log(f"champion train codes {CODES.shape} {CODES.dtype} nnz/row={int((CODES[0]!=0).sum())}")

eval_accs = set(np.load(RR / "eval_protein_codes.npz", allow_pickle=True)["accs"].tolist())
lab_big = json.load(open(RR / "clf_labels_big.json"))
log(f"eval proteins to HOLD OUT of training: {len(eval_accs)}")

# training pool = corpus proteins NOT in eval, with >=1 leaf label
train_pool = [i for i, a in enumerate(accs) if a not in eval_accs and lab_big.get(a)]
_LIM = int(os.environ.get("TRAIN_LIMIT", 0))
if _LIM:
    train_pool = train_pool[:_LIM]
    log(f"TRAIN_LIMIT={_LIM} (smoke)")
log(f"training pool (corpus minus eval, labelled): {len(train_pool)}")

# support (binary) precompute lazily per batch; keep codes as-is (fp16 memmap-like ndarray)

def train_aspect(asp):
    full = ASPECTS[asp]; aspset = NSSET[asp]
    log(f"===== {asp} ({full}) =====")
    # ---- per-aspect propagated labels over the training pool ----
    rows, cols = [], []
    term_count = collections.Counter()
    prot_terms = []
    for pi in train_pool:
        a = accs[pi]; prop = set()
        for g in lab_big.get(a, ()):
            g = alt.get(g, g)
            if g in aspset: prop |= anc(g, aspset)
        prot_terms.append(prop)
        for g in prop: term_count[g] += 1
    vocab = sorted([g for g, c in term_count.items() if c >= MIN_COUNT and IA.get(g, 0) > 0])
    tidx = {g: j for j, g in enumerate(vocab)}; n_terms = len(vocab)
    log(f"  {asp} vocab (>= {MIN_COUNT} train pos, IA>0): {n_terms} terms")
    for r, prop in enumerate(prot_terms):
        for g in prop:
            j = tidx.get(g)
            if j is not None: rows.append(r); cols.append(j)
    P = len(train_pool)
    Y = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(P, n_terms))
    log(f"  {asp} label matrix {Y.shape} nnz {Y.nnz:,}  mean pos/prot {Y.nnz/P:.1f}")

    # per-term bit budget by ontology depth (POC scheme)
    depth = np.array([len(anc(g, aspset)) - 1 for g in vocab], dtype=np.float32)
    dq = (np.argsort(np.argsort(depth)) / max(1, n_terms - 1))
    M_T = np.clip(np.round(M_MIN + dq * (M_MAX - M_MIN)).astype(np.int64), M_MIN, M_MAX)
    MTt = th.tensor(M_T, device=DEV)

    # normal forms restricted to vocab
    relations = {}
    def ridx(r):
        if r not in relations: relations[r] = len(relations)
        return relations[r]
    nf1, nf4 = [], []
    for line in open(NORM):
        s = line.strip().replace("_", ":")
        if " SubClassOf " not in s: continue
        left, right = s.split(" SubClassOf ")
        if len(left) == 10 and len(right) == 10:
            C = alt.get(left, left); D = alt.get(right, right)
            if C in tidx and D in tidx and C != D: nf1.append((tidx[C], tidx[D]))
        elif " some " in right:
            r, d = right.split(" some "); C = alt.get(left, left); D = alt.get(d, d)
            if C in tidx and D in tidx: nf4.append((tidx[C], ridx(r), tidx[D]))
    n_rels = max(1, len(relations))
    NF1 = th.LongTensor(nf1).to(DEV) if nf1 else th.zeros((0, 2), dtype=th.long, device=DEV)
    NF4 = th.LongTensor(nf4).to(DEV) if nf4 else th.zeros((0, 3), dtype=th.long, device=DEV)
    log(f"  {asp} axioms in-vocab nf1={len(nf1)} nf4={len(nf4)} rels={n_rels}")

    # ---- model: FROZEN protein supports, learned term codes ----
    def _thr(a, ks):
        srt = a.sort(1, descending=True).values
        gi = (ks - 1).clamp(0, a.shape[1] - 1).unsqueeze(1)
        return srt.gather(1, gi).detach()
    def soft_term(a, ks): return th.sigmoid((a - _thr(a, ks)) / TEMP)
    def hard_term(a, ks): return (a >= _thr(a, ks)).float()

    class TermTower(nn.Module):
        def __init__(self):
            super().__init__()
            self.term = nn.Embedding(n_terms, N); nn.init.normal_(self.term.weight, std=0.02)
            self.tln = nn.LayerNorm(N)
            self.rel = nn.Parameter(th.zeros(n_rels, N))
        def gates(self, hard=False):
            a = self.tln(self.term.weight)
            return hard_term(a, MTt) if hard else soft_term(a, MTt)
    def coverage(xb, gt): return (xb @ gt.t()) / (gt.sum(1) + 1e-6)   # xb (B,N) binary, gt (T,N)

    def supp_batch(pool_idx):
        raw = CODES[np.asarray(pool_idx)]           # (b,2048) fp16
        return th.tensor((raw != 0).astype(np.float32), device=DEV)

    idx_all = np.arange(P)
    val_aucs = []
    for seed in range(K_SEEDS):
        th.manual_seed(seed); np.random.seed(seed)
        net = TermTower().to(DEV); opt = th.optim.Adam(net.parameters(), lr=LR)
        rng = np.random.default_rng(seed); order0 = idx_all.copy(); rng.shuffle(order0)
        nval = int(0.05 * P); vi = order0[:nval]; ti = order0[nval:]
        Xv = supp_batch([train_pool[k] for k in vi]); Yv = th.tensor(Y[vi].toarray())
        best_auc, best_state = 0.0, None
        for ep in range(EPOCHS):
            net.train(); perm = np.random.permutation(ti); tot = 0.0; nb = 0
            for s in range(0, len(perm), BATCH):
                b = perm[s:s + BATCH]
                xb = supp_batch([train_pool[k] for k in b])
                yb = th.tensor(Y[b].toarray(), device=DEV)
                gt = net.gates()
                S = coverage(xb, gt).clamp(1e-6, 1 - 1e-6)
                w = th.where(yb > 0, POS_W, 1.0)
                loss = F.binary_cross_entropy(S, yb, weight=w)
                neg = S.masked_fill(yb > 0, -1.0)
                hn = neg.topk(min(HARDNEG, S.shape[1]), 1).values
                pos_mean = (S * yb).sum(1) / (yb.sum(1) + 1e-6)
                loss = loss + BETA_MARG * th.relu(MARGIN - pos_mean.unsqueeze(1) + hn).mean()
                ax = th.zeros((), device=DEV)
                if len(NF1):
                    sel = NF1[th.randint(0, len(NF1), (min(AX_BATCH, len(NF1)),), device=DEV)]
                    gC = gt[sel[:, 0]]; gD = gt[sel[:, 1]].detach()
                    ax = ax + th.relu(gD - gC).sum(1).mean() / M_T.mean()
                if len(NF4):
                    sel = NF4[th.randint(0, len(NF4), (min(AX_BATCH, len(NF4)),), device=DEV)]
                    gC = gt[sel[:, 0]]; mask = th.sigmoid(net.rel[sel[:, 1]]); gD = gt[sel[:, 2]].detach()
                    ax = ax + th.relu(gD * mask - gC).sum(1).mean() / M_T.mean()
                loss = loss + GAMMA_AX * ax
                opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item(); nb += 1
            net.eval()
            with th.no_grad():
                gt = net.gates()
                preds = []
                for s in range(0, len(Xv), 4096):
                    preds.append(coverage(Xv[s:s+4096], gt).cpu().numpy())
                Pm = np.concatenate(preds).ravel(); Yt = Yv.numpy().ravel()
                pos = np.where(Yt > 0)[0]; ne = np.where(Yt == 0)[0]
                negs = np.random.default_rng(0).choice(ne, min(len(pos)*20, len(ne)), replace=False)
                selv = np.concatenate([pos, negs]); auc = roc_auc_score(Yt[selv], Pm[selv])
            if auc > best_auc: best_auc = auc; best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            log(f"  {asp} seed {seed} ep {ep}: loss {tot/nb:.4f} val_auc {auc:.4f} (best {best_auc:.4f})")
        th.save(best_state, MDIR / f"{asp}_seed{seed}.th")
        val_aucs.append(best_auc)

    # ---- persist per-aspect hard+soft term codes for all seeds + meta ----
    hard_codes = []; soft_codes = []
    for seed in range(K_SEEDS):
        net = TermTower().to(DEV); net.load_state_dict(th.load(MDIR / f"{asp}_seed{seed}.th")); net.eval()
        with th.no_grad():
            hard_codes.append(net.gates(hard=True).cpu().numpy().astype(bool))
            soft_codes.append(net.gates(hard=False).cpu().numpy().astype(np.float16))
    np.savez_compressed(OUT / f"termcodes_{asp}.npz",
                        vocab=np.array(vocab), M_T=M_T, depth=depth,
                        train_support=np.array([term_count[g] for g in vocab]),
                        hard=np.stack(hard_codes), soft=np.stack(soft_codes),
                        nf1=np.array(nf1) if nf1 else np.zeros((0, 2), int),
                        nf4=np.array(nf4) if nf4 else np.zeros((0, 3), int))
    st["aspects"][asp] = {"n_terms": n_terms, "n_train": P, "nf1": len(nf1), "nf4": len(nf4),
                          "val_auc_per_seed": [round(x, 4) for x in val_aucs],
                          "mean_val_auc": round(float(np.mean(val_aucs)), 4)}
    save_status()
    log(f"  {asp} DONE mean val_auc {np.mean(val_aucs):.4f}")

st["stage"] = "training"; save_status()
for asp in ["bpo", "mfo", "cco"]:
    train_aspect(asp)
st["stage"] = "done"; save_status()
log("TRAIN ALL DONE")
