"""SSE full-corpus trainer: per-aspect (MFO/BPO/CCO) Sparse Semantic-Entailment.

Scale-up of storage/sse_poc/sse_train_infer.py. One independent SSE model PER ASPECT, trained on the
FULL experimental corpus (all non-eval proteins with >=1 annotation in that aspect; ESM2-3B mean input).
Temporal/leakage gate: EVAL proteins (LK+PK groundtruth, ALL aspects) are excluded from every aspect's
training set. Ensemble K seeds; MIN consensus at inference. EL axioms (go_2025-07-22.norm) as soft
set-containment penalties, restricted to each aspect's in-vocab terms.

Saves per aspect: models/{asp}_{seed}.th, {asp}_meta.npz (terms, M_T, mu, sd), and appends to
sse_full_train.json (val AUC per seed, axiom satisfaction, config). Run under PROTEA/.venv (torch+CUDA).
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
GF = ROOT / "storage/cooc_experiment/generator_frames"
OBO = ROOT / "protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
NORM = ROOT / "storage/deepgose_kwta/data/go_2025-07-22.norm"
GTDIR = ROOT / "repositories/protea-reranker-lab/results/sparse_classifier/lafa_gt"
OUT = ROOT / "storage/sse_full"; OUT.mkdir(exist_ok=True)
MDIR = OUT / "models"; MDIR.mkdir(exist_ok=True)
DEV = "cuda:0"

# ---- hyperparameters (full-corpus; matches POC SDR geometry) ----
N        = 2048
PK       = 128
PM       = 64
M_MIN    = 16
M_MAX    = 112
MIN_SUB  = int(os.environ.get("SSE_MINSUB", 40))   # min positives in corpus to keep a term
K_SEEDS  = int(os.environ.get("SSE_SEEDS", 5))
EPOCHS   = int(os.environ.get("SSE_EPOCHS", 14))
MAXPROT  = int(os.environ.get("SSE_MAXPROT", 0))    # 0 = use ALL eligible
BATCH    = 512
TEMP     = 0.3
POS_W    = 20.0
MARGIN   = 0.3
BETA_MARG= 1.0
GAMMA_AX = float(os.environ.get("SSE_GAMMA", 0.8))
HARDNEG  = 20
AX_BATCH = 4096
LR       = 1e-3
ASPECTS  = {"mfo": "molecular_function", "bpo": "biological_process", "cco": "cellular_component"}

# ---------------- obo: namespaces, ancestors, alt ----------------
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
_anc = {}
def anc_ns(t, NSET):
    t = alt.get(t, t)
    key = (t, id(NSET))
    if key in _anc: return _anc[key]
    seen, st = set(), [t]
    while st:
        x = st.pop()
        if x in seen: continue
        seen.add(x)
        for p in par.get(x, ()): st.append(p)
    r = frozenset(x for x in seen if x in NSET); _anc[key] = r; return r

# ---------------- shared corpus data ----------------
accs = json.load(open(GF / "accs.json"))
acc_idx = {a: i for i, a in enumerate(accs)}
EMB = np.load(GF / "raw_esm2_3b.npy", mmap_mode="r")
vocab = json.load(open(GF / "vocab.json"))
L = np.load(GF / "labels.npz")
A = sp.csr_matrix((L["data"], L["indices"], L["indptr"]), shape=tuple(L["shape"]))

import pandas as pd
eval_prots = set()
for fn in ["groundtruth_LK.tsv", "groundtruth_PK.tsv"]:
    gt = pd.read_csv(GTDIR / fn, sep="\t")
    eval_prots |= set(gt.EntryID.unique())
log(f"eval proteins held out of ALL training: {len(eval_prots)}")

# ---------------- model ----------------
def soft_kwta_scalar(a, k):
    thr = a.topk(k, dim=1).values[:, -1:].detach()
    return th.sigmoid((a - thr) / TEMP)
def hard_kwta_scalar(a, k):
    idx = a.topk(k, dim=1).indices
    m = th.zeros_like(a); m.scatter_(1, idx, 1.0); return m
def _thr_perrow(a, ks):
    srt = a.sort(dim=1, descending=True).values
    gi = (ks - 1).clamp(0, a.shape[1] - 1).unsqueeze(1)
    return srt.gather(1, gi).detach()
def soft_kwta_perrow(a, ks): return th.sigmoid((a - _thr_perrow(a, ks)) / TEMP)
def hard_kwta_perrow(a, ks): return (a >= _thr_perrow(a, ks)).float()

class SSE(nn.Module):
    def __init__(self, in_dim, n_terms, n_rels, MTt):
        super().__init__()
        self.MTt = MTt
        self.prot = nn.Sequential(nn.Linear(in_dim, 1024), nn.LayerNorm(1024), nn.GELU(),
                                  nn.Dropout(0.1), nn.Linear(1024, N))
        self.pln = nn.LayerNorm(N)
        self.term = nn.Embedding(n_terms, N); nn.init.normal_(self.term.weight, std=0.02)
        self.tln = nn.LayerNorm(N)
        self.rel = nn.Parameter(th.zeros(n_rels, N))
    def prot_gate(self, x, hard=False):
        a = self.pln(self.prot(x))
        return hard_kwta_scalar(a, PK) if hard else soft_kwta_scalar(a, PK)
    def term_gate(self, ids=None, hard=False):
        a = self.tln(self.term.weight if ids is None else self.term(ids))
        ks = self.MTt if ids is None else self.MTt[ids]
        return hard_kwta_perrow(a, ks) if hard else soft_kwta_perrow(a, ks)
    def coverage(self, gp, gt): return (gp @ gt.t()) / (gt.sum(1) + 1e-6)

RESULTS = {"config": {"N": N, "PK": PK, "PM": PM, "MIN_SUB": MIN_SUB, "K_SEEDS": K_SEEDS,
    "EPOCHS": EPOCHS, "GAMMA_AX": GAMMA_AX, "corpus": "cooc_experiment/generator_frames raw_esm2_3b",
    "n_eval_heldout": len(eval_prots)}, "aspects": {}}
def save_results(): json.dump(RESULTS, open(OUT / "sse_full_train.json", "w"), indent=1, default=float)

for asp, nsname in ASPECTS.items():
    log(f"########## ASPECT {asp} ({nsname}) ##########")
    NSET = frozenset(t for t, n in ns.items() if n == nsname)
    has = np.asarray(A[:, [i for i in range(len(vocab)) if vocab[i] in NSET]].sum(1)).ravel() > 0
    elig = [i for i, a in enumerate(accs) if has[i] and a not in eval_prots]
    rng = np.random.default_rng(0); rng.shuffle(elig)
    if MAXPROT and len(elig) > MAXPROT: elig = elig[:MAXPROT]
    sub = np.array(sorted(elig))
    log(f"{asp}: corpus {len(sub)} proteins")

    Asub = A[sub].tocsc()
    ns_cols = np.array([i for i in range(len(vocab)) if vocab[i] in NSET])
    cnt = np.asarray(Asub[:, ns_cols].sum(0)).ravel()
    keep = ns_cols[cnt >= MIN_SUB]
    terms = [vocab[i] for i in keep]; term_idx = {g: j for j, g in enumerate(terms)}
    n_terms = len(terms)
    log(f"{asp}: vocab (>= {MIN_SUB} pos) {n_terms} terms")

    depth = np.array([len(anc_ns(g, NSET)) - 1 for g in terms], dtype=np.float32)
    dq = (np.argsort(np.argsort(depth)) / max(1, n_terms - 1))
    M_T = np.clip(np.round(M_MIN + dq * (M_MAX - M_MIN)).astype(np.int64), M_MIN, M_MAX)
    Ysub = Asub[:, keep].toarray().astype(np.float32)
    log(f"{asp}: labels {Ysub.shape} pos/prot {Ysub.sum(1).mean():.1f}")
    Xsub = np.ascontiguousarray(EMB[sub].astype(np.float32))
    mu = Xsub.mean(0); sd = Xsub.std(0) + 1e-6; Xsub = (Xsub - mu) / sd

    # axioms restricted to aspect vocab
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
    log(f"{asp}: axioms nf1 {len(nf1)} nf4 {len(nf4)} rels {n_rels}")
    MTt = th.tensor(M_T, device=DEV)

    def train_one(seed):
        th.manual_seed(seed); np.random.seed(seed)
        net = SSE(Xsub.shape[1], n_terms, n_rels, MTt).to(DEV)
        opt = th.optim.Adam(net.parameters(), lr=LR)
        idx = np.arange(len(sub)); np.random.default_rng(seed).shuffle(idx)
        nval = int(0.1 * len(idx)); vi = idx[:nval]; ti = idx[nval:]
        Xt = th.tensor(Xsub); Yt = th.tensor(Ysub)
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
                    gC = gt[sel[:, 0]]; gD = gt[sel[:, 1]].detach()
                    ax = ax + th.relu(gD - gC).sum(1).mean() / PM
                if len(NF4):
                    sel = NF4[th.randint(0, len(NF4), (min(AX_BATCH, len(NF4)),), device=DEV)]
                    gC = gt[sel[:, 0]]; mask = th.sigmoid(net.rel[sel[:, 1]]); gD = gt[sel[:, 2]].detach()
                    ax = ax + th.relu(gD * mask - gC).sum(1).mean() / PM
                loss = loss_bce + BETA_MARG * loss_marg + GAMMA_AX * ax
                opt.zero_grad(); loss.backward(); opt.step()
                tot += loss.item(); nb += 1
            net.eval()
            with th.no_grad():
                gt = net.term_gate(); preds = []
                for s in range(0, len(Xv), 1024):
                    gp = net.prot_gate(Xv[s:s + 1024]); preds.append(net.coverage(gp, gt).cpu().numpy())
                P = np.concatenate(preds).ravel(); Ytrue = Yv.numpy().ravel()
                pos = np.where(Ytrue > 0)[0]; ne = np.where(Ytrue == 0)[0]
                negs = np.random.default_rng(0).choice(ne, min(len(pos) * 20, len(ne)), replace=False)
                sel = np.concatenate([pos, negs]); auc = roc_auc_score(Ytrue[sel], P[sel])
            if auc > best_auc:
                best_auc = auc; best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            log(f"  {asp} seed {seed} ep {ep}: loss {tot/nb:.4f} val_auc {auc:.4f} (best {best_auc:.4f})")
        th.save(best_state, MDIR / f"{asp}_{seed}.th")
        return best_auc

    aucs = {}
    for seed in range(K_SEEDS):
        aucs[seed] = train_one(seed)
        log(f"{asp} model {seed} done val_auc {aucs[seed]:.4f} | elapsed {time.time()-t0:.0f}s")

    np.savez_compressed(OUT / f"{asp}_meta.npz", terms=np.array(terms), M_T=M_T, mu=mu, sd=sd,
                        n_rels=np.array([n_rels]))

    # axiom satisfaction (hard, seed0)
    net0 = SSE(Xsub.shape[1], n_terms, n_rels, MTt).to(DEV)
    net0.load_state_dict(th.load(MDIR / f"{asp}_0.th")); net0.eval()
    with th.no_grad(): codes = net0.term_gate(hard=True).cpu().numpy().astype(bool)
    ax_report = {}
    for name, NFt, is_ex in [("nf1_subsumption", nf1, False), ("nf4_existential", nf4, True)]:
        if not NFt: continue
        rel_mask = None
        if is_ex:
            rw = th.sigmoid(net0.rel).detach().cpu().numpy(); rel_mask = np.zeros_like(rw, dtype=bool)
            for r in range(rw.shape[0]): rel_mask[r, np.argsort(-rw[r])[:PM]] = True
        ratios = []; full = 0
        for tup in NFt:
            if is_ex: C, r, D = tup; supD = codes[D] & rel_mask[r]
            else: C, D = tup; supD = codes[D]
            nd = supD.sum()
            if nd == 0: continue
            inter = (supD & codes[C]).sum(); ratios.append(inter / nd); full += int(inter == nd)
        ax_report[name] = {"n_pairs": len(ratios), "mean_containment": float(np.mean(ratios)) if ratios else None,
                           "frac_fully_contained": full / max(1, len(ratios))}
    RESULTS["aspects"][asp] = {"n_corpus": int(len(sub)), "n_terms": n_terms, "nf1": len(nf1),
        "nf4": len(nf4), "mean_val_auc": float(np.mean(list(aucs.values()))),
        "val_auc_per_seed": {str(k): float(v) for k, v in aucs.items()}, "axioms": ax_report}
    log(f"{asp}: mean_val_auc {RESULTS['aspects'][asp]['mean_val_auc']:.4f}")
    save_results()

log("SSE FULL TRAIN DONE")
print(json.dumps(RESULTS, indent=1, default=float))
