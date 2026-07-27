"""SSE (Sparse Semantic-Entailment) POC: train + infer, from scratch, on a SUBSAMPLE.

Everything is a SPARSE code (SDR). Entailment = SPARSE SET CONTAINMENT (coverage). Ontology axioms
= differentiable soft-containment constraints on term codes. Ensemble consensus = MIN over K seeds.

  1. PROTEIN TOWER: dense ESM2-3B mean (2560-d, generator_frames/raw_esm2_3b.npy) -> MLP -> soft
     k-WTA(top-PK of N) -> sparse protein code x_p.
  2. GO-TERM TOWER: embedding table (n_terms x N) -> soft k-WTA(top-PM) -> sparse term code z_t.
  3. AXIOM REGULARIZER (t0 GO EL normal forms go_2025-07-22.norm):
       nf1 subsumption  C<=D : supp(z_D) subset of supp(z_C)   (parent's feats subset of child's)
       nf4 existential  C<=R.D: supp(R_r . z_D) subset of supp(z_C), R_r a learned sparse relation mask
     as differentiable soft-containment penalties relu(g_D - g_C).
  4. ENTAILMENT SCORER: s(p,t) = |supp(x_p) cap supp(z_t)| / |supp(z_t)| (soft, differentiable).
  5. ENSEMBLE: K independently-seeded code sets; S(p,t) = MIN_k s_k(p,t).
  Loss = weighted-BCE(coverage) + margin(true > hard negatives) + axiom-containment penalty.

TEMPORAL GATE + LEAKAGE: labels = generator_frames t0/v227 BP labels, ancestor-propagated. EVAL
proteins (LK/PK BP targets) are EXCLUDED from the training subsample entirely (no per-protein
memorization; the term tower still generalizes across terms). Eval-window-new BP annotations are
blind by construction. LK has no -known exclusion, so any LK positive is flagged as suspect.
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
OUT = ROOT / "storage/sse_poc"; OUT.mkdir(exist_ok=True)
MDIR = OUT / "models"; MDIR.mkdir(exist_ok=True)
DEV = "cuda:0"

# ---- hyperparameters (POC: small + fast) ----
N          = 2048                                  # SDR width
PK         = 128                                   # protein active dims (fixed)
PM         = 64                                    # term active dims (nominal / mean)
M_MIN      = 16                                    # term bits for the most general terms (root-ish)
M_MAX      = 112                                   # term bits for the most specific (deep leaf) terms
N_SUB      = int(os.environ.get("SSE_NSUB", 25000))
MIN_SUB    = int(os.environ.get("SSE_MINSUB", 25)) # min positives in subsample to keep a term
K_SEEDS    = int(os.environ.get("SSE_SEEDS", 5))
EPOCHS     = int(os.environ.get("SSE_EPOCHS", 12))
BATCH      = 512
TEMP       = 0.3                                   # soft k-WTA temperature (post-layernorm scale)
POS_W      = 20.0                                  # BCE positive weight
MARGIN     = 0.3
BETA_MARG  = 1.0
GAMMA_AX   = float(os.environ.get("SSE_GAMMA", 0.8))  # axiom penalty weight
HARDNEG    = 20
AX_BATCH   = 4096                                  # sampled axiom pairs / step
LR         = 1e-3

prog = {"stage": "start", "config": {
    "N": N, "PK": PK, "PM": PM, "N_SUB": N_SUB, "MIN_SUB": MIN_SUB, "K_SEEDS": K_SEEDS,
    "EPOCHS": EPOCHS, "BATCH": BATCH, "TEMP": TEMP, "POS_W": POS_W, "MARGIN": MARGIN,
    "BETA_MARG": BETA_MARG, "GAMMA_AX": GAMMA_AX}, "models": {}}
def save(): json.dump(prog, open(OUT / "train_progress.json", "w"), indent=1)

# ---------------- obo: BP, ancestors, alt ----------------
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
BP = {t for t, n in ns.items() if n == "biological_process"}
_anc = {}
def anc(t):
    t = alt.get(t, t)
    if t in _anc: return _anc[t]
    seen, st = set(), [t]
    while st:
        x = st.pop()
        if x in seen: continue
        seen.add(x)
        for p in par.get(x, ()): st.append(p)
    r = frozenset(x for x in seen if x in BP); _anc[t] = r; return r

# ---------------- data: dense ESM2-3B, propagated t0 labels, subsample ----------------
accs = json.load(open(GF / "accs.json"))
acc_idx = {a: i for i, a in enumerate(accs)}
EMB = np.load(GF / "raw_esm2_3b.npy", mmap_mode="r")           # (88212, 2560) fp32
vocab = json.load(open(GF / "vocab.json"))
L = np.load(GF / "labels.npz")
A = sp.csr_matrix((L["data"], L["indices"], L["indptr"]), shape=tuple(L["shape"]))  # propagated

# eval proteins to EXCLUDE from training (leakage discipline)
import pandas as pd
eval_prots = set()
for fn in ["groundtruth_LK.tsv", "groundtruth_PK.tsv"]:
    gt = pd.read_csv(GTDIR / fn, sep="\t")
    eval_prots |= set(gt[gt.aspect == "P"].EntryID.unique())
log(f"eval proteins to hold out of training: {len(eval_prots)}")

has_bp = np.asarray(A[:, [i for i in range(len(vocab)) if vocab[i] in BP]].sum(1)).ravel() > 0
elig = [i for i, a in enumerate(accs) if has_bp[i] and a not in eval_prots]
rng = np.random.default_rng(0); rng.shuffle(elig)
sub = np.array(sorted(elig[:N_SUB]))
log(f"training subsample: {len(sub)} proteins (of {len(elig)} eligible non-eval BP-annotated)")

# vocab = BP terms with >= MIN_SUB positives WITHIN the subsample
Asub = A[sub].tocsc()
bp_cols = np.array([i for i in range(len(vocab)) if vocab[i] in BP])
cnt = np.asarray(Asub[:, bp_cols].sum(0)).ravel()
keep = bp_cols[cnt >= MIN_SUB]
terms = [vocab[i] for i in keep]
term_idx = {g: j for j, g in enumerate(terms)}
n_terms = len(terms)
log(f"BP vocab (min {MIN_SUB} positives in subsample): {n_terms} terms")

# per-term SDR cardinality m_t: general terms (shallow) few bits, specific (deep) many bits, so
# subsumption supp(z_D) subset supp(z_C) is achievable (parent fewer bits, subset of child's).
depth = np.array([len(anc(g)) - 1 for g in terms], dtype=np.float32)   # proper BP ancestors
dq = (np.argsort(np.argsort(depth)) / max(1, n_terms - 1))             # rank in [0,1]
M_T = np.clip(np.round(M_MIN + dq * (M_MAX - M_MIN)).astype(np.int64), M_MIN, M_MAX)
log(f"per-term bits m_t: min {M_T.min()} max {M_T.max()} mean {M_T.mean():.1f} "
    f"(depth range {depth.min():.0f}..{depth.max():.0f})")

# label matrix over subsample x vocab
Ysub = Asub[:, keep].toarray().astype(np.float32)   # (N_SUB, n_terms) propagated binary
log(f"label matrix {Ysub.shape} pos/protein mean {Ysub.sum(1).mean():.1f}")

# input features standardized (per-dim) on the subsample
Xsub = np.ascontiguousarray(EMB[sub].astype(np.float32))
mu = Xsub.mean(0); sd = Xsub.std(0) + 1e-6
Xsub = (Xsub - mu) / sd
log(f"features {Xsub.shape} standardized")

# ---------------- normal forms restricted to vocab (nf1 subsumption, nf4 existential) ----------------
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
        if C in term_idx and D in term_idx and C != D:
            nf1.append((term_idx[C], term_idx[D]))              # C<=D : supp(D) subset supp(C)
    elif " some " in right:
        r, d = right.split(" some ")
        C = alt.get(left, left); D = alt.get(d, d)
        if C in term_idx and D in term_idx:
            nf4.append((term_idx[C], ridx(r), term_idx[D]))      # C<=R.D : supp(R.D) subset supp(C)
n_rels = max(1, len(relations))
NF1 = th.LongTensor(nf1).to(DEV) if nf1 else th.zeros((0, 2), dtype=th.long, device=DEV)
NF4 = th.LongTensor(nf4).to(DEV) if nf4 else th.zeros((0, 3), dtype=th.long, device=DEV)
log(f"axioms in vocab: nf1(subsumption) {len(nf1)}  nf4(existential) {len(nf4)}  rels {n_rels}")
prog["config"].update({"n_terms": n_terms, "n_sub": len(sub), "nf1": len(nf1), "nf4": len(nf4),
                       "n_rels": n_rels, "n_eval_heldout": len(eval_prots)})

# ---------------- model ----------------
MTt = th.tensor(M_T, device=DEV)                       # per-term bit budget

def soft_kwta_scalar(a, k):
    thr = a.topk(k, dim=1).values[:, -1:].detach()
    return th.sigmoid((a - thr) / TEMP)

def hard_kwta_scalar(a, k):
    idx = a.topk(k, dim=1).indices
    m = th.zeros_like(a); m.scatter_(1, idx, 1.0); return m

def _thr_perrow(a, ks):
    srt = a.sort(dim=1, descending=True).values          # (T,N)
    gi = (ks - 1).clamp(0, a.shape[1] - 1).unsqueeze(1)
    return srt.gather(1, gi).detach()                     # (T,1)

def soft_kwta_perrow(a, ks):
    return th.sigmoid((a - _thr_perrow(a, ks)) / TEMP)

def hard_kwta_perrow(a, ks):
    thr = _thr_perrow(a, ks)
    return (a >= thr).float()

class SSE(nn.Module):
    def __init__(self, in_dim, n_terms, n_rels):
        super().__init__()
        self.prot = nn.Sequential(nn.Linear(in_dim, 1024), nn.LayerNorm(1024), nn.GELU(),
                                  nn.Dropout(0.1), nn.Linear(1024, N))
        self.pln = nn.LayerNorm(N)
        self.term = nn.Embedding(n_terms, N)
        nn.init.normal_(self.term.weight, std=0.02)
        self.tln = nn.LayerNorm(N)
        self.rel = nn.Parameter(th.zeros(n_rels, N))   # sigmoid -> soft relation mask
    def prot_gate(self, x, hard=False):
        a = self.pln(self.prot(x))
        return hard_kwta_scalar(a, PK) if hard else soft_kwta_scalar(a, PK)
    def term_gate(self, ids=None, hard=False):
        a = self.tln(self.term.weight if ids is None else self.term(ids))
        ks = MTt if ids is None else MTt[ids]
        return hard_kwta_perrow(a, ks) if hard else soft_kwta_perrow(a, ks)
    def coverage(self, gp, gt):
        # gp (B,N) gt (T,N) -> (B,T) soft set containment coverage
        return (gp @ gt.t()) / (gt.sum(1) + 1e-6)

def train_one(seed):
    th.manual_seed(seed); np.random.seed(seed)
    net = SSE(Xsub.shape[1], n_terms, n_rels).to(DEV)
    opt = th.optim.Adam(net.parameters(), lr=LR)
    # split
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
            S = net.coverage(gp, gt).clamp(1e-6, 1 - 1e-6)          # (B,T)
            w = th.where(yb > 0, POS_W, 1.0)
            loss_bce = F.binary_cross_entropy(S, yb, weight=w)
            # margin: true positives > hardest negatives
            neg = S.masked_fill(yb > 0, -1.0)
            hn = neg.topk(min(HARDNEG, S.shape[1]), dim=1).values     # (B,H)
            pos_mean = (S * yb).sum(1) / (yb.sum(1) + 1e-6)
            loss_marg = th.relu(MARGIN - pos_mean.unsqueeze(1) + hn).mean()
            # axiom containment on current term gates
            # containment is the CHILD's duty: child C must cover parent D's required features.
            # Detach the parent gate (anchored by its own BCE) so only C rises -> no "push D down" escape.
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
        # val AUC
        net.eval()
        with th.no_grad():
            gt = net.term_gate()
            preds = []
            for s in range(0, len(Xv), 1024):
                gp = net.prot_gate(Xv[s:s + 1024])
                preds.append(net.coverage(gp, gt).cpu().numpy())
            P = np.concatenate(preds).ravel(); Ytrue = Yv.numpy().ravel()
            pos = np.where(Ytrue > 0)[0]; ne = np.where(Ytrue == 0)[0]
            negs = np.random.default_rng(0).choice(ne, min(len(pos) * 20, len(ne)), replace=False)
            sel = np.concatenate([pos, negs]); auc = roc_auc_score(Ytrue[sel], P[sel])
        if auc > best_auc:
            best_auc = auc; best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        log(f"  seed {seed} ep {ep}: loss {tot/nb:.4f} val_auc {auc:.4f} (best {best_auc:.4f})")
    th.save(best_state, MDIR / f"sse_{seed}.th")
    return best_auc

# ---------------- train ensemble ----------------
prog["stage"] = "training"; save()
aucs = {}
for seed in range(K_SEEDS):
    a = train_one(seed); aucs[seed] = a
    prog["models"][str(seed)] = {"val_auc": a}; save()
    log(f"model {seed} done val_auc {a:.4f} | elapsed {time.time()-t0:.0f}s")
prog["config"]["mean_val_auc"] = float(np.mean(list(aucs.values())))
save()

# ---------------- load ensemble ----------------
nets = []
for seed in range(K_SEEDS):
    net = SSE(Xsub.shape[1], n_terms, n_rels).to(DEV)
    net.load_state_dict(th.load(MDIR / f"sse_{seed}.th")); net.eval(); nets.append(net)

# ---------------- (a) axiom satisfaction (hard set containment) ----------------
def hard_term_codes(net):
    with th.no_grad():
        return net.term_gate(hard=True).cpu().numpy().astype(bool)   # (T,N)
ax_report = {}
with th.no_grad():
    for name, NFt, is_ex in [("nf1_subsumption", nf1, False), ("nf4_existential", nf4, True)]:
        if not NFt: continue
        ratios = []; full = 0
        codes = hard_term_codes(nets[0])
        # relation mask: keep the PM strongest dims (matches the soft mask's role as a sparse selector)
        rel_mask = None
        if is_ex:
            rw = th.sigmoid(nets[0].rel).cpu().numpy()
            rel_mask = np.zeros_like(rw, dtype=bool)
            for r in range(rw.shape[0]):
                rel_mask[r, np.argsort(-rw[r])[:PM]] = True
        for tup in NFt:
            if is_ex:
                C, r, D = tup; supD = codes[D] & rel_mask[r]
            else:
                C, D = tup; supD = codes[D]
            supC = codes[C]; nd = supD.sum()
            if nd == 0: continue
            inter = (supD & supC).sum()
            ratios.append(inter / nd); full += int(inter == nd)
        # random-pair floor: shuffle the "parent" index -> containment expected by chance
        rr = np.random.default_rng(7); Dperm = rr.permutation(len(NFt)); rand_ratios = []
        for a_i, tup in enumerate(NFt):
            C = tup[0]; D = NFt[Dperm[a_i]][-1]
            supD = codes[D]
            if is_ex: supD = supD & rel_mask[NFt[Dperm[a_i]][1]]
            nd = supD.sum()
            if nd == 0: continue
            rand_ratios.append((supD & codes[C]).sum() / nd)
        ax_report[name] = {"n_pairs": len(ratios), "mean_containment": float(np.mean(ratios)),
                           "frac_fully_contained": full / max(1, len(ratios)),
                           "random_pair_floor": float(np.mean(rand_ratios)) if rand_ratios else None}
        log(f"  AXIOM {name}: mean_containment {ax_report[name]['mean_containment']:.3f} "
            f"frac_full {ax_report[name]['frac_fully_contained']:.3f} "
            f"(random floor {ax_report[name]['random_pair_floor']:.3f}) (seed0 hard codes)")

# ---------------- (d) interpretability smoke ----------------
def overlap(a, b, codes, cmap):
    if a not in cmap or b not in cmap: return None
    ca, cb = codes[cmap[a]], codes[cmap[b]]
    inter = (ca & cb).sum(); union = (ca | cb).sum()
    return {"overlap": int(inter), "jaccard": float(inter / max(1, union))}
codes0 = hard_term_codes(nets[0])
interp = {"term_bits_min": int(M_T.min()), "term_bits_max": int(M_T.max()),
          "term_bits_mean": float(M_T.mean())}
pairs = {"glycolysis~gluconeogenesis (co-process, carb metab)": ("GO:0006096", "GO:0006094"),
         "glycolysis~translation (unrelated)": ("GO:0006096", "GO:0006412"),
         "glycolysis~TCA cycle (co-process, energy)": ("GO:0006096", "GO:0006099"),
         "translation~rRNA processing (co-process, ribo)": ("GO:0006412", "GO:0006364"),
         "translation~fatty acid beta-ox (unrelated)": ("GO:0006412", "GO:0006635")}
for lab, (x, y) in pairs.items():
    o = overlap(x, y, codes0, term_idx)
    interp[lab] = o
    if o: log(f"  INTERP {lab}: jaccard {o['jaccard']:.3f} overlap {o['overlap']}/{PM}")
# random baseline: mean jaccard over random term pairs
rngp = np.random.default_rng(1); js = []
for _ in range(2000):
    i, j = rngp.integers(0, n_terms, 2)
    if i == j: continue
    inter = (codes0[i] & codes0[j]).sum(); union = (codes0[i] | codes0[j]).sum()
    js.append(inter / max(1, union))
interp["random_pair_mean_jaccard"] = float(np.mean(js))
log(f"  INTERP random-pair mean jaccard {interp['random_pair_mean_jaccard']:.3f}")

# ---------------- inference: eval LK/PK x BP vocab, ensemble min/avg/max of SOFT coverage ----------------
prog["stage"] = "inference"; save()
def score_eval(prot_accs):
    ridx_ = [acc_idx[a] for a in prot_accs]
    X = (np.ascontiguousarray(EMB[ridx_].astype(np.float32)) - mu) / sd
    Xd = th.tensor(X)
    per = []
    with th.no_grad():
        for net in nets:
            gt = net.term_gate()   # soft term gates
            outs = []
            for s in range(0, len(Xd), 256):
                gp = net.prot_gate(Xd[s:s + 256].to(DEV))
                outs.append(net.coverage(gp, gt).cpu().numpy())
            per.append(np.concatenate(outs).astype(np.float32))
    st = np.stack(per)
    return {"avg": st.mean(0), "min": st.min(0), "max": st.max(0)}

for cell, fn in [("LK", "groundtruth_LK.tsv"), ("PK", "groundtruth_PK.tsv")]:
    gt = pd.read_csv(GTDIR / fn, sep="\t")
    targets = sorted(gt[gt.aspect == "P"].EntryID.unique())
    targets = [t for t in targets if t in acc_idx]
    sc = score_eval(targets)
    np.savez_compressed(OUT / f"sse_scores_{cell}.npz",
                        proteins=np.array(targets), terms=np.array(terms),
                        avg=sc["avg"], min=sc["min"], max=sc["max"])
    log(f"{cell}: saved SSE scores {sc['avg'].shape} -> sse_scores_{cell}.npz")

diag = {"config": prog["config"], "val_auc_per_seed": aucs, "axiom_satisfaction": ax_report,
        "interpretability": interp}
json.dump(diag, open(OUT / "sse_diagnostics.json", "w"), indent=1, default=float)
prog["stage"] = "done"; save()
log("SSE TRAIN+INFER DONE")
print(json.dumps(diag, indent=1, default=str))
