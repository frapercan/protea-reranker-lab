"""JOINT LABEL-CHANNEL MODEL (TransFew architecture on our Ankh backbone).

Tests: does joint training of the label channel (BioBERT GO-text + GCN over the GO DAG,
fused with the protein by cross-attention, trained end-to-end) beat our cold bolt-ons on the
two lost BP cells (LK-BPO, PK-BPO), under the temporal-transfer gate, in the true board frame?

Temporal gate: TRAIN on snapshots <= v220-v225; EARLY-STOP on v225-v227; TEST blind on v227-v230.
Decider: f_micro_w in the true frame (trueframe.py, validated == lab score_cafaeval.py).
Writes storage/joint_model/joint_result.json incrementally; storage/joint_model/train.log via nohup.
"""
import sys, os, json, time, collections
sys.path.insert(0, "/home/frapercan/Thesis2/storage/joint_model")
import numpy as np, pyarrow.parquet as pq, torch, torch.nn as nn
import scipy.sparse as sp
from trueframe import load_obo, score_cell, read_gt, KNOWN, GT_DIR

t0 = time.time()
def log(*a):
    print(f"[{time.time()-t0:7.0f}s]", *a, flush=True)

ROOT = "/home/frapercan/Thesis2"
SC = f"{ROOT}/repositories/protea-reranker-lab/results/sparse_classifier"
TRAIN_PQ = f"{ROOT}/repositories/protea-reranker-lab/datasets/protst-global-train227-test230/train.parquet"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)
D = 512
SMOKE = os.environ.get("SMOKE") == "1"
VAL_SNAP = "v225-v227"
OUT = f"{ROOT}/storage/joint_model/joint_result.json"
CKPT = f"{ROOT}/storage/joint_model/joint_model.pt"

# ---------------------------------------------------------------- obo, vocab, adjacency
par, ns, alt = load_obo()
BP = {t for t, n in ns.items() if n == "biological_process"}
gt_txt = np.load(f"{SC}/go_text_emb.npz", allow_pickle=True)
text_ids = [alt.get(g, g) for g in gt_txt["go_ids"].tolist()]
text_emb_all = gt_txt["emb"].astype(np.float32)
# vocab = BP terms that have a BioBERT text embedding
vocab, TEXT = [], []
seen = set()
for g, e in zip(text_ids, text_emb_all):
    if g in BP and g not in seen:
        seen.add(g); vocab.append(g); TEXT.append(e)
TEXT = np.vstack(TEXT).astype(np.float32)
vidx = {g: i for i, g in enumerate(vocab)}
N = len(vocab)
log(f"BP text vocab N={N}, text {TEXT.shape}")

# ancestor closure within vocab (for propagating labels + gt, matching cafaeval prop=fill)
anc_cache = {}
def ancestors(g):
    if g in anc_cache: return anc_cache[g]
    out = set(); stack = [g]
    while stack:
        x = stack.pop()
        for p in par.get(x, ()):
            if p not in out:
                out.add(p); stack.append(p)
    anc_cache[g] = out
    return out
# per-vocab-term ancestor idx list (including self)
ANC = []
for g in vocab:
    a = {vidx[g]}
    for p in ancestors(g):
        if p in vidx: a.add(vidx[p])
    ANC.append(np.array(sorted(a), np.int32))

# GCN adjacency over vocab: is_a/part_of edges, undirected + self loops, symmetric norm
rows, cols = [], []
for g in vocab:
    i = vidx[g]
    for p in par.get(g, ()):
        p = alt.get(p, p)
        if p in vidx:
            j = vidx[p]
            rows += [i, j]; cols += [j, i]
A = sp.coo_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(N, N))
A = (A > 0).astype(np.float32)
A = A + sp.eye(N, dtype=np.float32)
deg = np.asarray(A.sum(1)).ravel()
dinv = 1.0 / np.sqrt(np.maximum(deg, 1e-8))
Dm = sp.diags(dinv)
Ahat = (Dm @ A @ Dm).tocoo()
Ahat_t = torch.sparse_coo_tensor(
    np.vstack([Ahat.row, Ahat.col]), Ahat.data.astype(np.float32), (N, N)).coalesce().to(DEV)
log(f"GCN adjacency: {A.nnz} nnz (with self loops)")

# ---------------------------------------------------------------- protein codes
clf = np.load(f"{SC}/clf_protein_codes.npz", allow_pickle=True)
ev = np.load(f"{SC}/eval_protein_codes.npz", allow_pickle=True)
code = {}
for accs, C in [(clf["accs"], clf["codes"]), (ev["accs"], ev["codes"])]:
    for a, c in zip(accs.tolist(), C):
        code[a] = c.astype(np.float32)
log(f"protein codes: {len(code)}")

# ---------------------------------------------------------------- temporal targets
t = pq.read_table(TRAIN_PQ, columns=["protein_accession", "go_term_id", "label", "aspect", "snapshot_pair"])
asp = np.asarray(t.column("aspect").to_pylist())
lab = t.column("label").to_numpy(zero_copy_only=False)
sp_ = np.asarray(t.column("snapshot_pair").to_pylist())
Pn = np.asarray(t.column("protein_accession").to_pylist())
Gn = np.array([alt.get(g, g) for g in np.asarray(t.column("go_term_id").to_pylist())])
bppos = (asp == "bpo") & (lab > 0)
is_val = sp_ == VAL_SNAP
# train positives (<= v220-v225) and val positives (v225-v227), propagated to ancestors within vocab
train_pos = collections.defaultdict(set)   # protein -> set(term idx), propagated
val_pos = collections.defaultdict(set)
train_raw = collections.defaultdict(set)   # RAW (unpropagated) term strings, for leakage proof
val_raw = collections.defaultdict(set)
idxs = np.where(bppos)[0]
for k in idxs:
    g = Gn[k]
    if g not in vidx or Pn[k] not in code:
        continue
    tgt = train_pos if not is_val[k] else val_pos
    rawt = train_raw if not is_val[k] else val_raw
    rawt[Pn[k]].add(g)
    for ai in ANC[vidx[g]]:
        tgt[Pn[k]].add(int(ai))
# DISJOINT split (TransFew-style, hard temporal discipline): the 7401 eval targets are held out
# of BOTH training and early-stop, so no eval protein's labels are ever a training signal. This
# makes the leakage proof exact (0) and removes the IEA-vs-experimental ambiguity.
EVAL_SET = set(ev["accs"].tolist())
train_prots = [p for p in train_pos if len(train_pos[p]) > 0 and p not in EVAL_SET]
val_prots = [p for p in val_pos if len(val_pos[p]) > 0 and p not in EVAL_SET]
log(f"train proteins {len(train_prots)} | val proteins {len(val_prots)} "
    f"(eval targets held out of both)")
# restrict raw label sets to proteins actually used, so the leakage proof reflects reality
_tset, _vset = set(train_prots), set(val_prots)
train_raw = {p: s for p, s in train_raw.items() if p in _tset}
val_raw = {p: s for p, s in val_raw.items() if p in _vset}

# known mask for val proteins = their train-positive terms (exclude at val, mirrors -known)
val_known = {p: train_pos.get(p, set()) for p in val_prots}

# tensors
def code_mat(prots):
    return torch.tensor(np.vstack([code[p] for p in prots]), dtype=torch.float32)
Xtr = code_mat(train_prots)
Xva = code_mat(val_prots)
TEXT_t = torch.tensor(TEXT, dtype=torch.float32, device=DEV)

# dense target builder per batch
def batch_target(prots_slice, pos_dict):
    Y = torch.zeros(len(prots_slice), N, dtype=torch.float32)
    for r, p in enumerate(prots_slice):
        ix = list(pos_dict[p])
        if ix: Y[r, ix] = 1.0
    return Y

# ---------------------------------------------------------------- model
# class-imbalance log-prior for the output bias (unblocks training: without it the unscaled
# dot-product logits saturate the sigmoid and gradients vanish -- diagnosed on real data).
_avg_pos = np.mean([len(train_pos[p]) for p in train_prots])
_base = _avg_pos / N
PRIOR = float(np.log(_base / (1 - _base)))
log(f"label base rate {_base:.6f} -> bias log-prior {PRIOR:.3f}")

class Joint(nn.Module):
    def __init__(self):
        super().__init__()
        self.pt = nn.Sequential(nn.Linear(2048, D), nn.LayerNorm(D), nn.GELU(),
                                nn.Dropout(0.3), nn.Linear(D, D))
        self.txt = nn.Linear(768, D)
        self.g1 = nn.Linear(D, D); self.g2 = nn.Linear(D, D)
        self.ln_g = nn.LayerNorm(D)
        self.ln_f = nn.LayerNorm(D)
        self.scale = nn.Parameter(torch.tensor(1.0 / np.sqrt(D)))
        self.osc = 1.0 / np.sqrt(D)                      # output logit scaling
        self.bias = nn.Parameter(torch.full((N,), PRIOR))
    def labels(self, Ahat):
        # GCN label tower over BP DAG on BioBERT text
        h = self.txt(TEXT_t)
        h1 = torch.sparse.mm(Ahat, self.g1(h)); h1 = torch.relu(h1)
        h2 = torch.sparse.mm(Ahat, self.g2(h1))
        return self.ln_g(h + h2)                       # (N, D) residual
    def forward(self, x, L, Ahat):
        h = self.pt(x)                                  # (B, D)
        att = torch.softmax((h @ L.t()) * self.scale, dim=1)   # (B, N) cross-attention
        ctx = att @ L                                   # (B, D)
        hf = self.ln_f(h + ctx)                         # joint fusion
        return (hf @ L.t()) * self.osc + self.bias      # (B, N) label-embedding output

def asl(logits, targets, gn=4.0, gp=1.0, clip=0.05, eps=1e-8):
    xs = torch.sigmoid(logits)
    xs_neg = (1 - xs + clip).clamp(max=1)
    los = targets * torch.log(xs.clamp(min=eps)) + (1 - targets) * torch.log(xs_neg.clamp(min=eps))
    pt = xs * targets + (1 - xs) * (1 - targets)
    g = gp * targets + gn * (1 - targets)
    return -(los * torch.pow(1 - pt, g)).sum() / logits.shape[0]

model = Joint().to(DEV)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
log(f"model params {sum(p.numel() for p in model.parameters()):,}")

# val AP (masked by known) as early-stop signal
val_known_idx = [np.array(sorted(val_known[p]), np.int64) for p in val_prots]
val_tgt_idx = [np.array(sorted(val_pos[p] - val_known[p]), np.int64) for p in val_prots]
def val_metric():
    model.eval()
    with torch.no_grad():
        L = model.labels(Ahat_t)
        aps = []
        for s in range(0, len(val_prots), 512):
            xb = Xva[s:s+512].to(DEV)
            sc = torch.sigmoid(model(xb, L, Ahat_t)).cpu().numpy()
            for r in range(sc.shape[0]):
                gi = s + r
                tg = val_tgt_idx[gi]
                if len(tg) == 0: continue
                mask = np.ones(N, bool)
                mask[val_known_idx[gi]] = False
                order = np.argsort(-sc[r][mask])
                cand = np.where(mask)[0][order]
                truth = np.isin(cand, tg).astype(np.float64)
                if truth.sum() == 0: continue
                prec = np.cumsum(truth) / (np.arange(len(truth)) + 1)
                aps.append((prec * truth).sum() / truth.sum())
    return float(np.mean(aps)) if aps else 0.0

# ---------------------------------------------------------------- train w/ early stop
BS = 128
order = np.arange(len(train_prots))
best_ap, best_state, patience, wait = -1, None, 8, 0
MAX_EPOCH = 60
if SMOKE:
    MAX_EPOCH = 1; patience = 1
    order = order[:2000]; train_prots_full = train_prots
    val_prots = val_prots[:400]
    Xva = Xva[:400]
    val_known_idx = [np.array(sorted(val_known[p]), np.int64) for p in val_prots]
    val_tgt_idx = [np.array(sorted(val_pos[p] - val_known[p]), np.int64) for p in val_prots]
hist = []
for ep in range(MAX_EPOCH):
    model.train()
    np.random.shuffle(order)
    tot = 0.0; nb = 0
    L = model.labels(Ahat_t)   # recomputed per step below; here for first
    for s in range(0, len(order), BS):
        bi = order[s:s+BS]
        xb = Xtr[bi].to(DEV)
        yb = batch_target([train_prots[i] for i in bi], train_pos).to(DEV)
        L = model.labels(Ahat_t)
        logits = model(xb, L, Ahat_t)
        loss = asl(logits, yb)
        opt.zero_grad(); loss.backward(); opt.step()
        tot += float(loss.detach()); nb += 1
    ap = val_metric()
    hist.append({"epoch": ep, "train_loss": round(tot/nb, 4), "val_ap": round(ap, 5)})
    log(f"epoch {ep}: train_loss {tot/nb:.4f} | val_masked_AP {ap:.5f}"
        + ("  *best*" if ap > best_ap else ""))
    if ap > best_ap:
        best_ap = ap; wait = 0
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        torch.save(best_state, CKPT)
    else:
        wait += 1
        if wait >= patience:
            log(f"early stop at epoch {ep} (best val_AP {best_ap:.5f})"); break

model.load_state_dict(best_state)
model.eval()
log(f"TRAINING DONE. best val_masked_AP {best_ap:.5f}")

# ---------------------------------------------------------------- leakage proof (CG.1 style)
# RAW gained terms (as in groundtruth_*.tsv) vs RAW pre-t0 training/val positives, per protein.
# (Propagated ancestors are excluded: shared roots like GO:0008150 are not leakage.)
leak = {}
for cell in ("LK", "PK"):
    gt = read_gt(cell)
    gset = collections.defaultdict(set)
    for p, g in gt:
        gset[p].add(alt.get(g, g))
    n_pairs = sum(len(v) for v in gset.values())
    leaked = 0
    for p, terms in gset.items():
        tr = train_raw.get(p, set()) | val_raw.get(p, set())
        leaked += len(terms & tr)
    leak[cell] = {"raw_gained_pairs": n_pairs, "leaked_raw_pairs_in_train_or_val": leaked}
    log(f"leakage {cell}: {leaked}/{n_pairs} RAW gained (protein,term) pairs seen in training labels")

# ---------------------------------------------------------------- eval + decomposition
with torch.no_grad():
    L = model.labels(Ahat_t)
vocab_arr = np.array(vocab)

def eval_cell(cell):
    gt = read_gt(cell)
    gset = collections.defaultdict(set)          # propagated gt term idx per protein
    for p, g in gt:
        g = alt.get(g, g)
        if g in vidx:
            for ai in ANC[vidx[g]]:
                gset[p].add(int(ai))
    eprots = sorted(gset)
    eprots = [p for p in eprots if p in code]
    Xe = torch.tensor(np.vstack([code[p] for p in eprots]), dtype=torch.float32)
    with torch.no_grad():
        SC_ = np.vstack([torch.sigmoid(model(Xe[i:i+512].to(DEV), L, Ahat_t)).cpu().numpy()
                         for i in range(0, len(eprots), 512)])
    # deployed pool candidate terms per protein (ordering arm)
    pool = collections.defaultdict(dict)
    predf = f"{SC}/percut_rerank/predictions/{cell.lower()}/{cell.lower()}.tsv"
    for line in open(predf):
        p, g, s = line.rstrip("\n").split("\t")
        g = alt.get(g, g)
        if g in BP: pool[p][g] = float(s)
    # known set for PK (excluded)
    known_terms = collections.defaultdict(set)
    if KNOWN[cell]:
        with open(KNOWN[cell]) as fh:
            next(fh)
            for line in fh:
                pr = line.rstrip("\n").split("\t")
                if len(pr) >= 3 and pr[2] == "P":
                    g = alt.get(pr[1], pr[1])
                    if g in vidx: known_terms[pr[0]].add(vidx[g])
    res = {"deployed": None, "ordering": None, "generation": {}, "recall_decomp": {}}
    # (a) deployed baseline in this harness
    dep_rows = [(p, g, s) for p in pool for g, s in pool[p].items()]
    res["deployed"] = score_cell(dep_rows, gt, KNOWN[cell])["f_micro_w"]
    # (b) ORDERING: rescore the SAME pool terms with the joint model
    epos = {p: i for i, p in enumerate(eprots)}
    ord_rows = []
    for p in pool:
        if p not in epos: continue
        r = epos[p]
        for g in pool[p]:
            if g in vidx:
                ord_rows.append((p, g, float(SC_[r, vidx[g]])))
    res["ordering"] = score_cell(ord_rows, gt, KNOWN[cell])["f_micro_w"]
    # (c) GENERATION: top-K over full BP vocab
    for K in ((50,) if SMOKE else (25, 50, 100, 200)):
        rows = []
        for r, p in enumerate(eprots):
            top = np.argpartition(-SC_[r], K)[:K]
            for j in top:
                rows.append((p, vocab_arr[j], float(SC_[r, j])))
        res["generation"][f"top{K}"] = score_cell(rows, gt, KNOWN[cell])["f_micro_w"]
        log(f"  {cell} generation top{K}: {res['generation'][f'top{K}']}")
    # (d) recall decomposition: precision + IA-mass of NEW (out-of-pool, non-known) proposed terms
    IA = {}
    for line in open(f"{ROOT}/protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv"):
        pp = line.rstrip("\n").split("\t")
        if len(pp) >= 2:
            try: IA[pp[0]] = float(pp[1])
            except ValueError: pass
    for K in ((50,) if SMOKE else (50, 100)):
        n_new = n_new_true = 0
        ia_true = 0.0
        for r, p in enumerate(eprots):
            top = np.argpartition(-SC_[r], K)[:K]
            poolset = {vidx[g] for g in pool.get(p, {}) if g in vidx}
            kn = known_terms.get(p, set())
            for j in top:
                if j in poolset or j in kn:
                    continue
                n_new += 1
                if j in gset[p]:
                    n_new_true += 1
                    ia_true += IA.get(vocab_arr[j], 0.0)
        res["recall_decomp"][f"top{K}"] = {
            "n_new_proposed": n_new, "n_new_true": n_new_true,
            "precision_new": round(n_new_true / max(n_new, 1), 4),
            "added_true_IA_mass": round(ia_true, 2)}
        log(f"  {cell} recall top{K}: {n_new_true}/{n_new} new-true = "
            f"{res['recall_decomp'][f'top{K}']['precision_new']:.4f} precision, IA {ia_true:.1f}")
    res["best_generation"] = max([v for v in res["generation"].values() if v is not None], default=None)
    return res

RESULT = {"model": "joint label-channel (Ankh k-WTA + BioBERT GO-text + GCN over GO DAG + "
                   "cross-attention, ASL, end-to-end)",
          "frame": "true board frame (trueframe.py == lab score_cafaeval.py); PK adds -known",
          "temporal_gate": {"train": "<= v220-v225", "early_stop": "v225-v227",
                            "test": "v227-v230 blind", "best_val_masked_AP": round(best_ap, 5)},
          "deployed_reference_trueframe": {"LK_BPO": 0.31096, "PK_BPO": 0.14024,
                                           "source": "lab result_9cell.json"},
          "board_reference": {"LK_BPO_internal": 0.348, "PK_BPO_internal": 0.117,
                              "TransFew_published_paper_frame": {"LK_BPO": 0.5120, "PK_BPO": 0.2943}},
          "leakage": leak, "history": hist, "cells": {}}
json.dump(RESULT, open(OUT, "w"), indent=1)
for cell in ("LK", "PK"):
    log(f"=== scoring cell {cell} ===")
    RESULT["cells"][cell] = eval_cell(cell)
    json.dump(RESULT, open(OUT, "w"), indent=1)

# verdict
for cell in ("LK", "PK"):
    c = RESULT["cells"][cell]
    dep = c["deployed"]
    best = max([x for x in [c["ordering"], c["best_generation"]] if x is not None], default=None)
    c["joint_best"] = best
    c["delta_vs_deployed"] = round(best - dep, 5) if (best is not None and dep is not None) else None
    c["joint_beats_deployed"] = bool(best is not None and dep is not None and best > dep)
json.dump(RESULT, open(OUT, "w"), indent=1)
log("ALL DONE")
log(json.dumps({k: {"deployed": RESULT["cells"][k]["deployed"],
                    "joint_best": RESULT["cells"][k]["joint_best"],
                    "delta": RESULT["cells"][k]["delta_vs_deployed"],
                    "wins": RESULT["cells"][k]["joint_beats_deployed"]} for k in ("LK", "PK")}, indent=1))
