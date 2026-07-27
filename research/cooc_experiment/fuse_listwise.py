"""Fuse the vectors WITH the scalars under a listwise objective, against the deployed recipe.

WHERE THIS COMES FROM. `joint_full.py` showed the joint model beating a binary-objective GBDT by a
gated +0.02053 at full coverage, and then showed the thing that matters: **B scored 0.18844 while
the deployed reranker scores 0.21269 on the same cell and the same line.** Arm A there was not the
deployed booster; it was a binary LightGBM, and the deployed lambdarank is about 0.045 better. So
the joint model was winning a race the deployed system was not running in.

Two things were missing from B, and both were deliberate exclusions of that design: the **listwise
objective** and the **72 scalar features**. B saw three vectors and not one feature. TransFew never
chose between summaries and vectors; it fuses both. This does that.

THE THREE ARMS, because "fuse" hides two variables and moving them together would not say which paid:

  A  LightGBM lambdarank over the 72 scalars           the deployed recipe's shape. THE ANCHOR.
  B  neural listwise over the 72 scalars ONLY          isolates the ARCHITECTURE at equal input
  C  neural listwise over the scalars AND the vectors  isolates what the VECTORS add on top

  C - B  is the question: given a model that already has every scalar, do the raw vectors add?
  B - A  is the control: does the architecture alone move anything at equal input?

THE PRECONDITION, and it is the strict one this time. **Arm A must land within 0.01 of 0.21269**,
the deployed recipe's whole-cell figure on the board's line. Every previous run in this file's
lineage used a weaker baseline and had to caveat it; this one either reproduces the system we
actually run or it is void. If A comes back at 0.16 again, the objective was the difference all
along and nothing below it is worth reading.

THE GATE, quantity named: the quantity is (C - B) on the blind window. It must clear **0.0034**,
this cell's established fold noise, and its per-protein bootstrap 95% interval must exclude zero.
**And C must beat arm A**, because beating a neural baseline while losing to the deployed booster is
what the last run already did and it does not buy a system.

THE LISTWISE LOSS. lambdarank optimises the order within a protein and is invariant to per-protein
monotone rescaling, which is exactly the mismatch chapter 6 diagnoses. The neural arms use a
per-protein softmax cross-entropy (ListNet), the closest listwise analogue that trains stably: for
each protein, its candidates compete for the mass its true terms carry. Early stopping on
v225-v227, as deployed.

Temporal split as deployed: train v160..v225, early-stop v225-v227, score v227-v230 BLIND. Every
vector is t0 or older. Both neural arms share the positivity guard, so "> 0" means the same thing.
"""
import json, subprocess, tempfile, time, collections, gc
from pathlib import Path
import numpy as np, pyarrow.parquet as pq

W = Path("/home/frapercan/Thesis2/storage/cooc_experiment")
DS = Path("/home/frapercan/Thesis2/repositories/protea-reranker-lab/datasets/protst-global-train227-test230")
REL = Path("/home/frapercan/Thesis2/CAFA_forever/data/releases/Sep_2025_Mar_2026")
OBO = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
IA = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv"
TOI = str(REL / "groundtruth_terms_of_interest.txt")
PY = "/home/frapercan/Thesis2/repositories/PROTEA/.venv/bin/python"
SEED, VAL_PAIR = 42, "v225-v227"
DEPLOYED = 0.21269          # build_submission.py, guard_only, whole cell, the board's line

import torch, torch.nn as nn, lightgbm as lgb
torch.manual_seed(SEED)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device {DEV}", flush=True)

FULL = np.load(W / "d8979601_full" / "codes.npy")
pidx = {a: i for i, a in enumerate(json.load(open(W / "d8979601_full" / "accs.json")))}
gz = np.load("/home/frapercan/Thesis2/storage/two_tower_sparse/go_sparse_codes.npz", allow_pickle=True)
GV = gz["codes"].astype(np.float32)
gidx = {g: i for i, g in enumerate(gz["go_ids"].tolist())}
az = np.load("/home/frapercan/Thesis2/repositories/PROTEA/artifacts/anc2vec/anc2vec_2020-10.npz", allow_pickle=True)
AV = np.vstack([az["embeddings"].astype(np.float32), np.zeros((1, az["embeddings"].shape[1]), np.float32)])
aidx = {g: i for i, g in enumerate(az["go_ids"].tolist())}
NA = len(AV) - 1

EX = {"protein_accession", "go_term_id", "label", "category", "snapshot_pair", "qualifier",
      "evidence_code", "taxonomic_relation", "aspect"}
FEATS = [c for c in pq.ParquetFile(DS / "eval.parquet").schema_arrow.names if c not in EX]


def load(path, part=None):
    have = set(pq.ParquetFile(path).schema_arrow.names)
    cols = [c for c in dict.fromkeys(FEATS + ["protein_accession", "go_term_id", "category",
                                              "aspect", "label", "snapshot_pair"]) if c in have]
    t = pq.read_table(path, columns=cols)
    C = np.asarray(t.column("category").to_pylist()); A = np.asarray(t.column("aspect").to_pylist())
    m = (C == "pk") & (A == "bpo")
    P = np.asarray(t.column("protein_accession").to_pylist())[m]
    G = np.asarray(t.column("go_term_id").to_pylist())[m]
    S = (np.asarray(t.column("snapshot_pair").to_pylist())[m] if "snapshot_pair" in cols
         else np.full(len(P), "eval"))
    pi = np.array([pidx.get(a, -1) for a in P]); gi = np.array([gidx.get(g, -1) for g in G])
    ai = np.array([aidx.get(g, NA) for g in G])
    keep = (pi >= 0) & (gi >= 0)
    if part == "past":
        keep &= S != VAL_PAIR
    elif part == "val":
        keep &= S == VAL_PAIR
    X = np.empty((int(m.sum()), len(FEATS)), np.float32)
    for j, c in enumerate(FEATS):
        X[:, j] = t.column(c).to_numpy(zero_copy_only=False).astype(np.float32)[m]
    Y = (t.column("label").to_numpy(zero_copy_only=False).astype(np.int64)[m] > 0).astype(np.float32)
    Sk = S[keep]
    del t; gc.collect()
    return P[keep], G[keep], Y[keep], X[keep], pi[keep], gi[keep], ai[keep], Sk


t0 = time.time()
trP, trG, trY, trX, trpi, trgi, trai, trS = load(DS / "train.parquet", "past")
vaP, vaG, vaY, vaX, vapi, vagi, vaai, _ = load(DS / "train.parquet", "val")
evP, evG, evY, evX, evpi, evgi, evai, _ = load(DS / "eval.parquet")
print(f"train {len(trP):,}/{trY.sum():,.0f}pos | valid {len(vaP):,}/{vaY.sum():,.0f}pos | BLIND {len(evP):,}/{evY.sum():,.0f}pos  ({time.time()-t0:.0f}s)", flush=True)

# NaN -> 0 for the neural arms only; LightGBM routes NaN natively and must keep it (one variable:
# the arms differ in shape, not in what they are told about missingness).
Xn_tr = np.nan_to_num(trX, nan=0.0, posinf=0.0, neginf=0.0)
Xn_va = np.nan_to_num(vaX, nan=0.0, posinf=0.0, neginf=0.0)
Xn_ev = np.nan_to_num(evX, nan=0.0, posinf=0.0, neginf=0.0)
mu, sd = Xn_tr.mean(0), Xn_tr.std(0) + 1e-6
Xn_tr = (Xn_tr - mu) / sd; Xn_va = (Xn_va - mu) / sd; Xn_ev = (Xn_ev - mu) / sd

FULL_t = torch.tensor(FULL, device=DEV); GV_t = torch.tensor(GV, device=DEV); AV_t = torch.tensor(AV, device=DEV)


class Net(nn.Module):
    """Tokens compete in one cross-attention block. `use_vec` decides whether the raw vectors
    are among them, which is the only difference between arm B and arm C."""
    def __init__(self, use_vec, d=128, heads=4):
        super().__init__()
        self.use_vec = use_vec
        dims = [len(FEATS)] + ([FULL.shape[1], GV.shape[1], AV.shape[1]] if use_vec else [])
        self.proj = nn.ModuleList([nn.Linear(n, d) for n in dims])
        self.tok = nn.Parameter(torch.randn(len(dims), d) * 0.02)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, d * 2), nn.GELU(), nn.Linear(d * 2, d))
        self.head = nn.Sequential(nn.LayerNorm(d * len(dims)), nn.Linear(d * len(dims), d),
                                  nn.GELU(), nn.Linear(d, 1))

    def forward(self, x, q=None, g=None, a=None):
        vs = [x] + ([q, g, a] if self.use_vec else [])
        toks = torch.stack([p(v) for p, v in zip(self.proj, vs)], 1) + self.tok
        h, _ = self.attn(toks, toks, toks)
        h = self.norm(toks + h)
        h = h + self.ff(h)
        return self.head(h.flatten(1)).squeeze(-1)


# per-protein groups, so the listwise loss has lists to work with
def groups(P):
    d = collections.defaultdict(list)
    for i, p in enumerate(P):
        d[p].append(i)
    return {k: np.array(v) for k, v in d.items()}


TRG = groups(trP)
tr_keys = [k for k, v in TRG.items() if trY[v].sum() > 0]     # a list with no positive teaches nothing
print(f"training proteins with at least one positive: {len(tr_keys):,} of {len(TRG):,}", flush=True)
Xtr_t = torch.tensor(Xn_tr, device=DEV)


def auprc(y, s):
    o = np.argsort(-s); yy = y[o]
    return float((np.cumsum(yy) / np.arange(1, len(yy) + 1) * yy).sum() / max(yy.sum(), 1))


@torch.no_grad()
def infer(mdl, X, pi_, gi_, ai_, bs=16384):
    mdl.eval()
    Xt = torch.tensor(X, device=DEV)
    out = np.empty(len(pi_), np.float32)
    for s in range(0, len(pi_), bs):
        sl = slice(s, s + bs)
        out[sl] = (mdl(Xt[sl], FULL_t[pi_[sl]], GV_t[gi_[sl]], AV_t[ai_[sl]]) if mdl.use_vec
                   else mdl(Xt[sl])).cpu().numpy()
    del Xt; torch.cuda.empty_cache()
    return out


def train_listwise(use_vec, tag, n_prot=64, cand=192, epochs=8):
    """ListNet: each protein's candidates compete for the mass its true terms carry."""
    mdl = Net(use_vec).to(DEV)
    opt = torch.optim.AdamW(mdl.parameters(), lr=1e-3, weight_decay=1e-4)
    best, bstate, bad = -1.0, None, 0
    rng = np.random.default_rng(SEED)
    for e in range(epochs):
        mdl.train()
        order = rng.permutation(len(tr_keys)); tot = n = 0.0
        for s in range(0, len(order), n_prot):
            batch = [TRG[tr_keys[i]] for i in order[s:s + n_prot]]
            idx, off = [], [0]
            for g in batch:
                pos = g[trY[g] > 0]
                neg = g[trY[g] == 0]
                if len(neg) > cand - len(pos):
                    neg = rng.choice(neg, cand - len(pos), replace=False)
                sel = np.concatenate([pos, neg])
                idx.append(sel); off.append(off[-1] + len(sel))
            flat = np.concatenate(idx)
            out = (mdl(Xtr_t[flat], FULL_t[trpi[flat]], GV_t[trgi[flat]], AV_t[trai[flat]])
                   if use_vec else mdl(Xtr_t[flat]))
            yb = torch.tensor(trY[flat], device=DEV)
            loss = 0.0
            for j in range(len(batch)):
                sl = slice(off[j], off[j + 1])
                p = torch.log_softmax(out[sl], 0)
                q = yb[sl] / yb[sl].sum()
                loss = loss - (q * p).sum()
            loss = loss / len(batch)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()) * len(batch); n += len(batch)
        v = auprc(vaY, infer(mdl, Xn_va, vapi, vagi, vaai))
        print(f"  {tag} epoch {e} loss {tot/n:.4f} | {VAL_PAIR} AUPRC {v:.4f}  ({time.time()-t0:.0f}s)", flush=True)
        if v > best:
            best, bstate, bad = v, {k: t.detach().clone() for k, t in mdl.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 2:
                print(f"  {tag} early stop at epoch {e}; best AUPRC {best:.4f}", flush=True)
                break
    mdl.load_state_dict(bstate)
    return mdl


GT = collections.defaultdict(set)
with (REL / "groundtruth_PK.tsv").open() as fh:
    next(fh)
    for line in fh:
        p_, t_, a_ = line.rstrip("\n").split("\t")[:3]
        if a_ == "P":
            GT[p_].add(t_)

DRIVER = '''
import json, signal
from cafaeval.evaluation import cafa_eval
signal.signal(signal.SIGTERM, signal.SIG_DFL)
df,dfs=cafa_eval("{obo}","{pd}","{gt}",ia="{ia}",prop="fill",norm="cafa",no_orphans=True,
    toi_file="{toi}",max_terms=None,th_step=0.01,n_cpu=4,weighted_only=False)
out={{}}
for k,v in dfs.items(): out[k]=v.reset_index().to_dict(orient="records")
json.dump(out,open("{o}","w"),default=str)
'''


def score(s, prots=None):
    keep = s > 0
    if prots is not None:
        keep = keep & np.isin(evP, list(prots))
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "pd"; d.mkdir()
        with (d / "p.tsv").open("w") as fh:
            for p_, g_, v in zip(evP[keep], evG[keep], s[keep]):
                fh.write(f"{p_}\t{g_}\t{v:.6f}\n")
        gt = Path(td) / "gt.tsv"
        with gt.open("w") as fh:
            for p_ in (prots if prots is not None else GT):
                for g_ in GT.get(p_, ()):
                    fh.write(f"{p_}\t{g_}\n")
        raw = Path(td) / "r.json"; drv = Path(td) / "d.py"
        drv.write_text(DRIVER.format(obo=OBO, pd=str(d), gt=str(gt), ia=IA, toi=TOI, o=str(raw)))
        r = subprocess.run([PY, str(drv)], capture_output=True, text=True, timeout=7200)
        if r.returncode != 0:
            return None
        best = None
        for rec in json.loads(raw.read_text()).get("f_micro_w", []):
            if (rec.get("ns") or "") == "biological_process" and rec.get("f_micro_w") is not None:
                best = rec
        return round(float(best["f_micro_w"]), 5) if best else None


# ---- A: the deployed recipe's shape. lambdarank over the scalars. --------------
def sortgrp(P, X, Y, extra=None):
    o = np.argsort(P, kind="stable")
    _, sz = np.unique(P[o], return_counts=True)
    return o, sz


o, sz = sortgrp(trP, trX, trY)
ov, szv = sortgrp(vaP, vaX, vaY)
LGB = {"objective": "lambdarank", "metric": ["ndcg"], "ndcg_eval_at": [5, 10], "label_gain": [0, 1],
       "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 100, "feature_fraction": 0.9,
       "bagging_fraction": 0.9, "bagging_freq": 5, "seed": SEED, "verbose": -1, "num_threads": 12,
       "max_bin": 63, "two_round": True, "force_col_wise": True}
dtr = lgb.Dataset(trX[o], label=trY[o].astype(int), group=sz, feature_name=FEATS)
dva = lgb.Dataset(vaX[ov], label=vaY[ov].astype(int), group=szv, reference=dtr, feature_name=FEATS)
bst = lgb.train(LGB, dtr, num_boost_round=3000, valid_sets=[dva],
                callbacks=[lgb.early_stopping(50, verbose=False)])
a_s = bst.predict(evX, num_iteration=bst.best_iteration).astype(np.float32)
fa = score(a_s)
print(f"\n  A lambdarank over the scalars: {fa}  (best_iter {bst.best_iteration}, deployed {DEPLOYED})  ({time.time()-t0:.0f}s)", flush=True)

res = {"arms": {"A_lambdarank_scalars": fa}, "deployed_anchor": DEPLOYED,
       "PRECONDITION_A_reproduces_deployed": bool(fa is not None and abs(fa - DEPLOYED) < 0.01),
       "design": "A lambdarank over the 72 scalars (the deployed shape); B neural listwise over the "
                 "SAME scalars (isolates architecture); C neural listwise over scalars AND vectors "
                 "(isolates what the vectors add on top). C-B is the question, B-A the control.",
       "gate": "C-B must clear 0.0034 with a bootstrap CI excluding zero, AND C must beat A."}
json.dump(res, open(W / "fuse_listwise.json", "w"), indent=1)
if not res["PRECONDITION_A_reproduces_deployed"]:
    print(f"  PRECONDITION FAILED: A={fa} is not within 0.01 of the deployed {DEPLOYED}.", flush=True)
    print("  The run is VOID: without a baseline that reproduces the system we run, nothing below", flush=True)
    print("  it can be read as beating that system. Reporting and stopping.", flush=True)
    print("DONE", flush=True)
    raise SystemExit(0)

mb = train_listwise(False, "B scalars-only")
b_scores_ev = infer(mb, Xn_ev, evpi, evgi, evai)
fb = score(b_scores_ev)
del mb; torch.cuda.empty_cache()
mc = train_listwise(True, "C scalars+vectors")
c_s = infer(mc, Xn_ev, evpi, evgi, evai)
fc = score(c_s)
res["arms"].update({"B_listwise_scalars_only": fb, "C_listwise_scalars_plus_vectors": fc})
res["C_minus_B_the_vectors"] = round(fc - fb, 5) if (fb is not None and fc is not None) else None
res["B_minus_A_the_architecture"] = round(fb - fa, 5) if (fb is not None) else None
res["C_minus_A_vs_the_deployed_shape"] = round(fc - fa, 5) if (fc is not None) else None
json.dump(res, open(W / "fuse_listwise.json", "w"), indent=1)
print(f"\n  A {fa} | B {fb} | C {fc}", flush=True)

del mc; torch.cuda.empty_cache()

# The gate needs an interval on C - B, and one temporal split gives no folds, so the interval comes
# from resampling the blind window's proteins. Both arms are scored on the SAME resample, so the
# pairing is preserved and the difference is what is bootstrapped.
b_s_ev = None
if fb is not None and fc is not None:
    prots = sorted(set(evP.tolist()) & set(GT))
    rng = np.random.default_rng(SEED)
    boot = []
    for i in range(10):
        samp = set(rng.choice(prots, size=len(prots), replace=True).tolist())
        xb, xc = score(b_scores_ev, samp), score(c_s, samp)
        if xb is not None and xc is not None:
            boot.append(round(xc - xb, 5))
            print(f"    bootstrap {len(boot):2d}: C-B = {boot[-1]:+.5f}  ({time.time()-t0:.0f}s)", flush=True)
    if boot:
        lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
        res["bootstrap_C_minus_B"] = {"n": len(boot), "deltas": boot,
                                      "ci95": [round(lo, 5), round(hi, 5)], "excludes_zero": bool(lo > 0)}
        res["VECTORS_ADD"] = bool(res["C_minus_B_the_vectors"] is not None
                                  and res["C_minus_B_the_vectors"] > 0.0034 and lo > 0
                                  and fc is not None and fa is not None and fc > fa)
        json.dump(res, open(W / "fuse_listwise.json", "w"), indent=1)
print(f"\n=== fusing the vectors with the scalars, under a listwise objective ({time.time()-t0:.0f}s) ===", flush=True)
print(f"  PRECONDITION: A={fa} vs deployed {DEPLOYED} -> {res['PRECONDITION_A_reproduces_deployed']}", flush=True)
print(f"  A lambdarank, scalars only        {fa}", flush=True)
print(f"  B listwise neural, scalars only   {fb}   (B-A = {res['B_minus_A_the_architecture']}, the architecture)", flush=True)
print(f"  C listwise neural, scalars+vecs   {fc}   (C-B = {res['C_minus_B_the_vectors']}, THE VECTORS)", flush=True)
print(f"  C vs the deployed shape           {res['C_minus_A_vs_the_deployed_shape']}", flush=True)
print("DONE", flush=True)
