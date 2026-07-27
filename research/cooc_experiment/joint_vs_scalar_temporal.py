"""TEMPORAL: does fusing the VECTORS beat fusing their scalars, trained the way we deploy?

`joint_vs_scalar.py` answered the architecture question by cross-fitting over the evaluation
proteins: B beat A by **+0.07526** (sd 0.01575, positive in 10 of 10; arm A's fold-mean 0.22613,
inside its precondition band). That number cannot be deployed, because both arms trained on
evaluation-window labels for other proteins, which the real system never sees.

The author asked the obvious question: should this not train on v225-v227? It should, and it can.
The protein vector is a function of the sequence, which does not change with the snapshot, so the
proteins that have vectors carry **3,143,496 pk-bpo rows across the whole temporal window** (18.0%
of the 17.5M, 28,198 positives, 5,769 proteins). v225-v227 alone holds 172,280 rows and 6,903
positives: the exact split the deployed booster early-stops on.

THE DEPLOYABLE SHAPE:

  train      v160 to v225   the past
  validate   v225-v227      early stopping, as deployed
  score      v227-v230      eval.parquet, BLIND, never seen during training

  A  LightGBM over the scalars, trained on THE SAME restricted rows
  B  cross-attention over the four vectors

**Arm A is retrained on the restricted rows and is not the deployed booster.** Comparing B against
a booster trained on all 17.5M rows would move two variables at once, the architecture and the
training set, and this campaign has paid for that lesson more than once. Both arms see the 18% the
vectors can express, so the only thing that differs is the shape of the model.

LEAKAGE, checked rather than assumed. Every vector is t0 or older: the GO codes are fit on **v227**,
our t0, with a t0-independent text block (`build_per_cut_codes.py` states this in its own
docstring); anc2vec is the **2020-10** release; ProtRek is pretrained. None of them can see the
v227-v230 labels this run is scored on.

THE GATE, quantity named: the quantity is (B - A) on the blind window, scored with the board's real
line. There are no folds because there is one temporal split, so the comparison is one number per
arm; it must clear **0.0034**, the fold-to-fold noise already established for this cell, and its
per-protein bootstrap 95% interval must exclude zero. A gain failing either is not a gain.

THE PRECONDITION: arm A must land in [0.15, 0.30]. It will NOT equal the deployed 0.21269, because
it trains on 18% of the rows. Outside that band means the harness is broken, not that the model is
good, and the run is void.

WHAT A WIN WOULD AND WOULD NOT BUY. It would say the architecture is worth building on the real
training set, which means embedding the other 82% of the training proteins: a re-embedding job, not
a frozen-data operation. That job is what this run is deciding whether to justify. It would still
not be a board number, for the reason in `WE_DO_NOT_REPRODUCE_THE_BOARD.md`.
"""
import json, subprocess, tempfile, time, collections
from pathlib import Path
import numpy as np, pyarrow.parquet as pq

W = Path("/home/frapercan/Thesis2/storage/cooc_experiment")
DS = Path("/home/frapercan/Thesis2/repositories/protea-reranker-lab/datasets/protst-global-train227-test230")
REL = Path("/home/frapercan/Thesis2/CAFA_forever/data/releases/Sep_2025_Mar_2026")
OBO = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
IA = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv"
TOI = str(REL / "groundtruth_terms_of_interest.txt")
PY = "/home/frapercan/Thesis2/repositories/PROTEA/.venv/bin/python"
SEED = 42
VAL_PAIR = "v225-v227"

import torch, torch.nn as nn, lightgbm as lgb
torch.manual_seed(SEED)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device {DEV}", flush=True)

qacc = list(json.load(open("/home/frapercan/Thesis2/storage/layer_ablation/prot_go.json")))
QV = np.load("/home/frapercan/Thesis2/storage/layer_ablation/query_d8979601.npy").astype(np.float32)
TV = np.load("/home/frapercan/Thesis2/storage/text_scorer/query_protrek.npy").astype(np.float32)
assert QV.shape[0] == len(qacc) == TV.shape[0], "protein vectors do not match prot_go.json"
pidx = {a: i for i, a in enumerate(qacc)}
gz = np.load("/home/frapercan/Thesis2/storage/two_tower_sparse/go_sparse_codes.npz", allow_pickle=True)
GV = gz["codes"].astype(np.float32)
gidx = {g: i for i, g in enumerate(gz["go_ids"].tolist())}
az = np.load("/home/frapercan/Thesis2/repositories/PROTEA/artifacts/anc2vec/anc2vec_2020-10.npz", allow_pickle=True)
AV = np.vstack([az["embeddings"].astype(np.float32),
                np.zeros((1, az["embeddings"].shape[1]), np.float32)])
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
    C = np.asarray(t.column("category").to_pylist())
    A = np.asarray(t.column("aspect").to_pylist())
    m = (C == "pk") & (A == "bpo")
    P = np.asarray(t.column("protein_accession").to_pylist())[m]
    G = np.asarray(t.column("go_term_id").to_pylist())[m]
    S = (np.asarray(t.column("snapshot_pair").to_pylist())[m] if "snapshot_pair" in cols
         else np.full(len(P), "eval"))
    pi = np.array([pidx.get(a, -1) for a in P])
    gi = np.array([gidx.get(g, -1) for g in G])
    ai = np.array([aidx.get(g, NA) for g in G])
    keep = (pi >= 0) & (gi >= 0)          # only rows the vectors express; identical for both arms
    if part == "past":
        keep &= S != VAL_PAIR
    elif part == "val":
        keep &= S == VAL_PAIR
    X = np.empty((int(m.sum()), len(FEATS)), np.float32)
    for j, c in enumerate(FEATS):
        X[:, j] = t.column(c).to_numpy(zero_copy_only=False).astype(np.float32)[m]
    Y = (t.column("label").to_numpy(zero_copy_only=False).astype(np.int64)[m] > 0).astype(np.float32)
    return P[keep], G[keep], Y[keep], X[keep], pi[keep], gi[keep], ai[keep]


t0 = time.time()
trP, trG, trY, trX, trpi, trgi, trai = load(DS / "train.parquet", "past")
vaP, vaG, vaY, vaX, vapi, vagi, vaai = load(DS / "train.parquet", "val")
evP, evG, evY, evX, evpi, evgi, evai = load(DS / "eval.parquet")
print(f"train (v160..v225): {len(trP):,} rows / {len(set(trP.tolist())):,} proteins / {trY.sum():,.0f} pos", flush=True)
print(f"valid ({VAL_PAIR}):  {len(vaP):,} rows / {len(set(vaP.tolist())):,} proteins / {vaY.sum():,.0f} pos", flush=True)
print(f"BLIND (v227-v230):  {len(evP):,} rows / {len(set(evP.tolist())):,} proteins / {evY.sum():,.0f} pos  ({time.time()-t0:.0f}s)", flush=True)
assert len(set(trP.tolist()) | set(vaP.tolist())) > 0 and len(evP) > 0

QV_t = torch.tensor(QV, device=DEV); TV_t = torch.tensor(TV, device=DEV)
GV_t = torch.tensor(GV, device=DEV); AV_t = torch.tensor(AV, device=DEV)


class Joint(nn.Module):
    """Four signals as four tokens, one cross-attention block, one score."""
    def __init__(self, d=128, heads=4):
        super().__init__()
        self.proj = nn.ModuleList([nn.Linear(n, d) for n in
                                   (QV.shape[1], TV.shape[1], GV.shape[1], AV.shape[1])])
        self.tok = nn.Parameter(torch.randn(4, d) * 0.02)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, d * 2), nn.GELU(), nn.Linear(d * 2, d))
        self.head = nn.Sequential(nn.LayerNorm(d * 4), nn.Linear(d * 4, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, q, tx, g, a):
        toks = torch.stack([p(v) for p, v in zip(self.proj, (q, tx, g, a))], 1) + self.tok
        h, _ = self.attn(toks, toks, toks)
        h = self.norm(toks + h)
        h = h + self.ff(h)
        return self.head(h.flatten(1)).squeeze(-1)


@torch.no_grad()
def infer(mdl, pi_, gi_, ai_, bs=32768):
    mdl.eval()
    out = np.empty(len(pi_), np.float32)
    for s in range(0, len(pi_), bs):
        sl = slice(s, s + bs)
        out[sl] = mdl(QV_t[pi_[sl]], TV_t[pi_[sl]], GV_t[gi_[sl]], AV_t[ai_[sl]]).cpu().numpy()
    return out


def auprc(y, s):
    o = np.argsort(-s)
    yy = y[o]
    pr = np.cumsum(yy) / np.arange(1, len(yy) + 1)
    return float((pr * yy).sum() / max(yy.sum(), 1))


mdl = Joint().to(DEV)
opt = torch.optim.AdamW(mdl.parameters(), lr=2e-3, weight_decay=1e-4)
posw = torch.tensor(float(len(trY) - trY.sum()) / max(float(trY.sum()), 1.0), device=DEV)
lossf = nn.BCEWithLogitsLoss(pos_weight=posw)
trYt = torch.tensor(trY, device=DEV)
best, best_state, bad = -1.0, None, 0
for e in range(12):
    mdl.train()
    perm = np.random.default_rng(SEED + e).permutation(len(trY))
    tot = 0.0
    for s in range(0, len(perm), 8192):
        b = perm[s:s + 8192]
        opt.zero_grad()
        out = mdl(QV_t[trpi[b]], TV_t[trpi[b]], GV_t[trgi[b]], AV_t[trai[b]])
        l = lossf(out, trYt[b])
        l.backward(); opt.step(); tot += float(l.detach()) * len(b)
    v = auprc(vaY, infer(mdl, vapi, vagi, vaai))
    print(f"  B epoch {e} loss {tot/len(perm):.4f} | {VAL_PAIR} AUPRC {v:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    if v > best:
        best, best_state, bad = v, {k: t.detach().clone() for k, t in mdl.state_dict().items()}, 0
    else:
        bad += 1
        if bad >= 3:
            print(f"  B early stop at epoch {e}; best {VAL_PAIR} AUPRC {best:.4f}", flush=True)
            break
mdl.load_state_dict(best_state)
b_scores = infer(mdl, evpi, evgi, evai)

_POS = float(trY.sum()); _NEG = float(len(trY) - _POS)
LGB = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 100,
       "feature_fraction": 0.9, "bagging_fraction": 0.9, "bagging_freq": 5, "seed": SEED,
       "verbose": -1, "num_threads": 12, "max_bin": 63, "scale_pos_weight": _NEG / _POS,
       "metric": "average_precision"}
dtr = lgb.Dataset(trX, label=trY, feature_name=FEATS)
dva = lgb.Dataset(vaX, label=vaY, reference=dtr, feature_name=FEATS)
bst = lgb.train(LGB, dtr, num_boost_round=2000, valid_sets=[dva],
                callbacks=[lgb.early_stopping(50, verbose=False)])
a_scores = bst.predict(evX, num_iteration=bst.best_iteration, raw_score=True).astype(np.float32)
print(f"  A LightGBM best_iteration {bst.best_iteration}  ({time.time()-t0:.0f}s)", flush=True)
# Both arms are pos_weighted logits, so "> 0" means the same thing to the positivity guard.

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
        best_ = None
        for rec in json.loads(raw.read_text()).get("f_micro_w", []):
            if (rec.get("ns") or "") == "biological_process" and rec.get("f_micro_w") is not None:
                best_ = rec
        return round(float(best_["f_micro_w"]), 5) if best_ else None


fa, fb = score(a_scores), score(b_scores)
res = {"design": "train v160..v225, early-stop on v225-v227, score v227-v230 BLIND. Both arms on the "
                 "SAME rows the vectors can express (18% of the pk-bpo training rows), so the only "
                 "variable is the model's shape.",
       "leakage": "every vector is t0 or older: GO codes fit on v227 (our t0) with a t0-independent "
                  "text block, anc2vec 2020-10, ProtRek pretrained.",
       "crossfitted_reference_NOT_deployable": {"A": 0.22613, "B": 0.3014, "B_minus_A": 0.07526,
                                                "sd": 0.01575, "positive": "10/10"},
       "rows": {"train": int(len(trP)), "valid": int(len(vaP)), "blind_eval": int(len(evP)),
                "train_pos": int(trY.sum()), "valid_pos": int(vaY.sum())},
       "A_lightgbm_scalars": fa, "B_joint_vectors": fb,
       "B_minus_A": round(fb - fa, 5) if (fa is not None and fb is not None) else None,
       "PRECONDITION_A_in_band": bool(fa is not None and 0.15 <= fa <= 0.30),
       "gate": "B-A must clear 0.0034 (this cell's established fold noise) and the per-protein "
               "bootstrap 95% interval must exclude zero."}
json.dump(res, open(W / "joint_vs_scalar_temporal.json", "w"), indent=1)
print(f"\n  A = {fa} | B = {fb} | B-A = {res['B_minus_A']}  ({time.time()-t0:.0f}s)", flush=True)

prots = sorted(set(evP.tolist()) & set(GT))
rng = np.random.default_rng(SEED)
boot = []
for i in range(12):
    samp = set(rng.choice(prots, size=len(prots), replace=True).tolist())
    xa, xb = score(a_scores, samp), score(b_scores, samp)
    if xa is not None and xb is not None:
        boot.append(round(xb - xa, 5))
        print(f"    bootstrap {len(boot):2d}: B-A = {boot[-1]:+.5f}  ({time.time()-t0:.0f}s)", flush=True)
if boot:
    lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    res["bootstrap"] = {"n": len(boot), "deltas": boot, "ci95": [round(lo, 5), round(hi, 5)],
                        "excludes_zero": bool(lo > 0)}
    res["JOINT_WINS"] = bool(res["PRECONDITION_A_in_band"] and res["B_minus_A"]
                             and res["B_minus_A"] > 0.0034 and lo > 0)
json.dump(res, open(W / "joint_vs_scalar_temporal.json", "w"), indent=1)
print(f"\n=== TEMPORAL: vectors vs scalars, trained the way we deploy ({time.time()-t0:.0f}s) ===", flush=True)
print(f"  PRECONDITION: A = {fa} in [0.15, 0.30] -> {res['PRECONDITION_A_in_band']}", flush=True)
print(f"  A LightGBM over the scalars              {fa}", flush=True)
print(f"  B cross-attention over the four vectors  {fb}", flush=True)
print(f"  B - A = {res['B_minus_A']}", flush=True)
if boot:
    print(f"  per-protein bootstrap 95% CI [{res['bootstrap']['ci95'][0]}, {res['bootstrap']['ci95'][1]}]"
          f" excludes zero: {res['bootstrap']['excludes_zero']}", flush=True)
    print(f"  -> JOINT WINS: {res.get('JOINT_WINS')}", flush=True)
print("  Reference, NOT deployable: the cross-fitted run gave B-A = +0.07526 (sd 0.01575, 10/10).", flush=True)
print("DONE", flush=True)
