"""The joint model at FULL coverage. Does the data volume explain the +0.020 / +0.075 gap?

WHAT CHANGED. `joint_vs_scalar_temporal.py` trained on 18% of the pk-bpo rows and reported that the
protein vectors "cover only 8.2% of train.parquet". That premise was inherited from
`storage/layer_ablation/query_d8979601.npy`, a 7,401-protein leftover of the layer-ablation
experiment, and it was false about the cause. `d8979601` is the DEPLOYED encoder and the live
database already holds 527,426 of its codes: `export_d8979601.py` pulled **84,370 x 2048** of them
read-only in 109 seconds and covers **100.0%** of the 51,096 pk-bpo training proteins. The leftover
covered 11.3%. Nothing ever needed computing.

THE QUESTION THIS ANSWERS. The same architecture gained **+0.07526** trained in distribution and
**+0.02032** trained the way we deploy. The obvious reading was that the model is data-hungry and
the temporal arm was starved at 18%. That is a hypothesis, and this campaign has killed six of them,
each one tidy and each one wrong. So it gets measured.

  coverage_18   the rows the leftover .npy could express   (the previous result)
  coverage_100  every row the exported codes can express

  A  LightGBM over the scalars, retrained at each coverage
  B  cross-attention over the tokens, retrained at each coverage

THREE TOKENS, NOT FOUR, AND THIS IS DELIBERATE. The ProtRek text vectors still cover only the 7,401
query proteins, so keeping the text token would change two things at once: more rows AND a token
that is absent on most of them. Both arms therefore drop text and use protein / GO code / anc2vec,
all three at 100%. **The 18% arm here is re-run with three tokens too**, so the comparison against
it is a comparison of data volume and nothing else. It is NOT directly comparable to the +0.02032
of the four-token run, and is not reported as if it were.

TEMPORAL SPLIT, as deployed: train v160..v225, early-stop on v225-v227, score v227-v230 blind.
Every vector is t0 or older, so none can see the window being scored.

THE GATE, quantity named: two quantities. (1) (B-A) at each coverage must clear 0.0034, this cell's
established fold noise. (2) The data-volume claim is (B-A at 100%) minus (B-A at 18%): if that is
not positive, **more data does not explain the gap** and the in-distribution +0.075 comes from
something else, which would be worth knowing and would kill the seventh tidy story.

PRECONDITION: arm A at 100% must land in [0.15, 0.30]. It trains on all the rows the parquet holds
for this cell, so it should sit nearer the deployed 0.21269 than the 18% arm's 0.16186 did; outside
the band means broken, not good.
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

import torch, torch.nn as nn, lightgbm as lgb
torch.manual_seed(SEED)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device {DEV}", flush=True)

FULL = np.load(W / "d8979601_full" / "codes.npy")
FACC = json.load(open(W / "d8979601_full" / "accs.json"))
pidx = {a: i for i, a in enumerate(FACC)}
OLD = set(json.load(open("/home/frapercan/Thesis2/storage/layer_ablation/prot_go.json")))
gz = np.load("/home/frapercan/Thesis2/storage/two_tower_sparse/go_sparse_codes.npz", allow_pickle=True)
GV = gz["codes"].astype(np.float32)
gidx = {g: i for i, g in enumerate(gz["go_ids"].tolist())}
az = np.load("/home/frapercan/Thesis2/repositories/PROTEA/artifacts/anc2vec/anc2vec_2020-10.npz", allow_pickle=True)
AV = np.vstack([az["embeddings"].astype(np.float32), np.zeros((1, az["embeddings"].shape[1]), np.float32)])
aidx = {g: i for i, g in enumerate(az["go_ids"].tolist())}
NA = len(AV) - 1
print(f"exported codes {FULL.shape} | GO codes {GV.shape} | anc2vec {AV.shape}", flush=True)

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
    pi = np.array([pidx.get(a, -1) for a in P])
    gi = np.array([gidx.get(g, -1) for g in G])
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
    del t; gc.collect()
    return P[keep], G[keep], Y[keep], X[keep], pi[keep], gi[keep], ai[keep]


t0 = time.time()
trP, trG, trY, trX, trpi, trgi, trai = load(DS / "train.parquet", "past")
vaP, vaG, vaY, vaX, vapi, vagi, vaai = load(DS / "train.parquet", "val")
evP, evG, evY, evX, evpi, evgi, evai = load(DS / "eval.parquet")
print(f"train v160..v225 {len(trP):,} rows / {trY.sum():,.0f} pos | valid {VAL_PAIR} {len(vaP):,} / {vaY.sum():,.0f} pos"
      f" | BLIND {len(evP):,} / {evY.sum():,.0f} pos  ({time.time()-t0:.0f}s)", flush=True)

FULL_t = torch.tensor(FULL, device=DEV)
GV_t = torch.tensor(GV, device=DEV)
AV_t = torch.tensor(AV, device=DEV)


class Joint(nn.Module):
    """Three signals as three tokens, one cross-attention block, one score."""
    def __init__(self, d=128, heads=4):
        super().__init__()
        self.proj = nn.ModuleList([nn.Linear(n, d) for n in (FULL.shape[1], GV.shape[1], AV.shape[1])])
        self.tok = nn.Parameter(torch.randn(3, d) * 0.02)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, d * 2), nn.GELU(), nn.Linear(d * 2, d))
        self.head = nn.Sequential(nn.LayerNorm(d * 3), nn.Linear(d * 3, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, q, g, a):
        toks = torch.stack([p(v) for p, v in zip(self.proj, (q, g, a))], 1) + self.tok
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
        out[sl] = mdl(FULL_t[pi_[sl]], GV_t[gi_[sl]], AV_t[ai_[sl]]).cpu().numpy()
    return out


def auprc(y, s):
    o = np.argsort(-s); yy = y[o]
    return float((np.cumsum(yy) / np.arange(1, len(yy) + 1) * yy).sum() / max(yy.sum(), 1))


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


res = {"correction": "the 18% figure came from a 7,401-protein leftover .npy, not from missing data. "
                     "d8979601 is the DEPLOYED encoder and the DB already held 527,426 of its codes; "
                     "the export covers 100.0% of the 51,096 pk-bpo training proteins in 109s.",
       "design": "three tokens (protein / GO code / anc2vec), all at 100%; the text token is dropped "
                 "at BOTH coverages so the only variable is data volume. Temporal split as deployed.",
       "gate": "(B-A) must clear 0.0034 at each coverage; the data-volume claim is (B-A at 100%) "
               "minus (B-A at 18%) and must be positive or more data does not explain the gap.",
       "coverages": {}}

for tag, restrict in (("coverage_18", True), ("coverage_100", False)):
    sel = np.isin(trP, list(OLD)) if restrict else np.ones(len(trP), bool)
    vsel = np.isin(vaP, list(OLD)) if restrict else np.ones(len(vaP), bool)
    print(f"\n=== {tag}: {sel.sum():,} train rows ({sel.mean():.1%}), {trY[sel].sum():,.0f} positives ===", flush=True)

    mdl = Joint().to(DEV)
    opt = torch.optim.AdamW(mdl.parameters(), lr=2e-3, weight_decay=1e-4)
    ys = trY[sel]
    posw = torch.tensor(float(len(ys) - ys.sum()) / max(float(ys.sum()), 1.0), device=DEV)
    lossf = nn.BCEWithLogitsLoss(pos_weight=posw)
    yt = torch.tensor(ys, device=DEV)
    ppi, pgi, pai = trpi[sel], trgi[sel], trai[sel]
    best, bstate, bad = -1.0, None, 0
    for e in range(12):
        mdl.train()
        perm = np.random.default_rng(SEED + e).permutation(len(ys))
        tot = 0.0
        for s in range(0, len(perm), 8192):
            b = perm[s:s + 8192]
            opt.zero_grad()
            l = lossf(mdl(FULL_t[ppi[b]], GV_t[pgi[b]], AV_t[pai[b]]), yt[b])
            l.backward(); opt.step(); tot += float(l.detach()) * len(b)
        v = auprc(vaY[vsel], infer(mdl, vapi[vsel], vagi[vsel], vaai[vsel]))
        print(f"  B epoch {e} loss {tot/len(perm):.4f} | {VAL_PAIR} AUPRC {v:.4f}  ({time.time()-t0:.0f}s)", flush=True)
        if v > best:
            best, bstate, bad = v, {k: t.detach().clone() for k, t in mdl.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 3:
                print(f"  B early stop at epoch {e}; best AUPRC {best:.4f}", flush=True)
                break
    mdl.load_state_dict(bstate)
    b_s = infer(mdl, evpi, evgi, evai)
    del mdl; torch.cuda.empty_cache()

    _P = float(ys.sum()); _N = float(len(ys) - _P)
    LGB = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 100,
           "feature_fraction": 0.9, "bagging_fraction": 0.9, "bagging_freq": 5, "seed": SEED,
           "verbose": -1, "num_threads": 12, "max_bin": 63, "scale_pos_weight": _N / _P,
           "metric": "average_precision"}
    dtr = lgb.Dataset(trX[sel], label=ys, feature_name=FEATS)
    dva = lgb.Dataset(vaX[vsel], label=vaY[vsel], reference=dtr, feature_name=FEATS)
    bst = lgb.train(LGB, dtr, num_boost_round=2000, valid_sets=[dva],
                    callbacks=[lgb.early_stopping(50, verbose=False)])
    a_s = bst.predict(evX, num_iteration=bst.best_iteration, raw_score=True).astype(np.float32)
    print(f"  A LightGBM best_iteration {bst.best_iteration}  ({time.time()-t0:.0f}s)", flush=True)

    fa, fb = score(a_s), score(b_s)
    res["coverages"][tag] = {"train_rows": int(sel.sum()), "train_positives": int(ys.sum()),
                             "A_lightgbm": fa, "B_joint": fb,
                             "B_minus_A": round(fb - fa, 5) if (fa is not None and fb is not None) else None}
    print(f"  {tag}: A={fa} B={fb} **B-A={res['coverages'][tag]['B_minus_A']}**", flush=True)
    json.dump(res, open(W / "joint_full.json", "w"), indent=1)

d18 = res["coverages"]["coverage_18"]["B_minus_A"]
d100 = res["coverages"]["coverage_100"]["B_minus_A"]
a100 = res["coverages"]["coverage_100"]["A_lightgbm"]
res["verdict"] = {
    "B_minus_A_at_18": d18, "B_minus_A_at_100": d100,
    "data_volume_effect": round(d100 - d18, 5) if (d18 is not None and d100 is not None) else None,
    "PRECONDITION_A100_in_band": bool(a100 is not None and 0.15 <= a100 <= 0.30),
    "MORE_DATA_EXPLAINS_THE_GAP": bool(d18 is not None and d100 is not None and d100 > d18),
    "each_clears_noise": bool(d18 is not None and d100 is not None and d18 > 0.0034 and d100 > 0.0034),
    "reference_4token_18pct_temporal": 0.02032, "reference_4token_crossfitted": 0.07526}
json.dump(res, open(W / "joint_full.json", "w"), indent=1)
v = res["verdict"]
print(f"\n=== the joint model at full coverage ({time.time()-t0:.0f}s) ===", flush=True)
print(f"  PRECONDITION: A at 100% = {a100} in [0.15, 0.30] -> {v['PRECONDITION_A100_in_band']}", flush=True)
print(f"  B-A at  18% of the rows : {d18}", flush=True)
print(f"  B-A at 100% of the rows : {d100}", flush=True)
print(f"  data-volume effect      : {v['data_volume_effect']}", flush=True)
print(f"  -> MORE DATA EXPLAINS THE GAP: {v['MORE_DATA_EXPLAINS_THE_GAP']}", flush=True)
print("  FALSE => the in-distribution +0.075 comes from something other than volume, and the", flush=True)
print("           seventh tidy story of this campaign dies.", flush=True)
print("DONE", flush=True)
