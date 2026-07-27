"""Does fusing the signal VECTORS beat fusing their scalar summaries? The surviving hypothesis.

THE PREMISE, CORRECTED FIRST. The campaign has been describing its own attempts as "grafts", as if
each new signal were bolted onto the outside of the model. That is not what happened. The deployed
reranker already fuses every signal TransFew fuses, jointly, in one model: `eval.parquet` carries
24 sequence/kNN columns, 12 ontology-structure columns (`anc2vec_*`), 6 alignment, 3 text
(`protst_*`) and 2 classifier. LightGBM sees them all at once. So "+0.0016 for ProtST" was not a
graft failing; it was a joint model failing to use a signal.

**The real difference from TransFew is dimensionality, not jointness.** We compress each signal to
one or three scalars BEFORE fusing (`protst_vote_fraction`, `anc2vec_neighbor_cos`,
`neighbor_vote_fraction`). TransFew fuses the vectors themselves with cross-attention. Whatever a
signal carries beyond its summary statistic, our reranker has never been shown.

THE QUESTION, therefore: given exactly the same rows and the same candidate pool, does a model that
attends over the raw vectors rank better than a gradient-boosted tree over their summaries?

  A  LightGBM over the 81 scalar features        the shape we have always used
  B  cross-attention over the four raw vectors   the shape TransFew uses

  protein  `storage/layer_ablation/query_d8979601.npy`      (7401, 2048)  the champion encoder
  GO term  `storage/two_tower_sparse/go_sparse_codes.npz`   (39906, 1024) + `go_ids`
  text     `storage/text_scorer/query_protrek.npy`          (7401, 1024)
  DAG      `PROTEA/artifacts/anc2vec/anc2vec_2020-10.npz`   (44261, 200) + `go_ids`

Coverage, checked rather than assumed: the protein vectors cover **99.0%** of the pk-bpo cell
(4,410 of 4,455) and the GO vectors cover **100%** of its terms.

WHAT THIS CANNOT ANSWER, and the limit is hard. The protein vectors cover only **8.2%** of
`train.parquet`'s 83,775 proteins, so a joint model **cannot be trained on the temporal window**
the deployed system trains on. Both arms are therefore cross-fitted over the evaluation proteins
themselves: trained on nine disjoint protein folds, scored blind on the tenth. **That makes this a
probe of the architecture, not a deployable result.** A win here would say the shape is worth
building; it would not say what the board would score, and reaching that would need the training
proteins re-embedded, which is not a frozen-data operation.

**Arm A is retrained on the same folds, not the deployed booster.** Comparing a cross-fitted neural
model against a temporally-trained GBDT would move two variables at once, and the whole campaign
has been a lesson in what that costs.

THE GATE, quantity named: the quantity is (B - A) held out, averaged over ten folds. It must be
positive in at least 8 of 10 AND exceed the fold-to-fold standard deviation. Below that it is
noise, and the tidy story that "they learn them jointly" dies like the five before it.

THE PRECONDITION: arm A's fold-mean must land in [0.18, 0.30]. It will NOT equal the deployed
0.21584, because A is trained in-distribution on evaluation proteins and that booster was not; a
figure outside that band means the harness is broken, not that the model is good.
"""
import json, subprocess, tempfile, time, collections, statistics
from pathlib import Path
import numpy as np, pyarrow.parquet as pq

W = Path("/home/frapercan/Thesis2/storage/cooc_experiment")
DS = Path("/home/frapercan/Thesis2/repositories/protea-reranker-lab/datasets/protst-global-train227-test230")
REL = Path("/home/frapercan/Thesis2/CAFA_forever/data/releases/Sep_2025_Mar_2026")
OBO = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
IA = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv"
TOI = str(REL / "groundtruth_terms_of_interest.txt")
PY = "/home/frapercan/Thesis2/repositories/PROTEA/.venv/bin/python"
SEED, K = 42, 10

import torch, torch.nn as nn, lightgbm as lgb
torch.manual_seed(SEED)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device {DEV}", flush=True)

# ---- the rows -------------------------------------------------------------------
EX = {"protein_accession", "go_term_id", "label", "category", "snapshot_pair", "qualifier",
      "evidence_code", "taxonomic_relation", "aspect"}
FEATS = [c for c in pq.ParquetFile(DS / "eval.parquet").schema_arrow.names if c not in EX]
cols = FEATS + ["protein_accession", "go_term_id", "category", "aspect", "label"]
t = pq.read_table(DS / "eval.parquet", columns=list(dict.fromkeys(cols)))
C = np.asarray(t.column("category").to_pylist())
A = np.asarray(t.column("aspect").to_pylist())
m = (C == "pk") & (A == "bpo")
P = np.asarray(t.column("protein_accession").to_pylist())[m]
G = np.asarray(t.column("go_term_id").to_pylist())[m]
Y = (t.column("label").to_numpy(zero_copy_only=False).astype(np.int64)[m] > 0).astype(np.float32)
X = np.empty((int(m.sum()), len(FEATS)), dtype=np.float32)
for j, c in enumerate(FEATS):
    X[:, j] = t.column(c).to_numpy(zero_copy_only=False).astype(np.float32)[m]
print(f"pk-bpo rows {len(P):,} | positives {Y.sum():,.0f} ({Y.mean():.2%}) | {len(FEATS)} scalar features", flush=True)

# ---- the vectors ----------------------------------------------------------------
qacc = list(json.load(open("/home/frapercan/Thesis2/storage/layer_ablation/prot_go.json")))
QV = np.load("/home/frapercan/Thesis2/storage/layer_ablation/query_d8979601.npy").astype(np.float32)
TV = np.load("/home/frapercan/Thesis2/storage/text_scorer/query_protrek.npy").astype(np.float32)
assert QV.shape[0] == len(qacc) == TV.shape[0], "protein vector rows do not match prot_go.json"
pidx = {a: i for i, a in enumerate(qacc)}
gz = np.load("/home/frapercan/Thesis2/storage/two_tower_sparse/go_sparse_codes.npz", allow_pickle=True)
GV = gz["codes"].astype(np.float32)
gidx = {g: i for i, g in enumerate(gz["go_ids"].tolist())}
az = np.load("/home/frapercan/Thesis2/repositories/PROTEA/artifacts/anc2vec/anc2vec_2020-10.npz", allow_pickle=True)
AV = az["embeddings"].astype(np.float32)
aidx = {g: i for i, g in enumerate(az["go_ids"].tolist())}

pi = np.array([pidx.get(a, -1) for a in P])
gi = np.array([gidx.get(g, -1) for g in G])
ai = np.array([aidx.get(g, -1) for g in G])
ok = (pi >= 0) & (gi >= 0)
print(f"vector coverage: protein {(pi>=0).mean():.1%} | GO {(gi>=0).mean():.1%} | anc2vec {(ai>=0).mean():.1%} "
      f"| rows usable by BOTH arms {ok.mean():.1%}", flush=True)
# one variable: both arms see EXACTLY the rows the vectors can express
P, G, Y, X, pi, gi, ai = P[ok], G[ok], Y[ok], X[ok], pi[ok], gi[ok], ai[ok]
AVz = np.vstack([AV, np.zeros((1, AV.shape[1]), np.float32)])
ai = np.where(ai >= 0, ai, len(AV))
print(f"rows kept for both arms: {len(P):,} over {len(set(P.tolist())):,} proteins", flush=True)

QV_t = torch.tensor(QV, device=DEV); TV_t = torch.tensor(TV, device=DEV)
GV_t = torch.tensor(GV, device=DEV); AV_t = torch.tensor(AVz, device=DEV)


class Joint(nn.Module):
    """Four signals as four tokens, one cross-attention block, one score.

    This is the smallest architecture that expresses TransFew's claim: each signal keeps its own
    dimensionality until the model decides how much of it to use, rather than being reduced to a
    summary before it is ever seen.
    """
    def __init__(self, d=128, heads=4):
        super().__init__()
        self.proj = nn.ModuleList([nn.Linear(n, d) for n in (QV.shape[1], TV.shape[1], GV.shape[1], AVz.shape[1])])
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


def train_joint(tr, ep=6, bs=8192):
    mdl = Joint().to(DEV)
    opt = torch.optim.AdamW(mdl.parameters(), lr=2e-3, weight_decay=1e-4)
    pos = Y[tr].sum(); w = torch.tensor((len(tr) - pos) / max(pos, 1), device=DEV)
    lossf = nn.BCEWithLogitsLoss(pos_weight=w)
    yt = torch.tensor(Y[tr], device=DEV)
    for e in range(ep):
        perm = np.random.default_rng(SEED + e).permutation(len(tr))
        tot = 0.0
        for s in range(0, len(tr), bs):
            b = tr[perm[s:s + bs]]; bt = perm[s:s + bs]
            opt.zero_grad()
            out = mdl(QV_t[pi[b]], TV_t[pi[b]], GV_t[gi[b]], AV_t[ai[b]])
            l = lossf(out, yt[bt]); l.backward(); opt.step(); tot += float(l) * len(b)
        print(f"      epoch {e} loss {tot/len(tr):.4f}", flush=True)
    return mdl


@torch.no_grad()
def pred_joint(mdl, idx, bs=32768):
    mdl.eval()
    out = np.empty(len(idx), np.float32)
    for s in range(0, len(idx), bs):
        b = idx[s:s + bs]
        out[s:s + bs] = mdl(QV_t[pi[b]], TV_t[pi[b]], GV_t[gi[b]], AV_t[ai[b]]).cpu().numpy()
    return out


# Both arms must mean the same thing by "> 0", because the positivity guard cuts there and
# `prop=fill` reads an unsubmitted cell as an abstention. B is a logit trained with pos_weight, so
# its zero sits at the reweighted prior. A therefore takes the SAME reweighting and is read as a
# RAW MARGIN, not a probability.
#
# The first version of this script took A's probability and subtracted 0.5. With a 2.47% base rate
# and no reweighting, LightGBM's probabilities cluster near 0.02, so A submitted almost nothing
# while B submitted a sensible fraction: the guard was cutting the two arms in different places,
# which is two variables rather than one. It produced A=0.08475 against B=0.30415 on fold 0, a
# spurious +0.2194. The precondition band [0.18, 0.30] caught it and the run was voided.
_POS = float(Y.sum()); _NEG = float(len(Y) - _POS)
LGB = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 100,
       "feature_fraction": 0.9, "bagging_fraction": 0.9, "bagging_freq": 5, "seed": SEED,
       "verbose": -1, "num_threads": 12, "max_bin": 63, "scale_pos_weight": _NEG / _POS}

GT = collections.defaultdict(set)
with (REL / "groundtruth_PK.tsv").open() as fh:
    next(fh)
    for line in fh:
        p_, t_, a_ = line.rstrip("\n").split("\t")[:3]
        if a_ == "P":
            GT[p_].add(t_)
prots = sorted(set(P.tolist()) & set(GT))
rng = np.random.default_rng(SEED); rng.shuffle(prots)
FOLDS = [set(prots[i::K]) for i in range(K)]
print(f"gt-covered proteins with vectors: {len(prots):,} -> {K} folds", flush=True)

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


def score(idx, s, prots_):
    """The positivity guard is forced: fill uses 0.0 as the sentinel for 'not submitted'."""
    keep = s > 0
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "pd"; d.mkdir()
        with (d / "p.tsv").open("w") as fh:
            for p_, g_, v in zip(P[idx][keep], G[idx][keep], s[keep]):
                fh.write(f"{p_}\t{g_}\t{v:.6f}\n")
        gt = Path(td) / "gt.tsv"
        with gt.open("w") as fh:
            for p_ in prots_:
                for g_ in GT[p_]:
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


t0 = time.time()
res = {"question": "does fusing the signal VECTORS beat fusing their scalar summaries?",
       "premise_corrected": "the deployed reranker ALREADY fuses all four signals jointly (24 kNN + "
                            "12 anc2vec + 6 alignment + 3 protst + 2 classifier columns). The "
                            "difference from TransFew is DIMENSIONALITY, not jointness: we compress "
                            "each signal to 1-3 scalars before fusing; it attends over the vectors.",
       "hard_limit": "the protein vectors cover 8.2% of train.parquet, so a joint model CANNOT be "
                     "trained on the temporal window. Both arms are cross-fitted over the EVALUATION "
                     "proteins. This probes the ARCHITECTURE and cannot produce a deployable number.",
       "gate": "(B-A) held out must be positive in >=8/10 folds AND exceed the fold-to-fold sd.",
       "precondition": "arm A's fold-mean must land in [0.18, 0.30]; it will NOT equal the deployed "
                       "0.21584 because A is retrained in-distribution.",
       "folds": []}

for i in range(K):
    te = np.where(np.isin(P, list(FOLDS[i])))[0]
    tr = np.where(~np.isin(P, list(FOLDS[i])))[0]
    b = lgb.train(LGB, lgb.Dataset(X[tr], label=Y[tr], feature_name=FEATS), num_boost_round=400)
    a_s = b.predict(X[te], raw_score=True)             # RAW MARGIN: "> 0" now means the same for both arms
    mdl = train_joint(tr)
    b_s = pred_joint(mdl, te)
    fa = score(te, np.asarray(a_s, np.float32), FOLDS[i])
    fb = score(te, b_s, FOLDS[i])
    rec = {"fold": i, "A_lightgbm_scalars": fa, "B_joint_vectors": fb,
           "B_minus_A": round(fb - fa, 5) if (fa is not None and fb is not None) else None}
    res["folds"].append(rec)
    print(f"  fold {i}: A={fa} B={fb}  **B-A={rec['B_minus_A']}**  ({time.time()-t0:.0f}s)", flush=True)
    json.dump(res, open(W / "joint_vs_scalar_v2.json", "w"), indent=1)
    del mdl; torch.cuda.empty_cache()

d = [x["B_minus_A"] for x in res["folds"] if x["B_minus_A"] is not None]
av = [x["A_lightgbm_scalars"] for x in res["folds"] if x["A_lightgbm_scalars"] is not None]
am = statistics.mean(av) if av else None
ok_pre = am is not None and 0.18 <= am <= 0.30
res["verdict"] = {
    "A_fold_mean": round(am, 5) if am else None, "PRECONDITION_holds": ok_pre,
    "B_fold_mean": round(statistics.mean([x["B_joint_vectors"] for x in res["folds"] if x["B_joint_vectors"] is not None]), 5) if d else None,
    "B_minus_A_mean": round(statistics.mean(d), 5) if d else None,
    "sd": round(statistics.stdev(d), 5) if len(d) > 1 else None,
    "positive": f"{sum(1 for x in d if x > 0)}/{len(d)}",
    "JOINT_WINS": bool(ok_pre and d and len(d) > 1 and statistics.mean(d) > statistics.stdev(d)
                       and sum(1 for x in d if x > 0) >= 8)}
json.dump(res, open(W / "joint_vs_scalar_v2.json", "w"), indent=1)
v = res["verdict"]
print(f"\n=== vectors vs scalars, {time.time()-t0:.0f}s ===", flush=True)
print(f"  PRECONDITION: A fold-mean {v['A_fold_mean']} in [0.18, 0.30] -> {ok_pre}", flush=True)
print(f"  A LightGBM over 81 scalars : {v['A_fold_mean']}", flush=True)
print(f"  B cross-attention over the four vectors : {v['B_fold_mean']}", flush=True)
print(f"  B-A mean {v['B_minus_A_mean']} sd {v['sd']} positive {v['positive']}", flush=True)
print(f"  -> JOINT WINS: {v['JOINT_WINS']}", flush=True)
print("  FALSE => the vectors carry nothing the summaries did not, and the sixth tidy story dies.", flush=True)
print("DONE", flush=True)
