"""Reproduce fuse_listwise's arm A (lambdarank -> 0.21969) in isolation, saving predictions.

My soft_cafaeval run got arm A = 0.13852, not 0.21969, so its precondition (rightly) voided it. Same
gt, same driver, same 72 features, same params, equivalent training. This isolates what differs by
copying fuse_listwise's arm A byte for byte, saving the predictions so scoring can be re-checked
cheaply. If this gives 0.219, the delta is in my data prep; if 0.138, it is in the params/scoring.
"""
import json, collections, subprocess, tempfile, time, gc
from pathlib import Path
import numpy as np, pyarrow.parquet as pq, lightgbm as lgb

W = Path("/home/frapercan/Thesis2/storage/cooc_experiment")
DS = Path("/home/frapercan/Thesis2/repositories/protea-reranker-lab/datasets/protst-global-train227-test230")
REL = Path("/home/frapercan/Thesis2/CAFA_forever/data/releases/Sep_2025_Mar_2026")
T0D = Path("/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025")
OBO, IA = str(T0D / "go-basic.obo"), str(T0D / "IA.tsv")
TOI = str(REL / "groundtruth_terms_of_interest.txt")
PY = "/home/frapercan/Thesis2/repositories/PROTEA/.venv/bin/python"
VAL_PAIR, SEED, DEPLOYED = "v225-v227", 42, 0.21269
t0 = time.time()

# the vector indices fuse_listwise uses only for its `keep` filter (the neural arms need them; arm A
# does not, but the filter drops the same rows, so keep it to match).
pidx = {a: i for i, a in enumerate(json.load(open(W / "d8979601_full" / "accs.json")))}
gz = np.load("/home/frapercan/Thesis2/storage/two_tower_sparse/go_sparse_codes.npz", allow_pickle=True)
gidx = {g: i for i, g in enumerate(gz["go_ids"].tolist())}

EX = {"protein_accession", "go_term_id", "label", "category", "snapshot_pair", "qualifier",
      "evidence_code", "taxonomic_relation", "aspect"}
FEATS = [c for c in pq.ParquetFile(DS / "eval.parquet").schema_arrow.names if c not in EX]
print(f"{len(FEATS)} features  ({time.time()-t0:.0f}s)", flush=True)


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
    return P[keep], G[keep], Y[keep], X[keep]


trP, trG, trY, trX = load(DS / "train.parquet", "past")
vaP, vaG, vaY, vaX = load(DS / "train.parquet", "val")
evP, evG, evY, evX = load(DS / "eval.parquet")
print(f"train {len(trP):,}/{trY.sum():,.0f}pos | val {len(vaP):,}/{vaY.sum():,.0f}pos | "
      f"blind {len(evP):,}/{evY.sum():,.0f}pos  ({time.time()-t0:.0f}s)", flush=True)

GT = collections.defaultdict(set)
with (REL / "groundtruth_PK.tsv").open() as fh:
    next(fh)
    for line in fh:
        f = line.rstrip("\n").split("\t")
        if len(f) >= 3 and f[2] == "P":
            GT[f[0]].add(f[1])

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


def score(s):
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "pd"; d.mkdir()
        with (d / "p.tsv").open("w") as fh:
            for p_, g_, v in zip(evP, evG, s):
                fh.write(f"{p_}\t{g_}\t{v:.6f}\n")
        gt = Path(td) / "gt.tsv"
        with gt.open("w") as fh:
            for p_ in GT:
                for g_ in GT[p_]:
                    fh.write(f"{p_}\t{g_}\n")
        raw = Path(td) / "r.json"; drv = Path(td) / "d.py"
        drv.write_text(DRIVER.format(obo=OBO, pd=str(d), gt=str(gt), ia=IA, toi=TOI, o=str(raw)))
        r = subprocess.run([PY, str(drv)], capture_output=True, text=True, timeout=7200)
        if r.returncode != 0:
            print(r.stderr[-800:], flush=True); return None
        best = None
        for rec in json.loads(raw.read_text()).get("f_micro_w", []):
            if (rec.get("ns") or "") == "biological_process" and rec.get("f_micro_w") is not None:
                best = rec
        return round(float(best["f_micro_w"]), 5) if best else None


o = np.argsort(trP, kind="stable"); _, sz = np.unique(trP[o], return_counts=True)
ov = np.argsort(vaP, kind="stable"); _, szv = np.unique(vaP[ov], return_counts=True)
LGB = {"objective": "lambdarank", "metric": ["ndcg"], "ndcg_eval_at": [5, 10], "label_gain": [0, 1],
       "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 100, "feature_fraction": 0.9,
       "bagging_fraction": 0.9, "bagging_freq": 5, "seed": SEED, "verbose": -1, "num_threads": 12,
       "max_bin": 63, "two_round": True, "force_col_wise": True}
dtr = lgb.Dataset(trX[o], label=trY[o].astype(int), group=sz, feature_name=FEATS)
dva = lgb.Dataset(vaX[ov], label=vaY[ov].astype(int), group=szv, reference=dtr, feature_name=FEATS)
bst = lgb.train(LGB, dtr, num_boost_round=3000, valid_sets=[dva],
                callbacks=[lgb.early_stopping(50, verbose=False)])
a_s = bst.predict(evX, num_iteration=bst.best_iteration).astype(np.float32)
np.savez(W / "arm_a_repro.npz", pred=a_s, evP=evP, evG=evG)
fa = score(a_s)
print(f"\n  arm A reproduction: {fa}  (best_iter {bst.best_iteration}, target {DEPLOYED}, "
      f"fuse_listwise 0.21969)  ({time.time()-t0:.0f}s)", flush=True)
print(f"  reproduces: {fa is not None and abs(fa - 0.21969) < 0.01}", flush=True)
print("DONE", flush=True)
