"""Isolate the POOL lever from the model, and test the propagation explanation.

compose_levers.py measured arm E (drop classifier-only candidates) = 0.2121, but it
scored the arm-D model (rank feats + class weight), which was the WORST arm (0.1441).
So +0.068 mixes model and pool. This isolates them.

The surprise to explain: the dropped candidates are RICHER in positives (4.45% vs
0.96%) yet dropping them raises f_micro_w. Hypothesis: with prop="fill", cafaeval
propagates every predicted term to its ancestors. Classifier-only candidates have no
neighbour evidence (that is WHY they are classifier-only), so the model scores them
poorly AND each one drags an ancestor chain in at the same score. Under one global
threshold that is a precision bomb whose true positives are mostly low-IA.

Arms, all scored by the SAME cafaeval vs the FULL gt, same proteins:
  B_full   binary model, full pool          (control, expect 0.1518)
  B_drop   binary model, classifier-only dropped   <- the clean pool lever
  L_full   lambdarank, full pool            (the pipeline baseline, expect ~0.1255)
  L_drop   lambdarank, classifier-only dropped     <- does the pool lever need binary?

Also reports, for the explanation: propagated precision/recall proxies and the IA mass
of the kept vs dropped positives.
"""
import json, subprocess, tempfile, time
from pathlib import Path
import numpy as np, pyarrow.parquet as pq, lightgbm as lgb

W = Path("/home/frapercan/Thesis2/storage/cooc_experiment")
DS = Path("/home/frapercan/Thesis2/repositories/protea-reranker-lab/datasets/protst-global-train227-test230")
OBO = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
IA = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv"
PY = "/home/frapercan/Thesis2/repositories/PROTEA/.venv/bin/python"
VAL = "v225-v227"
EX = {"protein_accession", "go_term_id", "label", "category", "snapshot_pair", "qualifier",
      "evidence_code", "taxonomic_relation", "aspect"}
FEATS = [c for c in pq.ParquetFile(DS / "train.parquet").schema_arrow.names if c not in EX]


def load(p):
    cols = FEATS + ["category", "aspect", "snapshot_pair", "protein_accession", "go_term_id", "label"]
    t = pq.read_table(p, columns=list(dict.fromkeys(cols)))
    cat = np.asarray(t.column("category").to_pylist())
    asp = np.asarray(t.column("aspect").to_pylist())
    m = (cat == "pk") & (asp == "bpo")
    X = np.empty((int(m.sum()), len(FEATS)), dtype=np.float32)
    for j, c in enumerate(FEATS):
        X[:, j] = t.column(c).to_numpy(zero_copy_only=False).astype(np.float32)[m]
    md = {k: np.asarray(t.column(k).to_pylist())[m]
          for k in ("snapshot_pair", "protein_accession", "go_term_id")}
    md["label"] = (t.column("label").to_numpy(zero_copy_only=False).astype(np.int64)[m] > 0).astype(np.int32)
    return X, md


DRIVER = '''
import json, signal
from cafaeval.evaluation import cafa_eval
signal.signal(signal.SIGTERM, signal.SIG_DFL)
df,dfs=cafa_eval("{obo}","{pred_dir}","{gt}",ia="{ia}",prop="fill",norm="cafa",
    no_orphans=True,max_terms=500,th_step=0.001,n_cpu=1,weighted_only=False)
out={{}}
for k,v in dfs.items(): out[k]=v.reset_index().to_dict(orient="records")
json.dump(out,open("{out_json}","w"),default=str)
'''
TRUTH = [tuple(l.rstrip("\n").split("\t")) for l in open(DS / "gt_pk_bp.tsv")]


def score(name, s, mev, keep=None):
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "pred_dir"
        d.mkdir(parents=True)
        with (d / f"{name}.tsv").open("w") as fh:
            for i, (p, g, v) in enumerate(zip(mev["protein_accession"], mev["go_term_id"], s)):
                if keep is not None and not keep[i]:
                    continue
                fh.write(f"{p}\t{g}\t{v:.6f}\n")
        gt = Path(td) / "gt.tsv"
        with gt.open("w") as fh:
            for p, g in TRUTH:
                fh.write(f"{p}\t{g}\n")
        raw = Path(td) / "r.json"
        drv = Path(td) / "d.py"
        drv.write_text(DRIVER.format(obo=OBO, pred_dir=str(d), gt=str(gt), ia=IA, out_json=str(raw)))
        r = subprocess.run([PY, str(drv)], capture_output=True, text=True, timeout=5400)
        if r.returncode != 0:
            print(f"  cafaeval FAILED {name}: {r.stderr[-300:]}", flush=True)
            return None
        data = json.loads(raw.read_text())
        best = None
        for rec in data.get("f_micro_w", []):
            if (rec.get("ns") or "") == "biological_process" and rec.get("f_micro_w") is not None:
                best = rec
        if not best:
            return None
        return {k: (round(float(best[k]), 4) if isinstance(best.get(k), (int, float)) else best.get(k))
                for k in best if k in ("f_micro_w", "pr_micro_w", "rc_micro_w", "tau", "cov_max")}


t0 = time.time()
Xtr, mtr = load(DS / "train.parquet")
Xev, mev = load(DS / "eval.parquet")
iv = mtr["snapshot_pair"] == VAL
ytr = mtr["label"]
kp = Xev[:, FEATS.index("knn_present")]
cp = Xev[:, FEATS.index("classifier_present")]
keep = ~((kp <= 0) & (cp > 0))
res = {"note": "same proteins, same gt, same harness; only the pool and the objective vary"}

# --- IA mass of kept vs dropped positives: is the drop cheap in WEIGHTED recall? ---
ia = {}
for line in open(IA):
    p = line.split()
    if len(p) >= 2:
        try:
            ia[p[0]] = float(p[1])
        except ValueError:
            pass
lab = mev["label"].astype(bool)
go = mev["go_term_id"]
ia_kept = sum(ia.get(g, 0.0) for g in go[lab & keep])
ia_drop = sum(ia.get(g, 0.0) for g in go[lab & ~keep])
res["positives_kept"] = int((lab & keep).sum())
res["positives_dropped"] = int((lab & ~keep).sum())
res["IA_mass_kept_positives"] = round(ia_kept, 1)
res["IA_mass_dropped_positives"] = round(ia_drop, 1)
res["mean_IA_kept_positive"] = round(ia_kept / max((lab & keep).sum(), 1), 3)
res["mean_IA_dropped_positive"] = round(ia_drop / max((lab & ~keep).sum(), 1), 3)
print(f"IA per kept positive={res['mean_IA_kept_positive']} vs dropped={res['mean_IA_dropped_positive']}", flush=True)

P = {"metric": ["auc"], "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 100,
     "feature_fraction": 0.9, "bagging_fraction": 0.9, "bagging_freq": 5, "seed": 42,
     "verbose": -1, "num_threads": 12, "max_bin": 63, "two_round": True, "force_col_wise": True}

# --- binary model -------------------------------------------------------------
Pb = dict(P, objective="binary")
b = lgb.train(Pb, lgb.Dataset(Xtr[~iv], label=ytr[~iv], feature_name=FEATS), num_boost_round=2000,
              valid_sets=[lgb.Dataset(Xtr[iv], label=ytr[iv], feature_name=FEATS)],
              callbacks=[lgb.early_stopping(50, verbose=False)])
sB = b.predict(Xev, num_iteration=b.best_iteration)
res["B_full"] = score("Bfull", sB, mev)
print(f"[B_full] {res['B_full']}  ({time.time()-t0:.0f}s)", flush=True)
res["B_drop"] = score("Bdrop", sB, mev, keep=keep)
print(f"[B_drop] {res['B_drop']}  ({time.time()-t0:.0f}s)", flush=True)
json.dump(res, open(W / "isolate_pool_lever.json", "w"), indent=1)

# --- lambdarank model: does the pool lever need the binary objective? ----------
grp_tr = mtr["protein_accession"][~iv]
grp_iv = mtr["protein_accession"][iv]


def groups(a):
    _, idx = np.unique(a, return_index=True)
    order = np.argsort(idx)
    return np.array([np.sum(a == u) for u in a[np.sort(idx)]])[order] if len(a) else np.array([])


ord_tr = np.argsort(grp_tr, kind="stable")
ord_iv = np.argsort(grp_iv, kind="stable")
Pl = dict(P, objective="lambdarank", metric=["ndcg"], ndcg_eval_at=[5, 10])
dtr = lgb.Dataset(Xtr[~iv][ord_tr], label=ytr[~iv][ord_tr], feature_name=FEATS,
                  group=np.unique(grp_tr[ord_tr], return_counts=True)[1])
div = lgb.Dataset(Xtr[iv][ord_iv], label=ytr[iv][ord_iv], feature_name=FEATS,
                  group=np.unique(grp_iv[ord_iv], return_counts=True)[1])
l = lgb.train(Pl, dtr, num_boost_round=2000, valid_sets=[div],
              callbacks=[lgb.early_stopping(50, verbose=False)])
sL = l.predict(Xev, num_iteration=l.best_iteration)
res["L_full"] = score("Lfull", sL, mev)
print(f"[L_full] {res['L_full']}  ({time.time()-t0:.0f}s)", flush=True)
res["L_drop"] = score("Ldrop", sL, mev, keep=keep)
print(f"[L_drop] {res['L_drop']}  ({time.time()-t0:.0f}s)", flush=True)

json.dump(res, open(W / "isolate_pool_lever.json", "w"), indent=1)
print("\n=== PK-BPO f_micro_w: pool lever isolated from the model ===", flush=True)
for k in ("L_full", "L_drop", "B_full", "B_drop"):
    v = res.get(k)
    print(f"  {k:8s} {v}", flush=True)
print("DONE", flush=True)
