"""THE ANCHOR. Reproduce the deployed recipe's PK-BPO number from a script, and run the
one controlled objective test that was missing.

Why this exists: the campaign's headline PK-BPO number has been wrong twice.
  1. 0.1255 was manufactured by a rankpct() normalisation in fuse_and_score.py.
  2. Its replacement, 0.2131, was computed in a /tmp heredoc that was then deleted, and
     shipped to three surfaces anyway. It currently has NO script.
This file is the script. It must exist before any surface quotes the anchor again.

ARM A (no training). Score the ACTUAL deployed booster, whose per-candidate scores are
already frozen in rerank_out/eval_scores.parquet (written by train_allfeat.py: per-category
lambdarank, aspects pooled, grouped by (snapshot_pair, protein, aspect)). Score it RAW.
This is the anchor: the number the deployed recipe actually delivers on this pool.
Expected ~0.2131. If it does not reproduce, the anchor is not what we think it is.

ARM B (one training). The controlled objective test at the DEPLOYED grouping, which the
campaign never ran. Existing evidence:
  - isolate_pool_lever.py: lambdarank 0.2222 vs binary 0.1518, same script/params, but
    grouped by protein ALONE (off-recipe) -> internally controlled, externally wrong.
  - isolate_percell_split.py + decompose_order_vs_count.py: BOTH give 0.2017 for
    per-cell lambdarank at the deployed grouping. So 0.2222 stands alone and off-recipe.
There is no binary arm at the deployed grouping, so "binary is worse by 0.070" rests on a
comparison nobody should quote. This trains binary with EXACTLY train_allfeat.py's params
and grouping key, pooled aspects, so the only thing that differs from arm A's recipe is
the objective.

Both arms scored by the same cafaeval against the FULL ground truth.
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

DRIVER = '''
import json, signal
from cafaeval.evaluation import cafa_eval
signal.signal(signal.SIGTERM, signal.SIG_DFL)
df,dfs=cafa_eval("{obo}","{pd}","{gt}",ia="{ia}",prop="fill",norm="cafa",
    no_orphans=True,max_terms=500,th_step=0.001,n_cpu=1,weighted_only=False)
out={{}}
for k,v in dfs.items(): out[k]=v.reset_index().to_dict(orient="records")
json.dump(out,open("{o}","w"),default=str)
'''
TRUTH = [tuple(l.rstrip("\n").split("\t")) for l in open(DS / "gt_pk_bp.tsv")]


def score(name, prot, go, s):
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "pd"
        d.mkdir()
        with (d / f"{name}.tsv").open("w") as fh:
            for p, g, v in zip(prot, go, s):
                fh.write(f"{p}\t{g}\t{v:.6f}\n")
        gt = Path(td) / "gt.tsv"
        with gt.open("w") as fh:
            for p, g in TRUTH:
                fh.write(f"{p}\t{g}\n")
        raw = Path(td) / "r.json"
        drv = Path(td) / "d.py"
        drv.write_text(DRIVER.format(obo=OBO, pd=str(d), gt=str(gt), ia=IA, o=str(raw)))
        r = subprocess.run([PY, str(drv)], capture_output=True, text=True, timeout=5400)
        if r.returncode != 0:
            print(f"  cafaeval FAILED {name}: {r.stderr[-300:]}", flush=True)
            return None
        best = None
        for rec in json.loads(raw.read_text()).get("f_micro_w", []):
            if (rec.get("ns") or "") == "biological_process" and rec.get("f_micro_w") is not None:
                best = rec
        return {k: (round(float(best[k]), 4) if isinstance(best.get(k), (int, float)) else best.get(k))
                for k in ("f_micro_w", "pr_micro_w", "rc_micro_w", "tau", "cov_max")} if best else None


t0 = time.time()
res = {"what": "the anchor, scripted, plus the objective test at the DEPLOYED grouping"}

# ---- ARM A: the deployed booster's own frozen scores, raw. No training. ----------
t = pq.read_table(W / "rerank_out" / "eval_scores.parquet")
cat = np.asarray(t.column("category").to_pylist())
asp = np.asarray(t.column("aspect").to_pylist())
m = (cat == "pk") & (asp == "bpo")
prot_a = np.asarray(t.column("protein_accession").to_pylist())[m]
go_a = np.asarray(t.column("go_term_id").to_pylist())[m]
rr = t.column("reranker_score").to_numpy(zero_copy_only=False).astype(np.float64)[m]
res["A_anchor_deployed_booster_raw"] = score("A", prot_a, go_a, rr)
res["A_source"] = "rerank_out/eval_scores.parquet (train_allfeat.py booster), RAW score"
print(f"[A] ANCHOR deployed booster, raw  {res['A_anchor_deployed_booster_raw']}  ({time.time()-t0:.0f}s)", flush=True)


def rankpct(x):
    o = np.argsort(x, kind="stable")
    r = np.empty(len(x))
    r[o] = np.arange(len(x))
    return r / max(1, len(x) - 1)


res["A_same_scores_through_rankpct"] = score("Arp", prot_a, go_a, rankpct(rr))
print(f"[A'] the SAME scores via rankpct  {res['A_same_scores_through_rankpct']}  <- should be ~0.1255", flush=True)
json.dump(res, open(W / "anchor_deployed_recipe.json", "w"), indent=1)

# ---- ARM B: binary at the DEPLOYED grouping and params, pooled aspects -----------
def load(p):
    cols = FEATS + ["category", "aspect", "snapshot_pair", "protein_accession", "go_term_id", "label"]
    tb = pq.read_table(p, columns=list(dict.fromkeys(cols)))
    c = np.asarray(tb.column("category").to_pylist())
    mm = c == "pk"
    X = np.empty((int(mm.sum()), len(FEATS)), dtype=np.float32)
    for j, cc in enumerate(FEATS):
        X[:, j] = tb.column(cc).to_numpy(zero_copy_only=False).astype(np.float32)[mm]
    md = {k: np.asarray(tb.column(k).to_pylist())[mm]
          for k in ("snapshot_pair", "protein_accession", "go_term_id", "aspect")}
    md["label"] = (tb.column("label").to_numpy(zero_copy_only=False).astype(np.int64)[mm] > 0).astype(np.int32)
    return X, md


Xtr, mtr = load(DS / "train.parquet")
Xev, mev = load(DS / "eval.parquet")
iv = mtr["snapshot_pair"] == VAL
# exactly train_allfeat.py, except objective
P = {"objective": "binary", "metric": ["auc"], "learning_rate": 0.05, "num_leaves": 63,
     "min_data_in_leaf": 100, "feature_fraction": 0.9, "bagging_fraction": 0.9,
     "bagging_freq": 5, "seed": 42, "verbose": -1, "num_threads": 12,
     "max_bin": 63, "two_round": True, "force_col_wise": True}
b = lgb.train(P, lgb.Dataset(Xtr[~iv], label=mtr["label"][~iv], feature_name=FEATS),
              num_boost_round=3000,
              valid_sets=[lgb.Dataset(Xtr[iv], label=mtr["label"][iv], feature_name=FEATS)],
              callbacks=[lgb.early_stopping(50, verbose=False)])
s = b.predict(Xev, num_iteration=b.best_iteration)
bp = mev["aspect"] == "bpo"
res["B_binary_deployed_scope"] = score("B", mev["protein_accession"][bp], mev["go_term_id"][bp], s[bp])
res["B_note"] = ("binary trained on ALL pk rows (aspects pooled) with train_allfeat.py's params; "
                 "no group key needed for a pointwise objective, so this isolates the objective "
                 "against arm A's recipe")
res["B_best_iter"] = b.best_iteration
print(f"[B] binary at deployed scope      {res['B_binary_deployed_scope']}  ({time.time()-t0:.0f}s)", flush=True)

fA = (res.get("A_anchor_deployed_booster_raw") or {}).get("f_micro_w")
fB = (res.get("B_binary_deployed_scope") or {}).get("f_micro_w")
if fA and fB:
    res["objective_delta_at_deployed_scope"] = round(fB - fA, 4)
res["oracle_ceiling"] = 0.6077
if fA:
    res["capture_of_ceiling"] = round(fA / 0.6077, 4)
json.dump(res, open(W / "anchor_deployed_recipe.json", "w"), indent=1)

print("\n=== THE ANCHOR, with a script this time ===", flush=True)
print(f"  deployed booster, RAW      = {fA}   <- the anchor", flush=True)
print(f"  same scores via rankpct    = {(res.get('A_same_scores_through_rankpct') or {}).get('f_micro_w')}   <- the artefact", flush=True)
print(f"  binary at deployed scope   = {fB}", flush=True)
if fA and fB:
    print(f"  -> objective delta         = {fB-fA:+.4f}", flush=True)
if fA:
    print(f"  -> capture of the 0.6077 ceiling = {fA/0.6077:.1%}", flush=True)
print("DONE", flush=True)
