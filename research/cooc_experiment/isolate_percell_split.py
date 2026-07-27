"""Isolate the ONE change: per-category (aspects pooled) vs per-cell (PK-BPO only).

Why this exists: the published claim "the binary objective is worth +0.0263" is
CONFOUNDED. The 0.1255 baseline came from train_allfeat.py, which trains ONE
lambdarank booster per CATEGORY with MFO/BPO/CCO POOLED, grouped by
(snapshot_pair, protein, aspect). The 0.1518 binary arm was trained on PK-BPO ONLY.
So that comparison changed the objective AND the scope at once.

isolate_pool_lever.py then measured, in one script, same data, only the objective
varying, both PK-BPO-only:
    lambdarank 0.2222   vs   binary 0.1518
i.e. lambdarank BEATS binary by +0.070, the opposite of what was published, and it
suggests the real lever is the SCOPE, not the objective.

This script settles it. Both arms are lambdarank with IDENTICAL params and IDENTICAL
grouping semantics; only the training SCOPE differs:

  P (pooled)  train on ALL pk rows (mfo+bpo+cco), group by (snapshot_pair, protein,
              aspect)  -> the deployed recipe. Must reproduce ~0.1255 on PK-BPO.
  S (split)   train on pk+bpo rows only,          group by (snapshot_pair, protein,
              aspect)  -> aspect is constant, so the grouping key is the same tuple.

Both scored on the SAME PK-BPO eval rows by the SAME cafaeval against the FULL gt.
If P reproduces ~0.1255 and S lands ~0.22, the per-cell split is the lever and the
published attribution to the binary objective is wrong.

METHOD NOTE this run exists to enforce: AUC ranked these levers in the OPPOSITE order
to f_micro_w. technique_probe measured the split at +0.007 AUC (dismissed as minor)
and binary at +0.032 AUC (promoted). In f_micro_w the split is the big one and binary
is negative. Never rank levers by AUC.
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

# Exactly train_allfeat.py's deployed params.
P = {"objective": "lambdarank", "metric": ["ndcg", "map"], "ndcg_eval_at": [5, 10],
     "label_gain": [0, 1], "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 100,
     "feature_fraction": 0.9, "bagging_fraction": 0.9, "bagging_freq": 5, "seed": 42,
     "verbose": -1, "num_threads": 12, "max_bin": 63, "two_round": True, "force_col_wise": True}


def load(p, bpo_only):
    cols = FEATS + ["category", "aspect", "snapshot_pair", "protein_accession", "go_term_id", "label"]
    t = pq.read_table(p, columns=list(dict.fromkeys(cols)))
    cat = np.asarray(t.column("category").to_pylist())
    asp = np.asarray(t.column("aspect").to_pylist())
    m = (cat == "pk") & ((asp == "bpo") if bpo_only else np.ones(len(asp), bool))
    X = np.empty((int(m.sum()), len(FEATS)), dtype=np.float32)
    for j, c in enumerate(FEATS):
        X[:, j] = t.column(c).to_numpy(zero_copy_only=False).astype(np.float32)[m]
    md = {k: np.asarray(t.column(k).to_pylist())[m]
          for k in ("snapshot_pair", "protein_accession", "go_term_id", "aspect")}
    md["label"] = (t.column("label").to_numpy(zero_copy_only=False).astype(np.int64)[m] > 0).astype(np.int32)
    return X, md


def combo_key(md):
    """The deployed grouping key: (snapshot_pair, protein, aspect)."""
    return np.char.add(np.char.add(np.char.add(md["snapshot_pair"], "|"),
                                   np.char.add(md["protein_accession"], "|")), md["aspect"])


def fit(X, md, iv):
    """Train one lambdarank booster with the deployed grouping. Sorted by group key."""
    out = {}
    for tag, sel in (("tr", ~iv), ("va", iv)):
        k = combo_key({q: md[q][sel] for q in ("snapshot_pair", "protein_accession", "aspect")})
        order = np.argsort(k, kind="stable")
        _, sizes = np.unique(k[order], return_counts=True)
        out[tag] = (X[sel][order], md["label"][sel][order], sizes)
    dtr = lgb.Dataset(out["tr"][0], label=out["tr"][1], group=out["tr"][2], feature_name=FEATS)
    dva = lgb.Dataset(out["va"][0], label=out["va"][1], group=out["va"][2], reference=dtr,
                      feature_name=FEATS)
    return lgb.train(P, dtr, num_boost_round=3000, valid_sets=[dva],
                     callbacks=[lgb.early_stopping(50, verbose=False)])


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


def score(name, prot, go, s):
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "pred_dir"
        d.mkdir(parents=True)
        with (d / f"{name}.tsv").open("w") as fh:
            for p, g, v in zip(prot, go, s):
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
        best = None
        for rec in json.loads(raw.read_text()).get("f_micro_w", []):
            if (rec.get("ns") or "") == "biological_process" and rec.get("f_micro_w") is not None:
                best = rec
        return {k: (round(float(best[k]), 4) if isinstance(best.get(k), (int, float)) else best.get(k))
                for k in ("f_micro_w", "pr_micro_w", "rc_micro_w", "tau", "cov_max")} if best else None


t0 = time.time()
res = {"note": "ONLY the training scope varies: pooled-aspect vs PK-BPO-only. "
               "Same objective (lambdarank), same params, same grouping key "
               "(snapshot_pair, protein, aspect), same eval rows, same gt, same harness.",
       "published_baseline_to_reproduce": 0.1255}

# eval rows are ALWAYS the PK-BPO ones, for both arms
Xev_b, mev_b = load(DS / "eval.parquet", bpo_only=True)

# --- S: per-cell (PK-BPO only) ------------------------------------------------
Xtr_b, mtr_b = load(DS / "train.parquet", bpo_only=True)
bS = fit(Xtr_b, mtr_b, mtr_b["snapshot_pair"] == VAL)
res["S_percell_pkbpo"] = score("S", mev_b["protein_accession"], mev_b["go_term_id"],
                               bS.predict(Xev_b, num_iteration=bS.best_iteration))
res["S_best_iter"] = bS.best_iteration
print(f"[S] per-cell (pk+bpo only)   {res['S_percell_pkbpo']}  ({time.time()-t0:.0f}s)", flush=True)
json.dump(res, open(W / "isolate_percell_split.json", "w"), indent=1)
del Xtr_b, bS

# --- P: pooled aspects (the deployed recipe) ----------------------------------
Xtr_a, mtr_a = load(DS / "train.parquet", bpo_only=False)
print(f"  pooled train rows {Xtr_a.shape} (vs pk+bpo {mtr_b['label'].shape})", flush=True)
bP = fit(Xtr_a, mtr_a, mtr_a["snapshot_pair"] == VAL)
res["P_pooled_percategory"] = score("P", mev_b["protein_accession"], mev_b["go_term_id"],
                                    bP.predict(Xev_b, num_iteration=bP.best_iteration))
res["P_best_iter"] = bP.best_iteration
print(f"[P] pooled aspects (deployed) {res['P_pooled_percategory']}  ({time.time()-t0:.0f}s)", flush=True)

fS = (res.get("S_percell_pkbpo") or {}).get("f_micro_w")
fP = (res.get("P_pooled_percategory") or {}).get("f_micro_w")
if fS and fP:
    res["per_cell_split_delta"] = round(fS - fP, 4)
    res["reproduces_published_baseline"] = abs(fP - 0.1255) < 0.02
json.dump(res, open(W / "isolate_percell_split.json", "w"), indent=1)
print("\n=== PK-BPO f_micro_w: the per-cell split, isolated ===", flush=True)
print(f"  P pooled aspects (deployed) = {fP}   (published baseline 0.1255)", flush=True)
print(f"  S per-cell PK-BPO only      = {fS}", flush=True)
if fS and fP:
    print(f"  delta = {fS-fP:+.4f}   (board gap to close was +0.076)", flush=True)
    print(f"  reproduces the 0.1255 baseline: {res['reproduces_published_baseline']}", flush=True)
print("DONE", flush=True)
