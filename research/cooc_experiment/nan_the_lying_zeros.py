"""The cheap half of PKW.1: the zero-fills LIE. Turn them into NaN and remeasure.

DIAGNOSIS (merge_diag.py, verified). The classifier-proposed candidates are 43.4% of the
pk-bpo pool and carry 78% of the positives, and the feature producers never ran over them:

  NaN on 100% of classifier rows : distance, all nw/sw alignments, length_query, length_ref,
                                   taxonomic_distance, k_position, go_term_frequency,
                                   neighbor_*, tax_voters_*, anc2vec_neighbor_*, ...
  ZERO on 100% of classifier rows: vote_count, neighbor_vote_fraction,
                                   anc2vec_query_known_count, anc2vec_has_emb, lineage_*

`length_query` is the QUERY PROTEIN's length and is NaN on every classifier row: the same
protein carries it on its knn row. That is not a design, it is a coverage hole.

THE ASYMMETRY THIS EXPLOITS. A NaN routes as missing in LightGBM: the tree simply learns a
default direction. A ZERO does not. `anc2vec_query_known_count = 0` asserts the protein
knows no terms when it does; `vote_count = 0` asserts no neighbour voted when we never
asked. The model cannot tell "measured zero" from "never measured", so it learns a false
pattern over 43.4% of the pool. This is precisely the bug D45 / PROTEA#710 fixed by
switching `_lafa_default_fields()` from zero-fill to NaN, for four families. The KNN-path
families were never enrolled (the schema audit: the degeneracy check covers 4 of 20).

THE ARMS. Same script, same deployed params, same rows, same harness, same full gt. The
only thing that varies is whether the lying zeros are told apart from measured zeros.

  A  as shipped                                    control, must reproduce ~0.2177 (arm P)
  B  lying zeros -> NaN on classifier-only rows    the fix, offline

A zero is only "lying" where the producer did not run, so B rewrites ONLY the classifier-
only rows, and ONLY the columns that are zero on 100% of them while being live on the knn
rows. Measured zeros on knn rows are untouched. One variable.

WHY THIS IS THE CHEAP HALF. The real fix is producer coverage: run the KNN-path producers
over the classifier candidates too. That needs an export and a PR. This asks the prior
question for free: if the model cannot even use the honesty, coverage will not pay either.
GATE: if B does not move PK-BPO, say so and drop the export proposal.
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

# exactly train_allfeat.py's deployed params
P = {"objective": "lambdarank", "metric": ["ndcg", "map"], "ndcg_eval_at": [5, 10],
     "label_gain": [0, 1], "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 100,
     "feature_fraction": 0.9, "bagging_fraction": 0.9, "bagging_freq": 5, "seed": 42,
     "verbose": -1, "num_threads": 12, "max_bin": 63, "two_round": True, "force_col_wise": True}


def load(p):
    cols = FEATS + ["category", "aspect", "snapshot_pair", "protein_accession", "go_term_id", "label"]
    t = pq.read_table(p, columns=list(dict.fromkeys(cols)))
    c = np.asarray(t.column("category").to_pylist())
    m = c == "pk"                                   # deployed scope: per category, aspects pooled
    X = np.empty((int(m.sum()), len(FEATS)), dtype=np.float32)
    for j, col in enumerate(FEATS):
        X[:, j] = t.column(col).to_numpy(zero_copy_only=False).astype(np.float32)[m]
    md = {k: np.asarray(t.column(k).to_pylist())[m]
          for k in ("snapshot_pair", "protein_accession", "go_term_id", "aspect")}
    md["label"] = (t.column("label").to_numpy(zero_copy_only=False).astype(np.int64)[m] > 0).astype(np.int32)
    return X, md


t0 = time.time()
Xtr, mtr = load(DS / "train.parquet")
Xev, mev = load(DS / "eval.parquet")
print(f"loaded train {Xtr.shape} eval {Xev.shape} ({time.time()-t0:.0f}s)", flush=True)

KPI, CPI = FEATS.index("knn_present"), FEATS.index("classifier_present")


def clf_only(X):
    return (X[:, KPI] <= 0) & (X[:, CPI] > 0)


# find the lying columns FROM THE DATA, do not hardcode: zero on ~all classifier-only rows
# while being genuinely live on the knn rows.
co_tr, knn_tr = clf_only(Xtr), Xtr[:, KPI] > 0
lying = []
for j, col in enumerate(FEATS):
    if col in ("classifier_score", "classifier_present"):
        continue
    v = Xtr[:, j]
    zc = float(np.nanmean(v[co_tr] == 0)) if co_tr.sum() else 0.0
    zk = float(np.nanmean(v[knn_tr] == 0)) if knn_tr.sum() else 0.0
    if zc > 0.99 and zk < 0.90:          # constant zero where the producer never ran, live where it did
        lying.append((col, j, round(zk, 3)))
res = {"note": "the zero-fills on classifier-only rows are indistinguishable from measured "
               "zeros. NaN routes as missing in LightGBM; zero does not. One variable: "
               "only classifier-only rows, only columns zero on >99% of them and live on knn rows.",
       "lying_columns": [c for c, _, _ in lying],
       "clf_only_frac_train": round(float(co_tr.mean()), 4)}
print(f"\ncolumns that are a LIE on classifier-only rows ({len(lying)}):", flush=True)
for c, _, zk in lying:
    print(f"  {c:32s} zero on {zk:.1%} of knn rows (so zero is a real value there)", flush=True)
json.dump(res, open(W / "nan_the_lying_zeros.json", "w"), indent=1)

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


def key(md, sel):
    return np.char.add(np.char.add(np.char.add(md["snapshot_pair"][sel], "|"),
                                   np.char.add(md["protein_accession"][sel], "|")), md["aspect"][sel])


def run(name, Xtr_, Xev_):
    iv = mtr["snapshot_pair"] == VAL
    parts = {}
    for tag, sel in (("tr", ~iv), ("va", iv)):
        k = key(mtr, sel)
        o = np.argsort(k, kind="stable")
        _, sizes = np.unique(k[o], return_counts=True)
        parts[tag] = (Xtr_[sel][o], mtr["label"][sel][o], sizes)
    dtr = lgb.Dataset(parts["tr"][0], label=parts["tr"][1], group=parts["tr"][2], feature_name=FEATS)
    dva = lgb.Dataset(parts["va"][0], label=parts["va"][1], group=parts["va"][2], reference=dtr,
                      feature_name=FEATS)
    b = lgb.train(P, dtr, num_boost_round=3000, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
    s = b.predict(Xev_, num_iteration=b.best_iteration)
    bp = mev["aspect"] == "bpo"
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "pd"
        d.mkdir()
        with (d / f"{name}.tsv").open("w") as fh:
            for p_, g_, v in zip(mev["protein_accession"][bp], mev["go_term_id"][bp], s[bp]):
                fh.write(f"{p_}\t{g_}\t{v:.6f}\n")
        gt = Path(td) / "gt.tsv"
        with gt.open("w") as fh:
            for p_, g_ in TRUTH:
                fh.write(f"{p_}\t{g_}\n")
        raw = Path(td) / "r.json"
        drv = Path(td) / "d.py"
        drv.write_text(DRIVER.format(obo=OBO, pd=str(d), gt=str(gt), ia=IA, o=str(raw)))
        r = subprocess.run([PY, str(drv)], capture_output=True, text=True, timeout=5400)
        if r.returncode != 0:
            print(f"  cafaeval FAILED {name}: {r.stderr[-300:]}", flush=True)
            return None, b.best_iteration
        best = None
        for rec in json.loads(raw.read_text()).get("f_micro_w", []):
            if (rec.get("ns") or "") == "biological_process" and rec.get("f_micro_w") is not None:
                best = rec
        return ({k: round(float(best[k]), 4) for k in ("f_micro_w", "pr_micro_w", "rc_micro_w", "tau", "cov_max")}
                if best else None), b.best_iteration


a, ai = run("A", Xtr, Xev)
res["A_as_shipped"] = a
res["A_best_iter"] = ai
print(f"\n[A] as shipped (lying zeros)  {a}  ({time.time()-t0:.0f}s)", flush=True)
json.dump(res, open(W / "nan_the_lying_zeros.json", "w"), indent=1)

XtrB, XevB = Xtr.copy(), Xev.copy()
for _, j, _ in lying:
    XtrB[co_tr, j] = np.nan
    XevB[clf_only(Xev), j] = np.nan
b_, bi = run("B", XtrB, XevB)
res["B_lying_zeros_as_nan"] = b_
res["B_best_iter"] = bi
print(f"[B] lying zeros -> NaN        {b_}  ({time.time()-t0:.0f}s)", flush=True)

fa = (a or {}).get("f_micro_w")
fb = (b_ or {}).get("f_micro_w")
if fa and fb:
    res["delta"] = round(fb - fa, 4)
json.dump(res, open(W / "nan_the_lying_zeros.json", "w"), indent=1)
print("\n=== Does telling the model the truth help? ===", flush=True)
print(f"  A as shipped            = {fa}   (control, expect ~0.2177)", flush=True)
print(f"  B lying zeros -> NaN    = {fb}", flush=True)
if fa and fb:
    print(f"  -> delta = {fb-fa:+.4f}", flush=True)
    print("  POSITIVE => honesty alone pays, and producer COVERAGE is worth an export.", flush=True)
    print("  FLAT     => the model cannot use it; drop the export proposal and say so.", flush=True)
print("DONE", flush=True)
