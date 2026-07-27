"""Do the technique levers COMPOSE, measured in f_micro_w (never AUC alone)?

Constraint (author): techniques and simple data only. No structures, no text, no new
modality. Every arm uses the SAME rows and the SAME features already exported.

Baselines already measured on this harness:
  lambdarank ............... 0.1255   (what the pipeline delivers)
  binary objective ......... 0.1518   (+0.0263, the one lever that paid)
  oracle ceiling ........... 0.6077   (perfect ranking of this same pool)
Gap to close on PK-BPO: +0.076.

Arms, each cumulative on the last, each scored by the SAME cafaeval against the FULL
ground truth so a recall loss is penalised and a precision gain is credited:

  B  binary                                    (control; must reproduce ~0.1518)
  C  B + within-protein rank features          (rank + z-score inside each protein's
                                                candidate list; the model currently has
                                                to infer relative structure. No rank
                                                feature exists in the export: verified.)
  D  C + class weighting                       (PK-BPO positives are ~1% of train)
  E  D + drop classifier-only candidates       (knn_present==0 & classifier_present==1;
                                                unioning them takes the pool 62 -> 114
                                                per query. Dropping them raises precision
                                                and LOSES recall: the full-gt scoring is
                                                what makes that trade honest.)

Memory: the pk booster peaked at 53GB/62G and was OOM-killed twice with no log trace.
max_bin 63 + two_round + force_col_wise keep the binned dataset ~25GB.
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
BASE_FEATS = [c for c in pq.ParquetFile(DS / "train.parquet").schema_arrow.names if c not in EX]

# Features worth re-expressing relative to the protein's own candidate list.
RANK_SRC = ["distance", "classifier_score", "vote_count", "neighbor_vote_fraction",
            "protst_text_score", "association_score", "k_position"]
RANK_SRC = [c for c in RANK_SRC if c in BASE_FEATS]


def load(p):
    cols = BASE_FEATS + ["category", "aspect", "snapshot_pair", "protein_accession", "go_term_id", "label"]
    t = pq.read_table(p, columns=list(dict.fromkeys(cols)))
    cat = np.asarray(t.column("category").to_pylist())
    asp = np.asarray(t.column("aspect").to_pylist())
    m = (cat == "pk") & (asp == "bpo")
    X = np.empty((int(m.sum()), len(BASE_FEATS)), dtype=np.float32)
    for j, c in enumerate(BASE_FEATS):
        X[:, j] = t.column(c).to_numpy(zero_copy_only=False).astype(np.float32)[m]
    md = {k: np.asarray(t.column(k).to_pylist())[m]
          for k in ("snapshot_pair", "protein_accession", "go_term_id")}
    md["label"] = (t.column("label").to_numpy(zero_copy_only=False).astype(np.int64)[m] > 0).astype(np.int32)
    return X, md


def add_rank_feats(X, prot):
    """Rank and z-score each RANK_SRC column WITHIN each protein's candidate list.

    Vectorised by sorting on protein: a per-protein python loop over ~4.4k proteins x
    7 columns is the kind of thing that ran on one core for an hour last time.
    """
    order = np.argsort(prot, kind="stable")
    starts = np.flatnonzero(np.r_[True, prot[order][1:] != prot[order][:-1]])
    bounds = np.r_[starts, len(order)]
    idx = [BASE_FEATS.index(c) for c in RANK_SRC]
    out = np.empty((X.shape[0], len(idx) * 2), dtype=np.float32)
    for a, b in zip(bounds[:-1], bounds[1:]):
        g = order[a:b]
        blk = X[np.ix_(g, idx)]
        n = blk.shape[0]
        # rank within the protein, normalised to [0,1]; ties get average rank
        r = np.argsort(np.argsort(blk, axis=0, kind="stable"), axis=0).astype(np.float32)
        out[g, :len(idx)] = r / max(n - 1, 1)
        mu = blk.mean(axis=0)
        sd = blk.std(axis=0)
        sd[sd == 0] = 1.0
        out[g, len(idx):] = (blk - mu) / sd
    return out


def score_arm(name, s, mev, keep=None):
    """Run the SAME cafaeval as every other arm, against the FULL ground truth."""
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
    truth = [tuple(l.rstrip("\n").split("\t")) for l in open(DS / "gt_pk_bp.tsv")]
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
            for p, g in truth:
                fh.write(f"{p}\t{g}\n")
        raw = Path(td) / "r.json"
        drv = Path(td) / "d.py"
        drv.write_text(DRIVER.format(obo=OBO, pred_dir=str(d), gt=str(gt), ia=IA, out_json=str(raw)))
        r = subprocess.run([PY, str(drv)], capture_output=True, text=True, timeout=5400)
        if r.returncode != 0:
            print(f"  cafaeval FAILED for {name}: {r.stderr[-300:]}", flush=True)
            return None
        data = json.loads(raw.read_text())
        for rec in data.get("f_micro_w", []):
            if (rec.get("ns") or "") == "biological_process" and rec.get("f_micro_w") is not None:
                return round(float(rec["f_micro_w"]), 4)
    return None


def train(Xtr, ytr, iv, feats, spw=None):
    P = {"objective": "binary", "metric": ["auc"], "learning_rate": 0.05, "num_leaves": 63,
         "min_data_in_leaf": 100, "feature_fraction": 0.9, "bagging_fraction": 0.9,
         "bagging_freq": 5, "seed": 42, "verbose": -1, "num_threads": 12,
         "max_bin": 63, "two_round": True, "force_col_wise": True}
    if spw:
        P["scale_pos_weight"] = spw
    tr = ~iv
    b = lgb.train(P, lgb.Dataset(Xtr[tr], label=ytr[tr], feature_name=feats),
                  num_boost_round=2000,
                  valid_sets=[lgb.Dataset(Xtr[iv], label=ytr[iv], feature_name=feats)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
    return b


t0 = time.time()
print("loading...", flush=True)
Xtr, mtr = load(DS / "train.parquet")
Xev, mev = load(DS / "eval.parquet")
iv = mtr["snapshot_pair"] == VAL
ytr = mtr["label"]
pos_rate = ytr.mean()
print(f"train {Xtr.shape} eval {Xev.shape} pos_rate={pos_rate:.4f} rank_src={RANK_SRC}", flush=True)

res = {"baseline_lambdarank": 0.1255, "oracle_ceiling": 0.6077, "board_gap_to_close": 0.076}

# ---- B: binary control -------------------------------------------------------
b = train(Xtr, ytr, iv, BASE_FEATS)
res["B_binary"] = score_arm("B", b.predict(Xev, num_iteration=b.best_iteration), mev)
print(f"[B] binary                 f={res['B_binary']}  ({time.time()-t0:.0f}s)", flush=True)
json.dump(res, open(W / "compose_levers.json", "w"), indent=1)

# ---- C: + within-protein rank features ---------------------------------------
print("building rank features...", flush=True)
Rtr = add_rank_feats(Xtr, mtr["protein_accession"])
Rev = add_rank_feats(Xev, mev["protein_accession"])
FEATS_C = BASE_FEATS + [f"rank_{c}" for c in RANK_SRC] + [f"z_{c}" for c in RANK_SRC]
XtrC = np.hstack([Xtr, Rtr])
XevC = np.hstack([Xev, Rev])
del Rtr, Rev
b = train(XtrC, ytr, iv, FEATS_C)
res["C_binary_plus_rank"] = score_arm("C", b.predict(XevC, num_iteration=b.best_iteration), mev)
print(f"[C] + within-protein rank  f={res['C_binary_plus_rank']}  ({time.time()-t0:.0f}s)", flush=True)
json.dump(res, open(W / "compose_levers.json", "w"), indent=1)

# ---- D: + class weighting ----------------------------------------------------
spw = float((1 - pos_rate) / pos_rate)
res["scale_pos_weight"] = round(spw, 2)
b = train(XtrC, ytr, iv, FEATS_C, spw=spw)
sD = b.predict(XevC, num_iteration=b.best_iteration)
res["D_plus_class_weight"] = score_arm("D", sD, mev)
print(f"[D] + class weight (spw={spw:.1f}) f={res['D_plus_class_weight']}  ({time.time()-t0:.0f}s)", flush=True)
json.dump(res, open(W / "compose_levers.json", "w"), indent=1)

# ---- E: + drop classifier-only candidates ------------------------------------
kp = Xev[:, BASE_FEATS.index("knn_present")]
cp = Xev[:, BASE_FEATS.index("classifier_present")]
clf_only = (kp <= 0) & (cp > 0)
keep = ~clf_only
res["eval_rows"] = int(len(keep))
res["classifier_only_rows_dropped"] = int(clf_only.sum())
res["fraction_dropped"] = round(float(clf_only.mean()), 4)
res["positives_lost_by_drop"] = int(mev["label"][clf_only].sum())
res["positives_total"] = int(mev["label"].sum())
print(f"  dropping {clf_only.sum()} of {len(keep)} eval rows ({clf_only.mean():.1%}); "
      f"they carry {mev['label'][clf_only].sum()} of {mev['label'].sum()} positives", flush=True)
res["E_plus_drop_clf_only"] = score_arm("E", sD, mev, keep=keep)
print(f"[E] + drop classifier-only f={res['E_plus_drop_clf_only']}  ({time.time()-t0:.0f}s)", flush=True)

best = max((v for k, v in res.items() if k.startswith(("B_", "C_", "D_", "E_")) and v), default=None)
res["best_arm_f_micro_w"] = best
if best:
    res["delta_vs_lambdarank"] = round(best - 0.1255, 4)
    res["still_short_of_board_gap_by"] = round(0.076 - (best - 0.1255), 4)
json.dump(res, open(W / "compose_levers.json", "w"), indent=1)
print("\n=== PK-BPO f_micro_w, composed technique levers ===", flush=True)
for k in ("baseline_lambdarank", "B_binary", "C_binary_plus_rank", "D_plus_class_weight",
          "E_plus_drop_clf_only", "oracle_ceiling"):
    print(f"  {k:24s} {res.get(k)}", flush=True)
if best:
    print(f"  best delta vs lambdarank = {best-0.1255:+.4f}   (gap to close was +0.076)", flush=True)
print("DONE", flush=True)
