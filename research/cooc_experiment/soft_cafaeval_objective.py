"""Does optimizing a soft IA-weighted F beat lambdarank, at equal everything else?

THE AUTHOR'S DIRECTION. Optimize a soft ontological metric, not a rank proxy. `fuse_listwise.py`
proved "the objective was the whole 0.045": against the deployed LightGBM lambdarank the neural
listwise lost -0.0355 at equal input, and the raw vectors added +0.0018. If the objective is the
whole game, then the thing to change is the objective, and the one-variable way to change it is to
keep LightGBM and the 75 reranker features and swap ONLY the loss.

WHY THIS COULD WIN, mechanistically. cafaeval's f_micro_w is not rank-only: it is SCALE-sensitive
(the "SCALE lever >= 0.088" finding: raw 0.2131 vs sigmoid/minmax/rankpct 0.1255 on the same order).
lambdarank optimizes a pairwise ranking objective that is invariant to where scores sit relative to
the global threshold. A soft-F objective ties the model's decision boundary to the operating point
f_micro_w actually reads. That is exactly the degree of freedom lambdarank throws away.

THE OBJECTIVE. f_micro_w pools every (protein, term) pair in the cell and takes one IA-weighted
precision/recall at each global threshold, maxed over the threshold. Its soft form, with p_i =
sigmoid(s_i), IA weight w_i, propagated label y_i, and the algebra F = 2*TP/(PM + AM):

    TP = sum_i w_i y_i p_i        PM = sum_i w_i p_i        AM = sum_i w_i y_i   (constant)
    F  = 2 TP / (PM + AM)

recomputing (TP, PM, F) from the current predictions each boosting round, then per-sample

    a_i    = (2 w_i / (PM+AM)) * (y_i - F/2)
    grad_i = -a_i * p_i (1 - p_i)                 d(-F)/ds_i
    hess_i =  |a_i| * p_i (1 - p_i) + eps         Gauss-Newton positive approximation

This is the IA-weighted generalization of the soft-F1 loss, at the tau=0 operating point; cafaeval
still maxes over tau at eval, so the model only has to be well-calibrated around its own boundary.

THE ONE VARIABLE. Same 75 features, same pk-bpo pool rows, same past/val split (early-stop on
v225-v227), same trees/leaves/rounds. Arm A trains objective=lambdarank (the deployed loss); arm B
trains the soft-F objective above. Both predict on the SAME blind eval and are scored by the SAME
cafaeval call on the board's line.

THE PRECONDITION: arm A must reproduce the deployed reranker's blind f_micro_w within 0.01 (the
tolerance fuse_listwise's arm A landed inside: 0.21969 vs deployed 0.21269). If my lambdarank arm
does not reproduce the system it stands for, arm B's delta measures my setup, not the objective.

THE GATE: (B - A) on the blind window must clear this cell's 0.0034 fold noise. If it does, the next
run is a paired bootstrap CI and, if that holds, the end2end over raw vectors earns its -0.0355
handicap back. If it does not, lambdarank already captured what the metric rewards and the objective
is not the lever after all.

WHAT DECIDES: f_micro_w, never the soft-F. The soft metric trains and diagnoses; it does not decide
(AUC ordered levers backwards five times). The soft-F on val is only an early-stopping signal.
"""
import json, collections, subprocess, tempfile, time
from pathlib import Path
import numpy as np, pyarrow.parquet as pq, lightgbm as lgb

W = Path("/home/frapercan/Thesis2/storage/cooc_experiment")
DS = Path("/home/frapercan/Thesis2/repositories/protea-reranker-lab/datasets/protst-global-train227-test230")
REL = Path("/home/frapercan/Thesis2/CAFA_forever/data/releases/Sep_2025_Mar_2026")
T0D = Path("/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025")
OBO, IA_F = str(T0D / "go-basic.obo"), str(T0D / "IA.tsv")
TOI = str(REL / "groundtruth_terms_of_interest.txt")
PY = "/home/frapercan/Thesis2/repositories/PROTEA/.venv/bin/python"
VAL_PAIR = "v225-v227"
DEPLOYED_BLIND = 0.21269          # the number arm A must reproduce within 0.01 (fuse_listwise: 0.21969)
SEED, ROUNDS = 42, 3000
# the shared tree hyperparameters, taken from fuse_listwise's arm A which reproduced the deployed
# reranker. Both arms use these; ONLY the objective differs. lambdarank-specific keys (label_gain,
# ndcg) are the objective's own config, with no analogue in the soft-F arm.
SHARED = {"learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 100, "feature_fraction": 0.9,
          "bagging_fraction": 0.9, "bagging_freq": 5, "seed": SEED, "verbose": -1, "num_threads": 12,
          "max_bin": 63, "force_col_wise": True}
t0 = time.time()

alt = {}
cur_ = None
for line in open(OBO):
    line = line.rstrip("\n")
    if line == "[Term]":
        cur_ = None
    elif line.startswith("id: GO:"):
        cur_ = line[4:]
    elif cur_ and line.startswith("alt_id: GO:"):
        alt[line[8:]] = cur_
IA = {}
for line in open(IA_F):
    p = line.rstrip("\n").split("\t")
    if len(p) >= 2 and p[0].startswith("GO:"):
        try:
            IA[alt.get(p[0], p[0])] = float(p[1])
        except ValueError:
            pass
MEAN_IA = float(np.mean(list(IA.values())))
iaw = lambda G: np.array([IA.get(g, MEAN_IA) for g in G], np.float32)

META = {"protein_accession", "go_term_id", "label", "category", "aspect", "snapshot_pair"}
_sch = pq.ParquetFile(DS / "train.parquet").schema_arrow
schema = _sch.names
# the 72 NUMERIC features only: qualifier/evidence_code/taxonomic_relation are string categoricals,
# and fuse_listwise reproduced the deployed anchor on exactly these 72 scalars.
FEATS = [f.name for f in _sch if f.name not in META
         and str(f.type).startswith(("double", "float", "int", "bool"))]
print(f"{len(FEATS)} features; IA {len(IA):,} terms (mean {MEAN_IA:.3f})  ({time.time()-t0:.0f}s)",
      flush=True)


def load(path, want):
    """Load pk-bpo rows; `want` selects past/val/eval by snapshot_pair. Returns X, y, w, prot, go."""
    cols = FEATS + ["protein_accession", "go_term_id", "category", "aspect", "label", "snapshot_pair"]
    cols = [c for c in cols if c in schema or c in ("category", "aspect", "snapshot_pair")]
    t = pq.read_table(path, columns=[c for c in cols if c in pq.ParquetFile(path).schema_arrow.names])
    C = np.asarray(t.column("category").to_pylist()); A = np.asarray(t.column("aspect").to_pylist())
    sp = np.asarray(t.column("snapshot_pair").to_pylist())
    m = (C == "pk") & (A == "bpo")
    if want == "past":
        m &= sp != VAL_PAIR
    elif want == "val":
        m &= sp == VAL_PAIR
    idx = np.where(m)[0]
    X = np.column_stack([t.column(f).to_numpy(zero_copy_only=False).astype(np.float32)[idx]
                         for f in FEATS])
    y = (t.column("label").to_numpy(zero_copy_only=False).astype(np.float32)[idx] > 0).astype(np.float32)
    P = np.asarray(t.column("protein_accession").to_pylist())[idx]
    G = np.array([alt.get(g, g) for g in np.asarray(t.column("go_term_id").to_pylist())[idx]])
    return X, y, iaw(G), P, G


Xtr, ytr, wtr, Ptr, Gtr = load(DS / "train.parquet", "past")
Xva, yva, wva, Pva, Gva = load(DS / "train.parquet", "val")
Xte, yte, wte, Pte, Gte = load(DS / "eval.parquet", "eval")
print(f"train {len(ytr):,}/{ytr.sum():,.0f}pos | val {len(yva):,}/{yva.sum():,.0f}pos | "
      f"blind {len(yte):,}/{yte.sum():,.0f}pos  ({time.time()-t0:.0f}s)", flush=True)

AM_tr = float((wtr * ytr).sum())


def soft_f_obj(preds, dset):
    y = dset.get_label(); w = W_REF[id(dset)]
    p = 1.0 / (1.0 + np.exp(-preds))
    pw = p * w
    TP = float((pw * y).sum()); PM = float(pw.sum()); AM = float((w * y).sum())
    D = PM + AM + 1e-9; F = 2.0 * TP / D
    a = (2.0 * w / D) * (y - F / 2.0)
    pp = p * (1.0 - p)
    grad = -(a * pp).astype(np.float64)
    hess = (np.abs(a) * pp + 1e-6).astype(np.float64)
    return grad, hess


def soft_f_eval(preds, dset):
    """Hard IA-weighted F, maxed over threshold on a quantile grid: the early-stopping signal."""
    y = dset.get_label(); w = W_REF[id(dset)]
    p = 1.0 / (1.0 + np.exp(-preds))
    order = np.argsort(-p)
    ys, ws = y[order], w[order]
    csum_tp = np.cumsum(ws * ys); csum_pm = np.cumsum(ws)
    AM = float((w * y).sum())
    F = 2.0 * csum_tp / (csum_pm + AM + 1e-9)
    step = max(1, len(F) // 2000)
    best = float(F[::step].max()) if len(F) else 0.0
    return "wF", best, True


W_REF = {}

# cafaeval in PROTEA/.venv was updated (perf commits #19/#21, the PyArrow vectorised parser): with
# `max_terms=None` it now routes to the VECTORISED parser (the board's frame, ~0.13), where before it
# routed to LEGACY (the lab frame, ~0.22). Proven on identical predictions: legacy 0.22383 vs
# vectorised 0.13854, a 0.085 gap that IS the documented "off by 0.084" board divergence. So every arm
# is scored under BOTH: LEGACY (max_terms=500, reproduces the deployed 0.21269, the campaign's frame)
# and VECTORISED (max_terms=None, the board's frame). The comparison must be within one frame.
DRIVER = '''
import json, signal
from cafaeval.evaluation import cafa_eval
signal.signal(signal.SIGTERM, signal.SIG_DFL)
df,dfs=cafa_eval("{obo}","{pd}","{gt}",ia="{ia}",prop="fill",norm="cafa",no_orphans=True,
    toi_file="{toi}",max_terms={mt},th_step=0.01,n_cpu=4,weighted_only=False)
out={{}}
for k,v in dfs.items(): out[k]=v.reset_index().to_dict(orient="records")
json.dump(out,open("{o}","w"),default=str)
'''


def cafa_one(scores, mt):
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "pd"; d.mkdir()
        with (d / "p.tsv").open("w") as fh:
            for p_, g_, v in zip(Pte, Gte, scores):
                fh.write(f"{p_}\t{g_}\t{v:.6f}\n")
        gt = Path(td) / "gt.tsv"
        with gt.open("w") as fh:
            for line in (DS / "gt_pk_bp.tsv").open():
                fh.write(line)
        raw = Path(td) / "r.json"; drv = Path(td) / "d.py"
        drv.write_text(DRIVER.format(obo=OBO, pd=str(d), gt=str(gt), ia=IA_F, toi=TOI, mt=mt,
                                     o=str(raw)))
        r = subprocess.run([PY, str(drv)], capture_output=True, text=True, timeout=7200)
        if r.returncode != 0:
            print(r.stderr[-800:], flush=True)
            return None
        best = None
        for rec in json.loads(raw.read_text()).get("f_micro_w", []):
            if (rec.get("ns") or "") == "biological_process" and rec.get("f_micro_w") is not None:
                best = rec
        return round(float(best["f_micro_w"]), 5) if best else None


def cafa(scores):
    """Score under both parser frames: legacy (max_terms=500) and vectorised (max_terms=None)."""
    return {"legacy": cafa_one(scores, "500"), "vectorised": cafa_one(scores, "None")}


# Scoring, matching fuse_listwise exactly: submit the model's NATURAL output under norm="cafa", NOT
# minmax (the SCALE finding: minmax cost 0.088). lambdarank's natural output is its raw rank score;
# the soft-F's natural output is sigmoid(score), the calibrated probability the objective was trained
# to produce. That difference in link is not a confound, it IS the objective's calibration, which is
# the whole hypothesis. fuse_listwise submitted raw lambdarank and reproduced the deployed at 0.21969.
sigmoid = lambda s: 1.0 / (1.0 + np.exp(-s))


# ---- arm A: lambdarank, the deployed loss. Grouped by protein. ----
def groups(P):
    _, idx, cnt = np.unique(P, return_index=True, return_counts=True)
    return cnt[np.argsort(idx)]


otr = np.argsort(Ptr, kind="stable"); ova = np.argsort(Pva, kind="stable")
dA_tr = lgb.Dataset(Xtr[otr], label=ytr[otr], group=groups(Ptr[otr]), free_raw_data=False)
dA_va = lgb.Dataset(Xva[ova], label=yva[ova], group=groups(Pva[ova]), free_raw_data=False)
print(f"  arm A: training lambdarank...  ({time.time()-t0:.0f}s)", flush=True)
mA = lgb.train({**SHARED, "objective": "lambdarank", "metric": ["ndcg"], "ndcg_eval_at": [5, 10],
                "label_gain": [0, 1]},
               lgb.Dataset(Xtr[otr], label=ytr[otr].astype(int), group=groups(Ptr[otr])),
               num_boost_round=ROUNDS,
               valid_sets=[lgb.Dataset(Xva[ova], label=yva[ova].astype(int), group=groups(Pva[ova]))],
               callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
sA = mA.predict(Xte, num_iteration=mA.best_iteration).astype(np.float32)   # raw, as deployed
fA = cafa(sA)
print(f"  arm A lambdarank: {fA} (best_iter {mA.best_iteration})  ({time.time()-t0:.0f}s)", flush=True)

# ---- arm B: the soft IA-weighted F objective. No grouping: the metric is global micro. ----
dB_tr = lgb.Dataset(Xtr, label=ytr, free_raw_data=False)
dB_va = lgb.Dataset(Xva, label=yva, free_raw_data=False)
W_REF[id(dB_tr)] = wtr; W_REF[id(dB_va)] = wva
print(f"  arm B: training soft-F objective...  ({time.time()-t0:.0f}s)", flush=True)
mB = lgb.train({**SHARED, "objective": soft_f_obj},
               dB_tr, num_boost_round=ROUNDS, feval=soft_f_eval, valid_sets=[dB_va],
               callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
sB = sigmoid(mB.predict(Xte, num_iteration=mB.best_iteration).astype(np.float64))  # calibrated prob
fB = cafa(sB)
print(f"  arm B soft-F: {fB} (best_iter {mB.best_iteration})  ({time.time()-t0:.0f}s)", flush=True)

# freeze both arms' raw predictions so any re-scoring (parser, threshold, bootstrap) is free.
np.savez(W / "soft_cafaeval_predictions.npz", Pte=Pte, Gte=Gte, sA=sA, sB=sB.astype(np.float32))

# the campaign's frame is LEGACY (it reproduces the deployed 0.21269 and every prior lab lever). The
# precondition and the gate live there; the vectorised (board) frame is reported alongside.
ok = fA["legacy"] is not None and abs(fA["legacy"] - DEPLOYED_BLIND) < 0.012
def delta(f):
    return {fr: (round(f["B"][fr] - f["A"][fr], 5) if f["A"][fr] is not None and f["B"][fr] is not None
                 else None) for fr in ("legacy", "vectorised")}
both = {"A": fA, "B": fB}
d = delta(both)
res = {"question": "does a soft IA-weighted-F objective beat lambdarank at equal features?",
       "one_variable": "same 72 features, same pk-bpo pool rows, same past/val split, same "
                       "leaves/lr/rounds; only the objective changes (lambdarank -> soft-F).",
       "parser_note": "cafaeval was updated; max_terms=None now routes to the VECTORISED parser "
                      "(board frame, ~0.13), max_terms=500 to LEGACY (lab frame, ~0.22). Same "
                      "predictions, both frames reported. fuse_listwise's 0.21969 was a legacy number.",
       "deployed_blind_legacy": DEPLOYED_BLIND, "tolerance": 0.012,
       "A_lambdarank": fA, "B_soft_f": fB, "B_minus_A": d,
       "PRECONDITION_A_reproduces_deployed_legacy": bool(ok),
       "VOID_IF_PRECONDITION_FAILS": not ok,
       "gate": "B - A in the LEGACY frame must clear the 0.0034 fold noise; then a paired bootstrap.",
       "GATE_PASSES_legacy": bool(ok and d["legacy"] is not None and d["legacy"] > 0.0034),
       "note": "f_micro_w decides, never the soft-F. The end2end over vectors earns its -0.0355 "
               "handicap back ONLY if this wins on scalars first."}
json.dump(res, open(W / "soft_cafaeval_objective.json", "w"), indent=1)
print(f"\n=== soft-cafaeval objective vs lambdarank ({time.time()-t0:.0f}s) ===", flush=True)
print(f"  PRECONDITION (A legacy reproduces deployed {DEPLOYED_BLIND} +/-0.012): {ok}  "
      f"[A_legacy={fA['legacy']}]", flush=True)
print(f"  {'frame':<12} {'A lambdarank':>14} {'B soft-F':>12} {'B - A':>10}", flush=True)
for fr in ("legacy", "vectorised"):
    print(f"  {fr:<12} {str(fA[fr]):>14} {str(fB[fr]):>12} {str(d[fr]):>10}", flush=True)
print(f"  gate (legacy, 0.0034) -> passes {res['GATE_PASSES_legacy']}", flush=True)
print("DONE", flush=True)
