"""Measure the evidence arms. Ranking metrics (per-protein AUC + pooled cross-protein AUC) for all arms
(full/noiea/exp) x cells (PK/LK) x aspects; one calibrated f_micro_w check on the wall cells (pk-bpo,
lk-bpo) = SSE as pool-reranker vs the reproduced deployed anchor (flooding not an issue: pool candidates
only). Arm (b) per-protein t0 calibration prior and arm (c) evidence-typed containment are evaluated on
the full-arm SSE score: pooled AUC (cross-protein, where a per-protein prior CAN move the metric) + a
calibrated pool-reranker f_micro_w vs raw SSE. True frame: prop=fill norm=cafa no_orphans toi, PK -known.
Run under PROTEA/.venv.
"""
import json, time, subprocess, tempfile, collections
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)
ROOT = Path("/home/frapercan/Thesis2")
R = ROOT / "repositories/protea-reranker-lab/results/sparse_classifier"
EVAL = R / "percut_rerank/eval.parquet"; PREDDIR = R / "percut_rerank/predictions"
GTDIR = R / "lafa_gt"
OBO = str(ROOT / "protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo")
IA_F = str(ROOT / "protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv")
TOI = str(GTDIR / "groundtruth_terms_of_interest.txt")
OUT = ROOT / "storage/sse_evidence"; PY = str(ROOT / "repositories/PROTEA/.venv/bin/python")
ARMS = ["full", "noiea", "exp"]
ASPECTS = {"mfo": "F", "bpo": "P", "cco": "C"}
NSMAP = {"F": "molecular_function", "P": "biological_process", "C": "cellular_component"}
WALL = [("pk", "bpo"), ("lk", "bpo")]
DEPLOYED_ANCHOR = {("pk","bpo"):0.14351, ("lk","bpo"):0.31323}

par = collections.defaultdict(set); ns = {}; alt = {}; cur = None
for line in open(OBO):
    line = line.rstrip("\n")
    if line == "[Term]": cur = None
    elif line.startswith("id: GO:"): cur = line[4:]
    elif cur and line.startswith("namespace: "): ns[cur] = line[11:]
    elif cur and line.startswith("is_a: GO:"): par[cur].add(line[6:].split(" ! ")[0].strip())
    elif cur and line.startswith("relationship: part_of GO:"):
        par[cur].add(line[len("relationship: part_of "):].split(" ! ")[0].strip())
    elif cur and line.startswith("alt_id: GO:"): alt[line[8:]] = cur

DRIVER = '''
import json, signal
from cafaeval.evaluation import cafa_eval
signal.signal(signal.SIGTERM, signal.SIG_DFL)
df, dfs = cafa_eval("{obo}", "{pd}", "{gt}", ia="{ia}", prop="fill", norm="cafa",
    no_orphans=True, toi_file="{toi}", exclude={known}, max_terms=None, th_step=0.01, n_cpu=6, weighted_only=False)
out = {{}}
for k, v in dfs.items(): out[k] = v.reset_index().to_dict(orient="records")
json.dump(out, open("{o}", "w"), default=str)
'''
def cafa(rows, gt_file, known, nsname):
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "pd"; d.mkdir()
        with (d / "p.tsv").open("w") as fh:
            for p_, g_, v in rows: fh.write(f"{p_}\t{g_}\t{v:.6f}\n")
        raw = Path(td) / "r.json"; drv = Path(td) / "d.py"
        knownrepr = f'"{known}"' if known else "None"
        drv.write_text(DRIVER.format(obo=OBO, pd=str(d), gt=gt_file, ia=IA_F, toi=TOI, known=knownrepr, o=str(raw)))
        r = subprocess.run([PY, str(drv)], capture_output=True, text=True, timeout=7200)
        if r.returncode != 0: print(r.stderr[-1200:], flush=True); return None
        best = None
        for rec in json.loads(raw.read_text()).get("f_micro_w", []):
            if (rec.get("ns") or "") == nsname and rec.get("f_micro_w") is not None: best = rec
        return round(float(best["f_micro_w"]), 5) if best else None

# calibration features + term profile
cf = np.load(OUT / "calib_features.npz", allow_pickle=True)
accs_all = json.load(open(ROOT / "storage/cooc_experiment/generator_frames/accs.json"))
cf_row = cf["acc_row"]; cf_exp = cf["exp_frac"]; cf_nonelec = cf["nonelec_frac"]
cf_total = cf["total"]; cf_breadth = cf["breadth"]; cf_hasexp = cf["has_exp"]
prior = {}   # acc -> dict
for i, r in enumerate(cf_row):
    prior[accs_all[r]] = dict(exp_frac=float(cf_exp[i]), nonelec_frac=float(cf_nonelec[i]),
        logtot=float(np.log1p(cf_total[i])), logbreadth=float(np.log1p(cf_breadth[i])), has_exp=float(cf_hasexp[i]))
tp = np.load(OUT / "term_evidence_direct.npz", allow_pickle=True)
tp_terms = tp["terms"].tolist(); tp_counts = tp["counts"]
tp_tot = tp_counts.sum(1); term_expfrac = {}
for i, t in enumerate(tp_terms):
    term_expfrac[alt.get(t, t)] = float(tp_counts[i, 0] / tp_tot[i]) if tp_tot[i] > 0 else 0.0

def priorvec(p):
    d = prior.get(p)
    if d is None: return np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    return np.array([d["exp_frac"], d["nonelec_frac"], d["logtot"], d["logbreadth"], d["has_exp"]])

RES = {"arms": {}, "arm_b_calibration": {}, "arm_c_evtyped": {}}

# ---- load per-cell pool + deployed + gt once ----
def load_cell(cell):
    cat = cell
    tb = pq.read_table(EVAL, columns=["protein_accession","go_term_id","label","aspect","category"]).to_pandas()
    tb = tb[tb.category == cat].copy(); tb["go"] = tb.go_term_id.map(lambda g: alt.get(g, g))
    dep = pd.read_csv(PREDDIR / cat / f"{cat}.tsv", sep="\t", header=None, names=["prot","term","score"])
    dep["term"] = dep.term.map(lambda g: alt.get(g, g))
    return tb, dep

CELLS_ASP = [(c, a) for c in ["pk", "lk"] for a in ["mfo", "bpo", "cco"]]
pools = {}
for cell in ["pk", "lk"]:
    tb, dep = load_cell(cell); pools[cell] = (tb, dep)

def cell_slice(cell, asp, S_terms, S_prots, S):
    """return arrays for pool candidates: y, sse, rr per (protein,term) that are in the scored slice."""
    tb, dep = pools[cell]; letter = ASPECTS[asp]
    m = tb[tb.aspect == asp]
    tpos = {g: j for j, g in enumerate(S_terms)}; pidx = {p: i for i, p in enumerate(S_prots)}
    rr = {(r.prot, r.term): r.score for r in dep.itertuples()}
    rows = []
    for r in m.itertuples():
        p, g, y = r.protein_accession, r.go, int(r.label)
        if p not in pidx or g not in tpos: continue
        rows.append((p, g, y, float(S[pidx[p], tpos[g]]), float(rr.get((p, g), 0.0))))
    return rows

# ---------------- Part 1+2: arms ranking + calibrated wall check ----------------
for arm in ARMS:
    RES["arms"][arm] = {}
    for asp in ["mfo", "bpo", "cco"]:
        Z = np.load(OUT / f"scores_{arm}_{asp}.npz", allow_pickle=True)
        S_terms = [alt.get(g, g) for g in Z["terms"].tolist()]; S_prots = Z["proteins"].tolist(); S = Z["score"]
        for cell in ["pk", "lk"]:
            rows = cell_slice(cell, asp, S_terms, S_prots, S)
            if not rows: continue
            df = pd.DataFrame(rows, columns=["p","g","y","sse","rr"])
            # per-protein AUC
            se_a, rr_a = [], []
            for p, gdf in df.groupby("p"):
                y = gdf.y.values
                if y.sum() == 0 or y.sum() == len(y) or len(y) < 5: continue
                se_a.append(roc_auc_score(y, gdf.sse.values)); rr_a.append(roc_auc_score(y, gdf.rr.values))
            pooled = roc_auc_score(df.y.values, df.sse.values) if df.y.nunique() > 1 else None
            pooled_rr = roc_auc_score(df.y.values, df.rr.values) if df.y.nunique() > 1 else None
            RES["arms"][arm][f"{cell}-{asp}"] = {
                "n_prot_auc": len(se_a),
                "sse_perprot_auc": round(float(np.mean(se_a)), 4) if se_a else None,
                "rr_perprot_auc": round(float(np.mean(rr_a)), 4) if rr_a else None,
                "sse_pooled_auc": round(float(pooled), 4) if pooled is not None else None,
                "rr_pooled_auc": round(float(pooled_rr), 4) if pooled_rr is not None else None,
                "n_pool_rows": len(df), "n_pos": int(df.y.sum())}
            log(f"[{arm}] {cell}-{asp}: SSE perprot {RES['arms'][arm][f'{cell}-{asp}']['sse_perprot_auc']} "
                f"rr {RES['arms'][arm][f'{cell}-{asp}']['rr_perprot_auc']} pooled {RES['arms'][arm][f'{cell}-{asp}']['sse_pooled_auc']}")
    # calibrated wall check: SSE as pool-reranker vs anchor (pk-bpo, lk-bpo)
    RES["arms"][arm]["calibrated_wall"] = {}
    for cell, asp in WALL:
        Z = np.load(OUT / f"scores_{arm}_{asp}.npz", allow_pickle=True)
        S_terms = [alt.get(g, g) for g in Z["terms"].tolist()]; S_prots = Z["proteins"].tolist(); S = Z["score"]
        rows = cell_slice(cell, asp, S_terms, S_prots, S)
        gt_file = str(GTDIR / f"groundtruth_{cell.upper()}.tsv")
        known = str(GTDIR / f"groundtruth_{cell.upper()}_known.tsv") if cell == "pk" else None
        pred = [(p, g, s) for (p, g, y, s, rv) in rows]
        f_sse = cafa(pred, gt_file, known, NSMAP[ASPECTS[asp]])
        if arm == "full":  # reproduce anchor once via deployed reranker over same pool
            predrr = [(p, g, rv) for (p, g, y, s, rv) in rows]
            f_anchor = cafa(predrr, gt_file, known, NSMAP[ASPECTS[asp]])
            RES.setdefault("anchor_poolreranker", {})[f"{cell}-{asp}"] = f_anchor
        RES["arms"][arm]["calibrated_wall"][f"{cell}-{asp}"] = {
            "sse_poolreranker_f": f_sse, "deployed_anchor_ref": DEPLOYED_ANCHOR.get((cell, asp))}
        log(f"[{arm}] calibrated {cell}-{asp}: SSE-pool-reranker f {f_sse}")
    json.dump(RES, open(OUT / "measure.json", "w"), indent=1, default=float)

# ---------------- Part 3: arm (b) per-protein t0 calibration prior (on FULL arm) ----------------
for cell, asp in WALL:
    Z = np.load(OUT / f"scores_full_{asp}.npz", allow_pickle=True)
    S_terms = [alt.get(g, g) for g in Z["terms"].tolist()]; S_prots = Z["proteins"].tolist(); S = Z["score"]
    rows = cell_slice(cell, asp, S_terms, S_prots, S)
    df = pd.DataFrame(rows, columns=["p","g","y","sse","rr"])
    PR = np.stack([priorvec(p) for p in df.p.values])
    X_base = df.sse.values.reshape(-1, 1)
    X_prior = np.hstack([X_base, PR])
    y = df.y.values
    cv_base = cross_val_predict(LogisticRegression(max_iter=1000), X_base, y, cv=5, method="predict_proba")[:, 1]
    cv_prior = cross_val_predict(LogisticRegression(max_iter=1000), X_prior, y, cv=5, method="predict_proba")[:, 1]
    auc_base = roc_auc_score(y, cv_base); auc_prior = roc_auc_score(y, cv_prior)
    gt_file = str(GTDIR / f"groundtruth_{cell.upper()}.tsv")
    known = str(GTDIR / f"groundtruth_{cell.upper()}_known.tsv") if cell == "pk" else None
    f_base = cafa([(df.p.iloc[i], df.g.iloc[i], cv_base[i]) for i in range(len(df))], gt_file, known, NSMAP[ASPECTS[asp]])
    f_prior = cafa([(df.p.iloc[i], df.g.iloc[i], cv_prior[i]) for i in range(len(df))], gt_file, known, NSMAP[ASPECTS[asp]])
    RES["arm_b_calibration"][f"{cell}-{asp}"] = {
        "pooled_auc_sse_only": round(float(auc_base), 4), "pooled_auc_sse_plus_prior": round(float(auc_prior), 4),
        "delta_pooled_auc": round(float(auc_prior - auc_base), 4),
        "f_sse_only_calibrated": f_base, "f_sse_plus_prior_calibrated": f_prior,
        "delta_f": round((f_prior - f_base), 5) if (f_prior and f_base) else None,
        "n_prot_with_prior": int(sum(1 for p in df.p.unique() if p in prior))}
    log(f"arm-b {cell}-{asp}: pooledAUC {auc_base:.4f}->{auc_prior:.4f} f {f_base}->{f_prior}")
    json.dump(RES, open(OUT / "measure.json", "w"), indent=1, default=float)

# ---------------- Part 4: arm (c) evidence-typed containment (term exp-frac scaling, FULL arm) ----------------
for cell, asp in WALL:
    Z = np.load(OUT / f"scores_full_{asp}.npz", allow_pickle=True)
    S_terms = [alt.get(g, g) for g in Z["terms"].tolist()]; S_prots = Z["proteins"].tolist(); S = Z["score"]
    rows = cell_slice(cell, asp, S_terms, S_prots, S)
    df = pd.DataFrame(rows, columns=["p","g","y","sse","rr"])
    rel = df.g.map(lambda g: term_expfrac.get(g, 0.0)).values
    df["sse_c"] = df.sse.values * rel
    se_raw, se_c = [], []
    for p, gdf in df.groupby("p"):
        yv = gdf.y.values
        if yv.sum() == 0 or yv.sum() == len(yv) or len(yv) < 5: continue
        se_raw.append(roc_auc_score(yv, gdf.sse.values)); se_c.append(roc_auc_score(yv, gdf.sse_c.values))
    pooled_raw = roc_auc_score(df.y.values, df.sse.values); pooled_c = roc_auc_score(df.y.values, df.sse_c.values)
    gt_file = str(GTDIR / f"groundtruth_{cell.upper()}.tsv")
    known = str(GTDIR / f"groundtruth_{cell.upper()}_known.tsv") if cell == "pk" else None
    f_c = cafa([(df.p.iloc[i], df.g.iloc[i], df.sse_c.iloc[i]) for i in range(len(df))], gt_file, known, NSMAP[ASPECTS[asp]])
    RES["arm_c_evtyped"][f"{cell}-{asp}"] = {
        "sse_perprot_auc": round(float(np.mean(se_raw)), 4), "sse_c_perprot_auc": round(float(np.mean(se_c)), 4),
        "delta_perprot_auc": round(float(np.mean(np.array(se_c) - np.array(se_raw))), 4),
        "sse_pooled_auc": round(float(pooled_raw), 4), "sse_c_pooled_auc": round(float(pooled_c), 4),
        "sse_c_poolreranker_f": f_c}
    log(f"arm-c {cell}-{asp}: perprotAUC {np.mean(se_raw):.4f}->{np.mean(se_c):.4f} pooled {pooled_raw:.4f}->{pooled_c:.4f} f {f_c}")
    json.dump(RES, open(OUT / "measure.json", "w"), indent=1, default=float)

json.dump(RES, open(OUT / "measure.json", "w"), indent=1, default=float)
log("SSE EV MEASURE DONE")
print(json.dumps(RES, indent=1, default=float))
