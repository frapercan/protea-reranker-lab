"""PHASE 2 + 3 of the transition-separability probe.

Rescore the DEPLOYED candidate set (pure rescoring, no new candidates -> dodges the cafaeval
scale/volume artifact) with the best known-conditioned OOF model score from Phase 1 (grouped-CV
by protein, so no protein is scored by a model that trained on it). Score each arm with cafaeval
in the TRUE frame (same flags as score_cafaeval.py, -known for PK).

Arm A scores the deployed prediction file VERBATIM (reproduces the incumbent cell number, same
frame + parser). Blended/rerank/rank-avg arms modify only BP-namespace rows, per deployed row,
using a deduped (prot,term)-level model map. Includes a FIXED-SCORE control (Phase 3).
Runs detached; writes storage/regen_headline/phase2_cafaeval.json incrementally.
"""
import json, tempfile, time
from pathlib import Path
import numpy as np, pandas as pd
from cafaeval.evaluation import cafa_eval

t0 = time.time()
ROOT = Path("/home/frapercan/Thesis2")
PREDDIR = ROOT / "repositories/protea-reranker-lab/results/sparse_classifier/percut_rerank/predictions"
GTDIR = ROOT / "repositories/protea-reranker-lab/results/sparse_classifier/lafa_gt"
OBO = str(ROOT / "protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo")
IA = str(ROOT / "protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv")
TOI = str(GTDIR / "groundtruth_terms_of_interest.txt")
OUT = ROOT / "storage/regen_headline"
GT = {"lk": (str(GTDIR / "groundtruth_LK.tsv"), None),
      "pk": (str(GTDIR / "groundtruth_PK.tsv"), str(GTDIR / "groundtruth_PK_known.tsv"))}

ns = {}; cur = None
for line in open(OBO):
    line = line.rstrip("\n")
    if line == "[Term]": cur = None
    elif line.startswith("id: GO:"): cur = line[4:]
    elif cur and line.startswith("namespace: "): ns[cur] = line[11:]
BP = {t for t, n in ns.items() if n == "biological_process"}


def score_bpo(pred_df, cat):
    with tempfile.TemporaryDirectory() as tmp:
        pd_dir = Path(tmp) / "pd"; pd_dir.mkdir()
        pred_df[["prot", "term", "score"]].to_csv(pd_dir / "p.tsv", sep="\t", header=False,
                                                  index=False, float_format="%.6f")
        gt_file, known = GT[cat]
        _, dfs = cafa_eval(OBO, str(pd_dir), gt_file, ia=IA, no_orphans=True, norm="cafa",
                           prop="fill", exclude=known, toi_file=TOI, th_step=0.01, n_cpu=8)
    best = dfs["f_micro_w"].reset_index()
    row = best[best.ns == "biological_process"]
    if len(row) == 0:
        return None, None
    r = row.iloc[0]
    return float(r["f_micro_w"]), float(r["tau"])


def minmax(x):
    x = np.nan_to_num(np.asarray(x, float)); lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def rankn(x):
    return pd.Series(x).rank(method="average").values / len(x)


PRIMARY = "gbdt_known_transition"
results = {}
for cat in ["pk", "lk"]:          # PK first (biggest AUC gap) per coordinator
    print(f"\n[{time.time()-t0:.0f}s] ===== {cat.upper()}-BPO =====", flush=True)
    ph1 = pd.read_parquet(OUT / f"phase1_scores_{cat}.parquet").rename(
        columns={"protein_accession": "prot", "go_term_id": "term"})
    ph1 = ph1.drop_duplicates(["prot", "term"])
    full = pd.read_csv(PREDDIR / cat / f"{cat}.tsv", sep="\t", header=None,
                       names=["prot", "term", "score"])
    full["isbp"] = full.term.isin(BP)
    for col in ["gbdt_known_transition", "gbdt_known"]:
        full = full.merge(ph1[["prot", "term", col]], on=["prot", "term"], how="left")
    cov = full.loc[full.isbp, PRIMARY].notna().mean()
    print(f"  deployed rows {len(full):,}; BP rows {int(full.isbp.sum()):,}; "
          f"model-map coverage on BP {100*cov:.1f}%", flush=True)
    bp = full.isbp.values
    dep = full.score.values.astype(float)
    dep_rank = np.zeros(len(full)); dep_rank[bp] = rankn(dep[bp])
    mm = {}
    for col in ["gbdt_known_transition", "gbdt_known"]:
        v = np.zeros(len(full)); mask = (bp & full[col].notna().values)
        v[mask] = minmax(full.loc[mask, col].values); mm[col] = v
    mrk = {}
    for col in mm:
        r = np.zeros(len(full)); r[bp] = rankn(mm[col][bp]); mrk[col] = r

    cell = {"deployed_rows": int(len(full)), "bp_rows": int(bp.sum()),
            "model_coverage_bp_pct": round(100 * float(cov), 2), "primary_model": PRIMARY,
            "leakage_guard": "OOF grouped-CV by protein_accession (GroupKFold, 5 folds); no protein "
                             "appears in both train and predict fold.", "arms": {}}

    def run(tag, new_scores):
        sub = full.assign(score=np.clip(new_scores, 0, 1))
        f, tau = score_bpo(sub, cat)
        cell["arms"][tag] = {"f_micro_w": round(f, 5) if f is not None else None, "tau": tau}
        print(f"    {tag:36s} f={f} tau={tau}  ({time.time()-t0:.0f}s)", flush=True)
        return f

    f_dep = run("A_deployed_incumbent", dep)
    cell["incumbent_f"] = round(f_dep, 5) if f_dep else None
    best = {"tag": None, "f": -1.0, "delta": None}
    def consider(tag, f):
        if f is not None and f > best["f"]:
            best.update(tag=tag, f=round(f, 5), delta=round(f - f_dep, 5))

    for col in ["gbdt_known_transition", "gbdt_known"]:
        for w in [0.5, 1.0, 2.0]:
            s = dep.copy(); s[bp] = dep[bp] + w * mm[col][bp]
            consider(f"B_{col}_add_w{w}", run(f"B_{col}_add_w{w}", s))
        s = dep.copy(); s[bp] = mm[col][bp]
        consider(f"C_{col}_rerank", run(f"C_{col}_rerank", s))
        s = dep.copy(); s[bp] = 0.5 * dep_rank[bp] + 0.5 * mrk[col][bp]
        consider(f"D_{col}_rankavg", run(f"D_{col}_rankavg", s))

    s = dep.copy(); s[bp] = 0.5
    f_const = run("FIXEDSCORE_const_bp", s)
    cell["best_positive_arm"] = best
    cell["fixedscore_const_delta"] = round(f_const - f_dep, 5) if (f_const and f_dep) else None
    cell["transfew_target"] = {"lk": 0.512, "pk": 0.294}[cat]
    cell["gap_to_transfew"] = {"lk": 0.072, "pk": 0.076}[cat]
    results[cat] = cell
    json.dump(results, open(OUT / "phase2_cafaeval.json", "w"), indent=2)

json.dump(results, open(OUT / "phase2_cafaeval.json", "w"), indent=2)
print(f"\n[{time.time()-t0:.0f}s] DONE wrote phase2_cafaeval.json", flush=True)
print(json.dumps(results, indent=2))
