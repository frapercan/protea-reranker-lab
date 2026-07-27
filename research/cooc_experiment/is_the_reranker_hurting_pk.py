"""Is our own reranker costing us the PK-BPO cell?

Lab #113 (2026-06-26) measured the per-category reranker REGRESSING on PK across the
board (-0.027 to -0.173; pk-bpo precision -0.196) and recommended "a per-category gate:
rerank NK/LK, raw-KNN PK". Its own note says the TEST frame v227->v230 needed a later
export. That export never happened and the lab has had no commit since. So the
recommendation has never been tested on the frame we are judged on. This tests it.

WHAT THE POOL ALLOWS, measured first, because it reframes the question:
  * `distance` is NaN on 43.4% of pk-bpo rows: those candidates came from the CLASSIFIER,
    not the KNN. Raw KNN cannot score them at all.
  * rows the KNN can speak for (knn_present>0) = 56.6%, and they carry only
    3,333 of 15,248 positives (21.9%).
So "turn the reranker off and serve raw KNN" is not available on this cell: the KNN pool
alone cannot reach 78% of the answers. The honest question is narrower.

ARMS. All scored by the same cafaeval against the FULL gt, so a pool that reaches fewer
answers is penalised for it rather than flattered.

  R_all    the deployed reranker over the whole pool          = the anchor (~0.2131)
  R_knn    the deployed reranker, KNN-reachable rows only     isolates the pool
  K_dist   raw KNN score 1 - distance/2 on those same rows    the documented baseline
  K_vote   neighbour vote fraction on those same rows         the other raw KNN signal

READ IT AS:
  * R_knn vs K_dist / K_vote  -> on the rows the KNN can speak for, does the reranker
    ADD over a plain similarity score, or does it destroy it? This is lab #113's claim,
    tested on the TEST frame at last.
  * R_all vs R_knn -> what the classifier-proposed candidates are worth once the reranker
    orders them.

Scores are kept in [0,1] deliberately: cafaeval sweeps tau on a fixed [0,1] grid, and a
transform that moves the score distribution is part of the measurement, not a formatting
choice. That is the rankpct lesson and it cost this campaign 0.088 once already.
`neighbor_vote_fraction` maxes at 3.83, so K_vote is divided by its max: a single global
monotone rescale, applied to the whole arm, exactly as the raw arm is left alone.
"""
import json, subprocess, tempfile, time
from pathlib import Path
import numpy as np, pyarrow.parquet as pq

W = Path("/home/frapercan/Thesis2/storage/cooc_experiment")
DS = Path("/home/frapercan/Thesis2/repositories/protea-reranker-lab/datasets/protst-global-train227-test230")
OBO = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
IA = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv"
PY = "/home/frapercan/Thesis2/repositories/PROTEA/.venv/bin/python"

# the deployed booster's own frozen scores
t = pq.read_table(W / "rerank_out" / "eval_scores.parquet")
cat = np.asarray(t.column("category").to_pylist())
asp = np.asarray(t.column("aspect").to_pylist())
m = (cat == "pk") & (asp == "bpo")
rp = np.asarray(t.column("protein_accession").to_pylist())[m]
rg = np.asarray(t.column("go_term_id").to_pylist())[m]
rr = t.column("reranker_score").to_numpy(zero_copy_only=False).astype(np.float64)[m]

# the frozen dataset, for the raw KNN signals
cols = ["category", "aspect", "protein_accession", "go_term_id", "label",
        "distance", "neighbor_vote_fraction", "knn_present"]
d = pq.read_table(DS / "eval.parquet", columns=cols)
dc = np.asarray(d.column("category").to_pylist())
da = np.asarray(d.column("aspect").to_pylist())
dm = (dc == "pk") & (da == "bpo")
dp = np.asarray(d.column("protein_accession").to_pylist())[dm]
dg = np.asarray(d.column("go_term_id").to_pylist())[dm]
dist = d.column("distance").to_numpy(zero_copy_only=False).astype(np.float64)[dm]
vote = d.column("neighbor_vote_fraction").to_numpy(zero_copy_only=False).astype(np.float64)[dm]
kp = d.column("knn_present").to_numpy(zero_copy_only=False).astype(np.float64)[dm]

# align the two by (protein, term)
key_r = {(a, b): i for i, (a, b) in enumerate(zip(rp, rg))}
idx = np.array([key_r.get((a, b), -1) for a, b in zip(dp, dg)])
ok = idx >= 0
print(f"aligned {ok.sum():,} of {len(dp):,} dataset rows to the deployed booster's scores", flush=True)
rr_al = np.full(len(dp), np.nan)
rr_al[ok] = rr[idx[ok]]

knn_rows = (kp > 0) & ~np.isnan(dist)
print(f"KNN-reachable rows: {knn_rows.sum():,} ({knn_rows.mean():.1%})", flush=True)

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


def score(name, sel, s):
    with tempfile.TemporaryDirectory() as td:
        pd_ = Path(td) / "pd"
        pd_.mkdir()
        n = 0
        with (pd_ / f"{name}.tsv").open("w") as fh:
            for p_, g_, v in zip(dp[sel], dg[sel], s):
                if np.isnan(v):
                    continue
                fh.write(f"{p_}\t{g_}\t{v:.6f}\n")
                n += 1
        gt = Path(td) / "gt.tsv"
        with gt.open("w") as fh:
            for p_, g_ in TRUTH:
                fh.write(f"{p_}\t{g_}\n")
        raw = Path(td) / "r.json"
        drv = Path(td) / "d.py"
        drv.write_text(DRIVER.format(obo=OBO, pd=str(pd_), gt=str(gt), ia=IA, o=str(raw)))
        r = subprocess.run([PY, str(drv)], capture_output=True, text=True, timeout=5400)
        if r.returncode != 0:
            print(f"  cafaeval FAILED {name}: {r.stderr[-300:]}", flush=True)
            return None
        best = None
        for rec in json.loads(raw.read_text()).get("f_micro_w", []):
            if (rec.get("ns") or "") == "biological_process" and rec.get("f_micro_w") is not None:
                best = rec
        out = {k: round(float(best[k]), 4) for k in ("f_micro_w", "pr_micro_w", "rc_micro_w", "tau", "cov_max")} if best else None
        if out:
            out["rows"] = n
        return out


t0 = time.time()
allrows = np.ones(len(dp), bool)
res = {"note": "does the deployed reranker add over a plain KNN score on the rows the KNN "
               "can speak for? Lab #113 said it regresses on PK and recommended raw-KNN "
               "there; that was the VALIDATION frame and was never tested on TEST.",
       "knn_reachable_frac": round(float(knn_rows.mean()), 4),
       "positives_reachable_by_knn": "3,333 of 15,248 (21.9%)"}

res["R_all_deployed_anchor"] = score("Rall", allrows, rr_al)
print(f"[R_all ] deployed reranker, whole pool   {res['R_all_deployed_anchor']}  ({time.time()-t0:.0f}s)", flush=True)
res["R_knn_reranker_on_knn_rows"] = score("Rknn", knn_rows, rr_al[knn_rows])
print(f"[R_knn ] deployed reranker, KNN rows     {res['R_knn_reranker_on_knn_rows']}", flush=True)
res["K_dist_raw_knn"] = score("Kdist", knn_rows, 1.0 - dist[knn_rows] / 2.0)
print(f"[K_dist] raw KNN 1-distance/2            {res['K_dist_raw_knn']}", flush=True)
vv = vote[knn_rows]
res["K_vote_raw_knn"] = score("Kvote", knn_rows, vv / max(np.nanmax(vv), 1e-9))
print(f"[K_vote] raw KNN vote fraction           {res['K_vote_raw_knn']}", flush=True)

g = lambda k: (res.get(k) or {}).get("f_micro_w")
if g("R_knn_reranker_on_knn_rows") and g("K_dist_raw_knn"):
    res["reranker_minus_raw_knn_on_knn_rows"] = round(g("R_knn_reranker_on_knn_rows") - g("K_dist_raw_knn"), 4)
if g("R_all_deployed_anchor") and g("R_knn_reranker_on_knn_rows"):
    res["classifier_candidates_worth"] = round(g("R_all_deployed_anchor") - g("R_knn_reranker_on_knn_rows"), 4)
json.dump(res, open(W / "is_the_reranker_hurting_pk.json", "w"), indent=1)

print("\n=== Is our own reranker costing us PK-BPO? ===", flush=True)
print(f"  R_all  deployed, whole pool     = {g('R_all_deployed_anchor')}   (the anchor)", flush=True)
print(f"  R_knn  deployed, KNN rows only  = {g('R_knn_reranker_on_knn_rows')}", flush=True)
print(f"  K_dist raw KNN, same rows       = {g('K_dist_raw_knn')}", flush=True)
print(f"  K_vote raw votes, same rows     = {g('K_vote_raw_knn')}", flush=True)
if res.get("reranker_minus_raw_knn_on_knn_rows") is not None:
    print(f"  -> reranker adds {res['reranker_minus_raw_knn_on_knn_rows']:+.4f} over raw KNN on the rows KNN can score", flush=True)
    print("     NEGATIVE => lab #113 was right on the TEST frame: the reranker hurts here.", flush=True)
if res.get("classifier_candidates_worth") is not None:
    print(f"  -> the classifier-proposed candidates are worth {res['classifier_candidates_worth']:+.4f} once reranked", flush=True)
print("DONE", flush=True)
