"""The one-variable ladder from PKW.8's +0.0092 to PKW.10's -0.0003. One of them is wrong.

THE CONTRADICTION. The same lever (prefilter the reranker at tau_pre=0.4), the same ten folds, the
same seed, measured twice by me:

  PKW.8  `prefilter_crossfit10.py`         +0.0092, sd 0.0034, unanimous 10/10
  PKW.10 `compose_prefilter_classifier.py` -0.00032, sd 0.00581, 7/10   (arm B - arm A)

FOUR THINGS DIFFER AND NONE WAS ISOLATED. This walks from one config to the other, changing
exactly one at each rung, so the step that moves the number is the cause.

  L0  bpo-only rows, aligned join, LAB line (max_terms=500, th_step=0.001, no toi)   = PKW.8
  L1  + the BOARD's line (max_terms=None => VECTORISED parser, th_step=0.01, -toi)
  L2  + all-aspect pk rows instead of bpo-only
  L3  + the DICT join                                                                = PKW.10

THE SUSPECT I ALREADY FOUND, and it is mine. PKW.10 built `RRmap = dict(zip(zip(prot, go), score))`.
The pk-bpo pool holds **616,223 rows over 576,031 distinct (protein, term) keys: 40,192 are
duplicated**, once from the KNN generator and once from the classifier. A dict keyed on
(protein, term) is **last-wins**, so it silently drops 40,192 rows. They are not copies: the two
scores of a duplicated pair differ by a **median of 3.60** and up to 9.00, and for **9,510 pairs the
two straddle zero**, i.e. the dict decides by row order whether that candidate is submitted at all.
PKW.8 read the aligned arrays and every row kept its own score. **This is the exact bug catalogued
last night in `is_the_reranker_hurting_pk.py`, and I wrote it again.**

If L3 is the rung that moves, PKW.8 stands and PKW.10's arms A and B are void. Arms C and D are
untouched by this: `classifier_score` was read aligned from `eval.parquet` and never went through
the dict, so **D - C = +0.00759 survives either way.**

MY PRECONDITION FAILED TO CATCH IT, and that is its own lesson. It required arm A's fold-mean to
land within **0.02** of the anchor while the effect under measurement was **0.021**. A tolerance
wider than the effect tests nothing. It passed a corrupted arm and told me the design was sound.

THE GATE, quantity named: the quantity is the prefilter delta (tau_pre 0.4 minus tau_pre 0) at each
rung, fold-mean over ten folds. **The rung where it crosses from +0.009 to ~0 is the cause.** If no
single rung explains it, they interact and I say so rather than picking one.
"""
import json, subprocess, tempfile, time, collections, statistics
from pathlib import Path
import numpy as np, pyarrow.parquet as pq

W = Path("/home/frapercan/Thesis2/storage/cooc_experiment")
DS = Path("/home/frapercan/Thesis2/repositories/protea-reranker-lab/datasets/protst-global-train227-test230")
REL = Path("/home/frapercan/Thesis2/CAFA_forever/data/releases/Sep_2025_Mar_2026")
OBO = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
IA = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv"
TOI = str(REL / "groundtruth_terms_of_interest.txt")
PY = "/home/frapercan/Thesis2/repositories/PROTEA/.venv/bin/python"
SEED, K = 42, 10

t = pq.read_table(W / "rerank_out" / "eval_scores.parquet")
P0 = np.asarray(t.column("protein_accession").to_pylist())
G0 = np.asarray(t.column("go_term_id").to_pylist())
C0 = np.asarray(t.column("category").to_pylist())
A0 = np.asarray(t.column("aspect").to_pylist())
R0 = t.column("reranker_score").to_numpy(zero_copy_only=False).astype(np.float64)

pk_bpo = (C0 == "pk") & (A0 == "bpo")
pk_all = (C0 == "pk")
print(f"eval_scores: pk-bpo {pk_bpo.sum():,} rows, pk-all {pk_all.sum():,} rows", flush=True)

# the dict join, rebuilt exactly as PKW.10 built it, so L3 reproduces the bug rather than describing it
RRmap = dict(zip(zip(P0, G0), R0))
R_dict = np.array([RRmap[(p_, g_)] for p_, g_ in zip(P0, G0)])
changed = (R_dict != R0).sum()
print(f"the dict join changes {changed:,} of {len(R0):,} rows ({changed/len(R0):.1%}) "
      f"-> last-wins on every duplicated (protein,term)", flush=True)

GTBP = collections.defaultdict(set)
with (REL / "groundtruth_PK.tsv").open() as fh:
    next(fh)
    for line in fh:
        p_, t_, a_ = line.rstrip("\n").split("\t")[:3]
        if a_ == "P":
            GTBP[p_].add(t_)
prots = sorted(GTBP)
rng = np.random.default_rng(SEED)
rng.shuffle(prots)
FOLDS = [set(prots[i::K]) for i in range(K)]
print(f"gt pk-BP proteins {len(prots):,} -> {K} folds (seed {SEED}, identical to PKW.8 and PKW.10)", flush=True)

DRIVER = '''
import json, signal
from cafaeval.evaluation import cafa_eval
signal.signal(signal.SIGTERM, signal.SIG_DFL)
df,dfs=cafa_eval("{obo}","{pd}","{gt}",ia="{ia}",prop="fill",norm="cafa",no_orphans=True,
    toi_file={toi},max_terms={mt},th_step={ts},n_cpu=4,weighted_only=False)
out={{}}
for k,v in dfs.items(): out[k]=v.reset_index().to_dict(orient="records")
json.dump(out,open("{o}","w"),default=str)
'''

RUNGS = {
    "L0_PKW8_bpo_aligned_labline":   dict(rows="bpo", join="aligned", toi=None, mt=500,  ts=0.001),
    "L1_plus_board_line":            dict(rows="bpo", join="aligned", toi=TOI,  mt=None, ts=0.01),
    "L2_plus_all_aspect_rows":       dict(rows="all", join="aligned", toi=TOI,  mt=None, ts=0.01),
    "L3_plus_dict_join_PKW10":       dict(rows="all", join="dict",    toi=TOI,  mt=None, ts=0.01),
}


def score(cfg, tau, prots_):
    rowmask = pk_bpo if cfg["rows"] == "bpo" else pk_all
    vals = R0 if cfg["join"] == "aligned" else R_dict
    keep = rowmask & np.isin(P0, list(prots_)) & (vals > tau)
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "pd"
        d.mkdir()
        with (d / "p.tsv").open("w") as fh:
            for p_, g_, x in zip(P0[keep], G0[keep], vals[keep]):
                fh.write(f"{p_}\t{g_}\t{x:.6f}\n")
        gt = Path(td) / "gt.tsv"
        with gt.open("w") as fh:
            for p_ in prots_:
                for g_ in GTBP[p_]:
                    fh.write(f"{p_}\t{g_}\n")
        raw = Path(td) / "r.json"
        drv = Path(td) / "d.py"
        drv.write_text(DRIVER.format(obo=OBO, pd=str(d), gt=str(gt), ia=IA, o=str(raw),
                                     toi=repr(cfg["toi"]), mt=cfg["mt"], ts=cfg["ts"]))
        r = subprocess.run([PY, str(drv)], capture_output=True, text=True, timeout=7200)
        if r.returncode != 0:
            return None
        best = None
        for rec in json.loads(raw.read_text()).get("f_micro_w", []):
            if (rec.get("ns") or "") == "biological_process" and rec.get("f_micro_w") is not None:
                best = rec
        return round(float(best["f_micro_w"]), 5) if best else None


t0 = time.time()
res = {"contradiction": "PKW.8 measured the reranker prefilter at +0.0092 unanimous 10/10; PKW.10 "
                        "measured the same lever at -0.00032, 7/10. Same folds, same seed.",
       "the_dict_bug": {"pk_bpo_rows": int(pk_bpo.sum()), "distinct_keys": 576031,
                        "duplicated_keys": 40192,
                        "median_within_duplicate_score_spread": 3.5972,
                        "duplicates_straddling_zero": 9510,
                        "rows_changed_by_the_dict": int(changed)},
       "gate": "the quantity is the prefilter delta (tau 0.4 minus tau 0) fold-mean over ten folds "
               "at each rung. The rung where it collapses is the cause.",
       "rungs": {}}

for name, cfg in RUNGS.items():
    deltas = []
    for i in range(K):
        a = score(cfg, 0.0, FOLDS[i])
        b = score(cfg, 0.4, FOLDS[i])
        if a is not None and b is not None:
            deltas.append(round(b - a, 5))
    m = round(statistics.mean(deltas), 5) if deltas else None
    sd = round(statistics.stdev(deltas), 5) if len(deltas) > 1 else None
    res["rungs"][name] = {"config": {k: str(v) for k, v in cfg.items()},
                          "prefilter_delta_fold_mean": m, "sd": sd,
                          "positive": f"{sum(1 for x in deltas if x > 0)}/{len(deltas)}",
                          "per_fold": deltas}
    print(f"  {name:32s} delta {m} sd {sd} positive {sum(1 for x in deltas if x>0)}/{len(deltas)}"
          f"  ({time.time()-t0:.0f}s)", flush=True)
    json.dump(res, open(W / "resolve_contradiction.json", "w"), indent=1)

print(f"\n=== which rung moves the number? ({time.time()-t0:.0f}s) ===", flush=True)
prev = None
for name in RUNGS:
    m = res["rungs"][name]["prefilter_delta_fold_mean"]
    step = f"  (step {m - prev:+.5f})" if prev is not None and m is not None else ""
    print(f"  {name:32s} {m}{step}", flush=True)
    prev = m
print("\n  L0 is PKW.8's config; L3 is PKW.10's. The rung with the big step is the cause.", flush=True)
print("  If L3 is the step: PKW.8 stands and PKW.10's arms A/B are void (the dict bug).", flush=True)
print("  Arms C/D never used the dict, so D-C = +0.00759 survives either way.", flush=True)
print("DONE", flush=True)
