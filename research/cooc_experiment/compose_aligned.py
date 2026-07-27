"""PKW.10. Two levers won on PK-BP. Are they two levers, or one lever twice?

(Named to avoid `compose_levers.py`, which is 16 July's technique-lever experiment and whose
recorded 0.1255 baseline and 0.6077 ceiling are both retracted. See
`compose_levers_STALE_RECEIPT.md`.)

WHAT IS ESTABLISHED, both within-frame, both gated, both cross-fitted over ten disjoint protein
folds with the choice made on nine pooled and applied blind to the tenth:

  prefilter tau_pre=0.4 on the reranker    +0.0092,  sd 0.0034,  unanimous 10/10
  classifier alone on PK-BP                +0.02106, sd 0.00789, unanimous 10/10

WHY THEY MIGHT NOT ADD. Both act by withholding weak candidates, and `prop=fill` pays for
withholding by handing the cell back to its best descendant as a free ancestor. The prefilter
withholds the reranker's low margins; the classifier arm submits only its own positive logits
(34.1% of pk-bpo rows against the reranker's 18.8%). And 16 July already measured that dropping
the classifier-only rows entirely (43.4% of the pool, carrying **78% of the positives**) leaves
the reranker at 0.2121, i.e. essentially unchanged: it scores them so badly that losing almost
all the positives costs nothing. If what the prefilter buys is mostly those same rows ceasing to
be scored badly, then the two levers are one act described twice and the sum is a fiction.

THE ARMS. One variable at a time. Every arm carries the positivity guard, which is forced rather
than chosen: the board reads us through cafaeval's vectorised parser, which has no positivity
guard, so a submitted non-positive score is stored and blocks the inheritance (measured cost of
getting that wrong: 0.0933).

  A  reranker, guard only                       the incumbent; the anchor
  B  reranker + prefilter tau_pre=0.4           expect ~+0.0092 over A
  C  classifier alone, guard only               expect ~+0.021 over A
  D  classifier alone + a prefilter on the classifier logit, threshold cross-fitted per fold

THE GATE, written before the number, quantity named (discipline #10): **the quantity is (D - C),
held out, averaged over the ten folds.** If it does not exceed the fold-to-fold sd of that same
difference, and is not positive in at least 8 of 10 folds, the levers OVERLAP: the prefilter adds
nothing once the classifier is doing the scoring, and only the classifier arm ships. Report the
per-fold threshold selections and the sd either way.

THE PRECONDITION, which has already saved one run tonight: arm A's fold-mean must land within 0.02
of **0.20316**, the fold-mean this exact design produced in `strategy_gate_crossfit.py`. If the
anchor does not come back, the run is void and nothing below it is reportable, however much I like
it.

THE FRAME, binding: the lab obo declares `data-version: releases/2025-07-22`; the board's t0 is
September 2025 and lives on the host's cluster. All four arms share one obo and one IA, so the
reweighting is common to them and cancels in the comparison. **No number here is a board number,
and none of these deltas may be added to 0.2181 to advertise one.**
"""
import json, subprocess, tempfile, time, collections, statistics
from pathlib import Path
import numpy as np, pyarrow.parquet as pq

W = Path("/home/frapercan/Thesis2/storage/cooc_experiment")
DS = Path("/home/frapercan/Thesis2/repositories/protea-reranker-lab/datasets/protst-global-train227-test230")
REL = Path("/home/frapercan/Thesis2/CAFA_forever/data/releases/Sep_2025_Mar_2026")
OBO = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"   # 2025-07-22, not the board's
IA = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv"
TOI = str(REL / "groundtruth_terms_of_interest.txt")
PY = "/home/frapercan/Thesis2/repositories/PROTEA/.venv/bin/python"
SEED, K = 42, 10
ANCHOR_A = 0.21584   # ALIGNED fold-mean from strategy_gate_aligned.py (the dict gave 0.20316)

t = pq.read_table(W / "rerank_out" / "eval_scores.parquet")
Ps = np.asarray(t.column("protein_accession").to_pylist())
Gs = np.asarray(t.column("go_term_id").to_pylist())
RRcol = t.column("reranker_score").to_numpy(zero_copy_only=False).astype(np.float64)
f = pq.read_table(DS / "eval.parquet",
                  columns=["protein_accession", "go_term_id", "category", "classifier_score"])
P = np.asarray(f.column("protein_accession").to_pylist())
G = np.asarray(f.column("go_term_id").to_pylist())
C = np.asarray(f.column("category").to_pylist())
CS = f.column("classifier_score").to_numpy(zero_copy_only=False).astype(np.float64)
# ALIGNED join, asserted. The first version of this script used
# `dict(zip(zip(prot, go), score))`, which is LAST-WINS over the pool's 40,192 duplicated
# (protein,term) keys: it rewrote 8.3% of rows, suppressed the reranker arm by 0.013, and was
# the ENTIRE cause of the PKW.8/PKW.10 contradiction (resolve_contradiction.py: flags -0.00002,
# row set +0.00000, dict -0.00954). The two tables are the same export in the same order.
assert len(Ps) == len(P) and (Ps == P).all() and (Gs == G).all(), (
    "eval_scores and eval.parquet are not row-aligned; an aligned join is invalid here")
RR = RRcol
pk = C == "pk"
P, G, CS, RR = P[pk], G[pk], CS[pk], RR[pk]
print(f"pk rows {len(P):,} | reranker matched {np.isfinite(RR).mean():.1%} | "
      f"classifier positive {np.nansum(CS > 0)/len(CS):.1%} | reranker positive {np.nansum(RR > 0)/len(RR):.1%}",
      flush=True)

GT = collections.defaultdict(set)
with (REL / "groundtruth_PK.tsv").open() as fh:
    next(fh)
    for line in fh:
        p_, t_, a_ = line.rstrip("\n").split("\t")[:3]
        if a_ == "P":
            GT[p_].add(t_)
prots = sorted(GT)
rng = np.random.default_rng(SEED)
rng.shuffle(prots)
FOLDS = [set(prots[i::K]) for i in range(K)]
print(f"gt pk-BP proteins {len(prots):,} -> {K} folds", flush=True)

DRIVER = '''
import json, signal
from cafaeval.evaluation import cafa_eval
signal.signal(signal.SIGTERM, signal.SIG_DFL)
df,dfs=cafa_eval("{obo}","{pd}","{gt}",ia="{ia}",prop="fill",norm="cafa",no_orphans=True,
    toi_file="{toi}",max_terms=None,th_step=0.01,n_cpu=4,weighted_only=False)
out={{}}
for k,v in dfs.items(): out[k]=v.reset_index().to_dict(orient="records")
json.dump(out,open("{o}","w"),default=str)
'''


def run(vals, tau, prots_):
    keep = np.isin(P, list(prots_)) & np.isfinite(vals) & (vals > tau)
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "pd"
        d.mkdir()
        with (d / "p.tsv").open("w") as fh:
            for p_, g_, x in zip(P[keep], G[keep], vals[keep]):
                fh.write(f"{p_}\t{g_}\t{x:.6f}\n")
        gt = Path(td) / "gt.tsv"
        with gt.open("w") as fh:
            for p_ in prots_:
                for g_ in GT[p_]:
                    fh.write(f"{p_}\t{g_}\n")
        raw = Path(td) / "r.json"
        drv = Path(td) / "d.py"
        drv.write_text(DRIVER.format(obo=OBO, pd=str(d), gt=str(gt), ia=IA, toi=TOI, o=str(raw)))
        r = subprocess.run([PY, str(drv)], capture_output=True, text=True, timeout=7200)
        if r.returncode != 0:
            return None
        best = None
        for rec in json.loads(raw.read_text()).get("f_micro_w", []):
            if (rec.get("ns") or "") == "biological_process" and rec.get("f_micro_w") is not None:
                best = rec
        return round(float(best["f_micro_w"]), 5) if best else None


pos = CS[np.isfinite(CS) & (CS > 0)]
CGRID = [0.0] + [round(float(np.quantile(pos, q)), 4) for q in (0.2, 0.4, 0.6, 0.8)]
print(f"classifier prefilter grid (quantiles of its own positive logits): {CGRID}", flush=True)

t0 = time.time()
res = {"question": "do the prefilter and the classifier arm compose, or are they one lever twice?",
       "frame": "LAB frame (obo data-version 2025-07-22). All four arms share one obo+IA so the "
                "reweighting cancels. NO NUMBER HERE IS A BOARD NUMBER.",
       "gate": "the quantity is (D - C) held-out, averaged over ten folds. If it does not exceed "
               "the fold-to-fold sd of that difference AND is not positive in >=8/10, the levers "
               "OVERLAP and only C ships.",
       "precondition": f"arm A fold-mean must land within 0.004 of {ANCHOR_A} or the run is void. The old 0.02 was WIDER than the 0.021 effect it was protecting and passed a corrupted arm.",
       "classifier_prefilter_grid": CGRID, "folds": []}

for i in range(K):
    rest = set().union(*[FOLDS[j] for j in range(K) if j != i])
    held = FOLDS[i]
    a = run(RR, 0.0, held)
    b = run(RR, 0.4, held)
    c = run(CS, 0.0, held)
    sel = {tau: run(CS, tau, rest) for tau in CGRID}
    sel = {k: v for k, v in sel.items() if v is not None}
    dtau = max(sel, key=sel.get) if sel else 0.0
    d = run(CS, dtau, held)
    rec = {"fold": i, "A_reranker": a, "B_reranker_prefilter": b, "C_classifier": c,
           "D_classifier_prefilter": d, "D_chosen_tau": dtau,
           "sweep_on_the_other_nine": sel,
           "D_minus_C": round(d - c, 5) if (d is not None and c is not None) else None,
           "B_minus_A": round(b - a, 5) if (b is not None and a is not None) else None,
           "C_minus_A": round(c - a, 5) if (c is not None and a is not None) else None}
    res["folds"].append(rec)
    print(f"  fold {i}: A={a} B={b} C={c} D={d} (tau={dtau}) | B-A={rec['B_minus_A']} "
          f"C-A={rec['C_minus_A']} **D-C={rec['D_minus_C']}**  ({time.time()-t0:.0f}s)", flush=True)
    json.dump(res, open(W / "compose_aligned.json", "w"), indent=1)


def agg(key):
    v = [x[key] for x in res["folds"] if x.get(key) is not None]
    if not v:
        return (None, None, 0, 0)
    return (round(statistics.mean(v), 5), round(statistics.stdev(v), 5) if len(v) > 1 else None,
            sum(1 for x in v if x > 0), len(v))


A_vals = [x["A_reranker"] for x in res["folds"] if x["A_reranker"] is not None]
A_mean = statistics.mean(A_vals) if A_vals else None
anchor_ok = A_mean is not None and abs(A_mean - ANCHOR_A) < 0.004
dc_m, dc_sd, dc_pos, dc_n = agg("D_minus_C")
ba_m, ba_sd, ba_pos, _ = agg("B_minus_A")
ca_m, ca_sd, ca_pos, _ = agg("C_minus_A")
res["verdict"] = {
    "A_fold_mean": round(A_mean, 5) if A_mean else None, "anchor": ANCHOR_A,
    "PRECONDITION_holds": anchor_ok,
    "B_minus_A_prefilter_on_reranker": {"mean": ba_m, "sd": ba_sd, "positive": f"{ba_pos}/{dc_n}"},
    "C_minus_A_classifier_alone": {"mean": ca_m, "sd": ca_sd, "positive": f"{ca_pos}/{dc_n}"},
    "D_minus_C_prefilter_on_classifier": {"mean": dc_m, "sd": dc_sd, "positive": f"{dc_pos}/{dc_n}"},
    "D_tau_selections": {str(k): v for k, v in collections.Counter(x["D_chosen_tau"] for x in res["folds"]).items()},
    "LEVERS_COMPOSE": bool(anchor_ok and dc_sd is not None and dc_m is not None
                           and dc_m > dc_sd and dc_pos >= 8),
}
json.dump(res, open(W / "compose_aligned.json", "w"), indent=1)
v = res["verdict"]
print(f"\n=== PKW.10: two levers, or one lever twice? ({time.time()-t0:.0f}s) ===", flush=True)
print(f"  PRECONDITION: arm A fold-mean {v['A_fold_mean']} vs anchor {ANCHOR_A} -> {anchor_ok}", flush=True)
print(f"  B-A  prefilter on the reranker         mean {ba_m} sd {ba_sd} positive {ba_pos}/{dc_n}", flush=True)
print(f"  C-A  classifier alone                  mean {ca_m} sd {ca_sd} positive {ca_pos}/{dc_n}", flush=True)
print(f"  D-C  prefilter ON TOP of the classifier mean {dc_m} sd {dc_sd} positive {dc_pos}/{dc_n}", flush=True)
print(f"  D's threshold selections: {v['D_tau_selections']}", flush=True)
print(f"  -> LEVERS COMPOSE: {v['LEVERS_COMPOSE']}", flush=True)
print("  FALSE => they overlap: the prefilter adds nothing once the classifier scores; only C ships.", flush=True)
print("DONE", flush=True)
