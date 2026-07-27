"""PKW.8. The prefilter threshold is a free parameter nobody chose. Is it worth anything?

THE MECHANISM (instrument audit, kill-tested). `prop="fill"` only overwrites cells that are
EXACTLY ZERO (graph.py:320-323), and the parser discards non-positive scores (parser.py:230,
`if prob_f > old` from old=0.0). So a candidate you do NOT submit becomes a zero cell that
`fill` overwrites with its best descendant's score: a free ancestor, and they arrive at
weighted precision 0.2967 against the arm's own 0.1801. A candidate you DO submit with a low
score is stored, and `fill` PROTECTS it, blocking that inheritance.

**Submitting a bad candidate is worse than not submitting it: it robs a good ancestor.**

Our 0.2131 comes from lambdarank's margins running [-8.48, 3.81] and the clamp silently
dropping the 80.9% that are <= 0. Nobody chose that threshold. The audit measured +0.5 ->
0.2226 (+0.0095) and +1.0 -> 0.1216 (destroys real information), so there is an optimum.
This looks for it, honestly.

WHY NOT "SELECT ON VALIDATION, APPLY BLIND TO TEST", which is the discipline this loop
demands. Two blockers, both checked rather than assumed:
  1. There is no FULL ground truth for the validation window. gt_pk_bp.tsv is the test gt
     and carries 41,727 pairs against only 15,248 label=1 pool rows, i.e. it includes terms
     OUTSIDE the pool. A validation gt rebuilt from label=1 rows would be pool-restricted,
     which inflates recall and biases the very threshold we are choosing.
  2. The deployed booster EARLY-STOPPED on v225-v227 (`train_allfeat.py: VAL_PAIR`), so its
     scores there are optimistic.
So the clean design available is CROSS-FITTING over test proteins: the booster never saw any
of them, and the real gt applies. Its limitation, stated up front: it measures generalisation
across PROTEINS, not across TIME. A threshold that transfers between protein folds may still
drift between snapshot windows, and this cannot see that.

DESIGN. Split the 4,402 gt proteins into 2 folds by protein. On fold A sweep the prefilter
over a grid and pick the best; apply that choice BLIND to fold B. Then swap. Compare against
tau_pre = 0, the accidental default that produced the published anchor, on the same folds.
Board flags exactly: prop=fill, norm=cafa, no_orphans, max_terms=500, th_step=0.001.

THE GATE, written before the number: the held-out gain must exceed the SELECTION SPREAD (how
much f wanders across thresholds on the selecting fold). If it does not, the threshold is
noise and this line closes. Report N and the spread either way.
"""
import json, subprocess, tempfile, time, collections
from pathlib import Path
import numpy as np, pyarrow.parquet as pq

W = Path("/home/frapercan/Thesis2/storage/cooc_experiment")
DS = Path("/home/frapercan/Thesis2/repositories/protea-reranker-lab/datasets/protst-global-train227-test230")
OBO = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
IA = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv"
PY = "/home/frapercan/Thesis2/repositories/PROTEA/.venv/bin/python"
SEED = 42

t = pq.read_table(W / "rerank_out" / "eval_scores.parquet")
cat = np.asarray(t.column("category").to_pylist())
asp = np.asarray(t.column("aspect").to_pylist())
m = (cat == "pk") & (asp == "bpo")
PROT = np.asarray(t.column("protein_accession").to_pylist())[m]
GO = np.asarray(t.column("go_term_id").to_pylist())[m]
RR = t.column("reranker_score").to_numpy(zero_copy_only=False).astype(np.float64)[m]

GT = collections.defaultdict(set)
for line in open(DS / "gt_pk_bp.tsv"):
    p, g = line.rstrip("\n").split("\t")
    GT[p].add(g)

prots = sorted(GT)
rng = np.random.default_rng(SEED)
rng.shuffle(prots)
half = len(prots) // 2
FOLD = {"A": set(prots[:half]), "B": set(prots[half:])}
print(f"{len(RR):,} pk-bpo rows | gt proteins {len(prots):,} -> fold A {len(FOLD['A']):,} / B {len(FOLD['B']):,}", flush=True)
print(f"score range [{RR.min():.4f}, {RR.max():.4f}]; {(RR <= 0).mean():.1%} are <= 0 and the clamp drops them", flush=True)

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


def score(fold, tau_pre):
    """Score ONE fold's proteins with rows below tau_pre withheld from the submission."""
    keep = (RR > tau_pre) & np.isin(PROT, list(FOLD[fold]))
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "pd"
        d.mkdir()
        n = 0
        with (d / "p.tsv").open("w") as fh:
            for p_, g_, v in zip(PROT[keep], GO[keep], RR[keep]):
                fh.write(f"{p_}\t{g_}\t{v:.6f}\n")
                n += 1
        gt = Path(td) / "gt.tsv"
        with gt.open("w") as fh:
            for p_ in FOLD[fold]:
                for g_ in GT[p_]:
                    fh.write(f"{p_}\t{g_}\n")
        raw = Path(td) / "r.json"
        drv = Path(td) / "d.py"
        drv.write_text(DRIVER.format(obo=OBO, pd=str(d), gt=str(gt), ia=IA, o=str(raw)))
        r = subprocess.run([PY, str(drv)], capture_output=True, text=True, timeout=3600)
        if r.returncode != 0:
            print(f"  cafaeval FAILED fold={fold} tau_pre={tau_pre}: {r.stderr[-200:]}", flush=True)
            return None
        best = None
        for rec in json.loads(raw.read_text()).get("f_micro_w", []):
            if (rec.get("ns") or "") == "biological_process" and rec.get("f_micro_w") is not None:
                best = rec
        if not best:
            return None
        return {"f": round(float(best["f_micro_w"]), 4), "pr": round(float(best["pr_micro_w"]), 4),
                "rc": round(float(best["rc_micro_w"]), 4), "rows_submitted": n}


# the grid. 0.0 is the accidental default that produced the published anchor.
GRID = [-2.0, -1.0, -0.5, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5]
t0 = time.time()
res = {"mechanism": "prop=fill overwrites only EXACTLY-zero cells; the parser drops non-positive "
                    "scores; so an unsubmitted candidate inherits its best descendant's score for "
                    "free while a submitted low-scoring one blocks it.",
       "design": "cross-fit over test proteins. Validation-then-blind is impossible here: no FULL "
                 "gt exists for v225-v227 and the deployed booster early-stopped on it. This "
                 "measures generalisation across PROTEINS, not across TIME.",
       "board_flags": "prop=fill norm=cafa no_orphans max_terms=500 th_step=0.001",
       "N_thresholds": len(GRID), "grid": GRID, "seed": SEED, "sweep": {}}

for fold in ("A", "B"):
    res["sweep"][fold] = {}
    for tp in GRID:
        r = score(fold, tp)
        res["sweep"][fold][str(tp)] = r
        if r:
            print(f"  fold {fold}  tau_pre={tp:>5}  f={r['f']:.4f}  pr={r['pr']:.4f} rc={r['rc']:.4f}  rows={r['rows_submitted']:,}", flush=True)
    json.dump(res, open(W / "prefilter_crossfit.json", "w"), indent=1)

# --- the cross-fitted verdict -------------------------------------------------
out = {}
for sel, app in (("A", "B"), ("B", "A")):
    vals = {tp: res["sweep"][sel][str(tp)]["f"] for tp in GRID if res["sweep"][sel].get(str(tp))}
    best_tp = max(vals, key=vals.get)
    spread = round(max(vals.values()) - min(vals.values()), 4)
    held = res["sweep"][app][str(best_tp)]["f"]
    base = res["sweep"][app]["0.0"]["f"]
    out[f"select_on_{sel}_apply_to_{app}"] = {
        "chosen_tau_pre": best_tp, "f_on_selecting_fold": vals[best_tp],
        "selection_spread_on_selecting_fold": spread,
        "held_out_f": held, "held_out_f_at_tau0": base,
        "held_out_gain": round(held - base, 4),
        "gain_exceeds_spread": (held - base) > spread,
    }
res["crossfit"] = out
json.dump(res, open(W / "prefilter_crossfit.json", "w"), indent=1)

print(f"\n=== PKW.8: is the prefilter threshold worth anything? ({time.time()-t0:.0f}s) ===", flush=True)
for k, v in out.items():
    print(f"  {k}: chose tau_pre={v['chosen_tau_pre']} -> held-out {v['held_out_f']} vs "
          f"{v['held_out_f_at_tau0']} at the accidental 0.0  =  {v['held_out_gain']:+.4f}", flush=True)
    print(f"    selection spread on the selecting fold = {v['selection_spread_on_selecting_fold']}  "
          f"-> gain exceeds spread: {v['gain_exceeds_spread']}", flush=True)
print(f"  N thresholds tried: {len(GRID)}", flush=True)
print("  GATE: if the gain does not exceed the spread it is noise and this line closes.", flush=True)
print("DONE", flush=True)
