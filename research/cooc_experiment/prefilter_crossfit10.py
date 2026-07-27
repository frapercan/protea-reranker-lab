"""PKW.8b. The 2-fold cross-fit said the prefilter transfers. This asks how much of that is noise.

WHY THIS EXISTS, PLAINLY. In `prefilter_crossfit.py` I wrote the gate before the number, as the
discipline demands, and I wrote it WRONG: I defined the "selection spread" as max-min of f over
the whole threshold grid. That grid deliberately includes thresholds that destroy information
(tau_pre=1.5 submits 177 rows and scores 0.026), so max-min measures the SHAPE OF THE CURVE, not
the noise in selecting a point on it. A gate no result could ever pass is not a gate. It is
reported there as False and it means nothing. This is the estimate it should have been.

WHAT THE 2-FOLD RUN ESTABLISHED (real, and kept):
  - tau_pre in [-2.0, 0.0] gives f IDENTICAL to 4dp on both folds while rows fall 185,070 ->
    57,784. Submitting a negative score and withholding it are the same act, because the parser
    discards non-positives (parser.py:230). That is the mechanism observed from outside.
  - Both folds independently chose tau_pre = 0.4, neither seeing the other.
  - Held-out gains over the accidental tau_pre=0: +0.0109 (A->B) and +0.0086 (B->A).

THE DESIGN. 10 disjoint protein folds. For each fold i: sweep the grid on the OTHER NINE pooled,
take the argmax, apply it BLIND to fold i, and compare against tau_pre=0 on that same fold i.
The selection never sees the fold it is scored on. Board flags exactly: prop=fill, norm=cafa,
no_orphans, max_terms=500, th_step=0.001.

THE GATE, and this one is a gate. Two conditions, both required:
  1. STABILITY: the ten independent selections must agree. If they scatter across the grid, the
     argmax is fitting noise and there is no threshold to ship.
  2. SIGN: the held-out delta must be positive in the clear majority of folds, and its mean must
     exceed one standard deviation across folds. mean <= sd means the gain is inside the noise.
Fail either and this line closes and I say so.

CAVEAT CARRIED FORWARD, unchanged: this measures generalisation across PROTEINS, not across TIME.
Validation-then-blind over the snapshot windows is impossible on frozen data (no full gt exists
for v225-v227, and the deployed booster early-stopped on it). A threshold that transfers between
protein folds may still drift between windows. This cannot see that, and does not claim to.
"""
import json, subprocess, tempfile, time, collections, statistics
from pathlib import Path
import numpy as np, pyarrow.parquet as pq

W = Path("/home/frapercan/Thesis2/storage/cooc_experiment")
DS = Path("/home/frapercan/Thesis2/repositories/protea-reranker-lab/datasets/protst-global-train227-test230")
OBO = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
IA = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv"
PY = "/home/frapercan/Thesis2/repositories/PROTEA/.venv/bin/python"
SEED, K = 42, 10

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
FOLDS = [set(prots[i::K]) for i in range(K)]
print(f"{len(RR):,} pk-bpo rows | {len(prots):,} gt proteins -> {K} folds of ~{len(FOLDS[0])}", flush=True)

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


def score(pset, tau_pre):
    keep = (RR > tau_pre) & np.isin(PROT, list(pset))
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "pd"
        d.mkdir()
        with (d / "p.tsv").open("w") as fh:
            for p_, g_, v in zip(PROT[keep], GO[keep], RR[keep]):
                fh.write(f"{p_}\t{g_}\t{v:.6f}\n")
        gt = Path(td) / "gt.tsv"
        with gt.open("w") as fh:
            for p_ in pset:
                for g_ in GT[p_]:
                    fh.write(f"{p_}\t{g_}\n")
        raw = Path(td) / "r.json"
        drv = Path(td) / "d.py"
        drv.write_text(DRIVER.format(obo=OBO, pd=str(d), gt=str(gt), ia=IA, o=str(raw)))
        r = subprocess.run([PY, str(drv)], capture_output=True, text=True, timeout=3600)
        if r.returncode != 0:
            return None
        best = None
        for rec in json.loads(raw.read_text()).get("f_micro_w", []):
            if (rec.get("ns") or "") == "biological_process" and rec.get("f_micro_w") is not None:
                best = rec
        return round(float(best["f_micro_w"]), 4) if best else None


GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]
t0 = time.time()
res = {"design": "10-fold cross-fit over test proteins: select tau_pre on nine folds pooled, "
                 "apply blind to the tenth. Measures generalisation across PROTEINS, not TIME.",
       "gate": "(1) the ten selections must agree; (2) the held-out delta must be positive in a "
               "clear majority and its mean must exceed one sd across folds.",
       "prior_gate_was_malformed": "prefilter_crossfit.py defined the spread as max-min over a grid "
                                   "containing deliberately destructive thresholds, so it measured "
                                   "the curve's shape, not selection noise. No result could pass it.",
       "board_flags": "prop=fill norm=cafa no_orphans max_terms=500 th_step=0.001",
       "grid": GRID, "K": K, "seed": SEED, "folds": []}

for i in range(K):
    rest = set().union(*[FOLDS[j] for j in range(K) if j != i])
    sel = {tp: score(rest, tp) for tp in GRID}
    sel = {k: v for k, v in sel.items() if v is not None}
    chosen = max(sel, key=sel.get)
    held = score(FOLDS[i], chosen)
    base = score(FOLDS[i], 0.0)
    rec = {"fold": i, "chosen_tau_pre": chosen, "sweep_on_the_other_nine": sel,
           "held_out_f": held, "held_out_f_at_tau0": base,
           "held_out_delta": round(held - base, 4) if (held and base) else None}
    res["folds"].append(rec)
    print(f"  fold {i}: chose tau_pre={chosen} -> held-out {held} vs {base} = {rec['held_out_delta']:+.4f}  ({time.time()-t0:.0f}s)", flush=True)
    json.dump(res, open(W / "prefilter_crossfit10.json", "w"), indent=1)

ch = [f["chosen_tau_pre"] for f in res["folds"]]
d = [f["held_out_delta"] for f in res["folds"] if f["held_out_delta"] is not None]
agree = collections.Counter(ch)
mean, sd = round(statistics.mean(d), 4), round(statistics.stdev(d), 4)
res["verdict"] = {
    "selections": dict(agree), "unanimous": len(agree) == 1,
    "n_positive": sum(1 for x in d if x > 0), "n_folds": len(d),
    "mean_held_out_delta": mean, "sd_across_folds": sd,
    "mean_exceeds_sd": mean > sd,
    "PASSES_GATE": len(agree) == 1 and sum(1 for x in d if x > 0) >= 8 and mean > sd,
}
json.dump(res, open(W / "prefilter_crossfit10.json", "w"), indent=1)
print(f"\n=== PKW.8b verdict ({time.time()-t0:.0f}s) ===", flush=True)
print(f"  selections across the ten folds: {dict(agree)}  (unanimous: {len(agree)==1})", flush=True)
print(f"  held-out delta: mean {mean:+.4f}  sd {sd:.4f}  positive in {sum(1 for x in d if x>0)}/{len(d)} folds", flush=True)
print(f"  mean exceeds sd: {mean > sd}", flush=True)
print(f"  GATE: {'PASS' if res['verdict']['PASSES_GATE'] else 'FAIL, and this line closes'}", flush=True)
print("DONE", flush=True)
