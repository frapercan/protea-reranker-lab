"""The lever, tested on a REAL submission file instead of on our parquet.

THE FOUR ARMS LINE UP BY HOW MUCH THEY WITHHOLD, and that is the whole story of PK-BP:

  predictions_7401_reranked.tsv   0% zeros, all scores in [0.0001, 1.0]   PK-BP  0.117
  predictions_percutgraft.tsv     11.25% exactly zero                     PK-BP  0.140
  predictions_protea.tsv          unknown (file is gone), optimal tau 0.06  PK-BP  0.2181
  our recipe, prefilter tau_pre=0.4                                       PK-BP  0.22288

`predictions_7401_reranked.tsv` (`results/clean_227230/lafa_submission/`, 27 June, 464,785 rows
over 7,401 proteins) submits every candidate with a strictly positive score. Under `prop=fill`
that means **no cell is ever the exact zero `fill` overwrites**, so the arm inherits nothing and
pays full price for its own worst candidates. Its locally-recorded PK-BP is 0.117, which is
essentially the 0.1196 measured for our own unfiltered file through the vectorised parser. Two
different arms, same failure, same number.

WHAT THIS SCRIPT ASKS. Everything so far was measured on `eval_scores.parquet`, our own
intermediate. This takes a real submission, one that was actually built to be sent, and withholds
its low-scoring rows. If the mechanism is what we say it is, the curve must rise steeply from
0.117 and peak well above it, because every withheld cell hands its ancestor back to `fill`.

  A  as-built              submit everything            expect ~0.117, the recorded value
  B..N  withhold below q   drop the bottom q quantile   sweep

The sweep is over QUANTILES, not raw thresholds, because this file's scale (a probability-like
[0.0001, 1.0]) is not the parquet's margin scale, and a quantile means the same thing in both.

GATE, naming the quantity (discipline #10): the quantity is the peak of the sweep minus arm A.
If the peak does not clear A by more than the 0.0034 fold-to-fold sd measured in PKW.8, the
lever does not survive contact with a real submission and this line closes. Reproducing A near
its recorded 0.117 is a precondition: if A does not reproduce, nothing below it is trustworthy
and the run is void.

CAVEAT, stated up front: this file covers all three categories with one score column, and the
board evaluates each category against its own ground truth. Each category is scored separately
here, exactly as the board does. The recorded results next to the file (NK 0.309, LK 0.348,
PK 0.117) are the control to reproduce.
"""
import json, subprocess, tempfile, time, collections
from pathlib import Path
import numpy as np

OUT = Path("/home/frapercan/Thesis2/storage/regen_headline")
REL = Path("/home/frapercan/Thesis2/CAFA_forever/data/releases/Sep_2025_Mar_2026")
SUB = Path("/home/frapercan/Thesis2/repositories/protea-reranker-lab/results/clean_227230/"
           "lafa_submission/predictions_7401_reranked.tsv")
OBO = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
IA = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv"
TOI = str(REL / "groundtruth_terms_of_interest.txt")
PY = "/home/frapercan/Thesis2/repositories/PROTEA/.venv/bin/python"

RECORDED = {"NK": 0.309, "LK": 0.348, "PK": 0.117}   # the run sitting next to the file

P, G, V = [], [], []
with SUB.open() as fh:
    for line in fh:
        p = line.rstrip("\n").split("\t")
        if len(p) >= 3:
            P.append(p[0]); G.append(p[1]); V.append(float(p[2]))
P = np.array(P); G = np.array(G); V = np.array(V)
print(f"{len(V):,} rows | {len(set(P.tolist())):,} proteins | range [{V.min():.6f}, {V.max():.6f}] "
      f"| exact zeros {(V == 0).mean():.2%}", flush=True)

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


def score(cat, q):
    """Withhold the bottom q quantile of the submission, then score `cat` the board's way."""
    thr = np.quantile(V, q) if q > 0 else -np.inf
    keep = V > thr
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "pd"
        d.mkdir()
        with (d / "predictions_7401_reranked.tsv").open("w") as fh:
            for p_, g_, v in zip(P[keep], G[keep], V[keep]):
                fh.write(f"{p_}\t{g_}\t{v:.6f}\n")
        raw = Path(td) / "r.json"
        drv = Path(td) / "d.py"
        drv.write_text(DRIVER.format(obo=OBO, pd=str(d), gt=str(REL / f"groundtruth_{cat}.tsv"),
                                     ia=IA, toi=TOI, o=str(raw)))
        r = subprocess.run([PY, str(drv)], capture_output=True, text=True, timeout=7200)
        if r.returncode != 0:
            print(f"  FAILED {cat} q={q}: {r.stderr[-250:]}", flush=True)
            return None
        for rec in json.loads(raw.read_text()).get("f_micro_w", []):
            if (rec.get("ns") or "") == "biological_process" and rec.get("f_micro_w") is not None:
                best = rec
        return {"f": round(float(best["f_micro_w"]), 5), "pr": round(float(best["pr_micro_w"]), 4),
                "rc": round(float(best["rc_micro_w"]), 4), "tau": round(float(best["tau"]), 3),
                "cov_w": round(float(best["cov_w"]), 4), "rows": int(keep.sum())}


QS = [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
t0 = time.time()
res = {"file": str(SUB), "recorded_next_to_it": RECORDED,
       "why": "predictions_7401_reranked.tsv submits every candidate strictly positive, so under "
              "prop=fill no cell is ever the exact zero fill overwrites: the arm inherits nothing "
              "and pays full price for its own worst candidates. Its recorded PK-BP 0.117 is "
              "essentially the 0.1196 we measured for our own unfiltered file through the "
              "vectorised parser.",
       "gate": "peak - arm A must exceed 0.0034 (the fold-to-fold sd from PKW.8). Reproducing A "
               "near its recorded value is a precondition; if A does not reproduce the run is void.",
       "sweep": {}}
for cat in ("PK", "LK", "NK"):
    res["sweep"][cat] = {}
    for q in QS:
        r = score(cat, q)
        res["sweep"][cat][str(q)] = r
        if r:
            print(f"  {cat}-BP  withhold bottom {q:>4.0%}  f={r['f']:.5f}  pr={r['pr']:.4f} "
                  f"rc={r['rc']:.4f}  cov_w={r['cov_w']}  rows={r['rows']:,}  ({time.time()-t0:.0f}s)",
                  flush=True)
        json.dump(res, open(OUT / "withhold_on_real_submission.json", "w"), indent=1)

print(f"\n=== Does withholding lift a REAL submission? ({time.time()-t0:.0f}s) ===", flush=True)
for cat in ("PK", "LK", "NK"):
    s = {float(k): v["f"] for k, v in res["sweep"][cat].items() if v}
    if not s:
        continue
    a = s[0.0]
    bq = max(s, key=s.get)
    res["sweep"][cat]["_verdict"] = {
        "arm_A_as_built": a, "recorded": RECORDED[cat], "A_reproduces": abs(a - RECORDED[cat]) < 0.005,
        "best_quantile": bq, "peak": s[bq], "gain": round(s[bq] - a, 5),
        "clears_pkw8_sd": (s[bq] - a) > 0.0034}
    v = res["sweep"][cat]["_verdict"]
    print(f"  {cat}-BP: as-built {a:.5f} (recorded {RECORDED[cat]}, reproduces: {v['A_reproduces']})"
          f" -> peak {s[bq]:.5f} withholding the bottom {bq:.0%}  = {v['gain']:+.5f}", flush=True)
json.dump(res, open(OUT / "withhold_on_real_submission.json", "w"), indent=1)
print("DONE", flush=True)
