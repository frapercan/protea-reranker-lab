"""The submission producer, and the first script that scores us the way the board does.

WHY THIS EXISTS. Our headline PK-BP figure, 0.2181, comes from `predictions_protea.tsv`, a file
that exists nowhere: no repo has ever tracked it, `predictions_uneval` and
`predictions_by_window` are both absent from disk, and the only match for the name in the whole
tree is `apps/web/lib/book.ts`, the surface that publishes the number. A number on a surface
with no script behind it is the one thing this campaign refuses to ship. This is the script.

IT MIRRORS THE BOARD RATHER THAN THE LAB. The board runs one evaluation per knowledge category
(`results_NK`, `results_LK`, `results_PK`), each over a predictions directory, against that
category's own ground truth, with this line (`CAFA_forever/modules/local/evaluation.nf:211-215`,
identical at 243/275/308/340/372):

    cafaeval <obo> <predDir> <gt> -ia <IA> -out_dir <..> -toi <toi> \
      -prop fill -norm cafa -threads N -no_orphans

**No `-max_terms`. No `-th_step`.** So the defaults hold: `max_terms=None`, `th_step=0.01`. Every
earlier lab run passed `max_terms=500, th_step=0.001`, and that was never the board's line.

THE POSITIVITY GUARD IS NOT COSMETIC. `max_terms` selects a parser, not a cap:
`_pred_parser_legacy` runs only when a cap is set, otherwise `_pred_parser_vectorised` does. The
legacy path stores a value only when `prob_f > old` from `old = 0.0` (`parser.py:230`), so
non-positive scores are dropped and their cells stay the exact zero that `prop=fill` overwrites
with a free ancestor. The vectorised path has no positivity guard, so `fill` protects those cells
and the inheritance is blocked. Measured on our pk-bpo file: legacy 0.21293, vectorised 0.11960.
**0.0933, purely the parser.** Since the board reads us through the vectorised path, any
submission carrying raw lambdarank margins loses about 0.09 with nothing to announce it. The
guard here makes that impossible by construction rather than by luck.

THE ARMS. One variable, and the baseline is the deployed behaviour:

  guard_only   submit score > 0        reproduces the deployed recipe on the board's own line
  prefilter    submit score > 0.4      PKW.8's threshold: unanimous across 10 folds, +0.0092
                                       held-out, selected without ever seeing the fold it scored

PKW.8's threshold was chosen on the legacy path, and that is legitimate here: above zero the two
paths agree exactly, because legacy discards non-positives anyway. The selection never touched
the region where they differ.

WHAT THIS CANNOT DO. It cannot recover `predictions_protea.tsv`, and it does not claim to. If
`guard_only` lands near 0.2181 that is evidence the deployed submission was the deployed recipe
with a positivity-preserving transform; if it lands elsewhere, the published headline came from
an arm we do not have, and that must be said plainly rather than papered over.
"""
import json, subprocess, tempfile, time, collections
from pathlib import Path
import numpy as np, pyarrow.parquet as pq

W = Path("/home/frapercan/Thesis2/storage/cooc_experiment")
OUT = Path("/home/frapercan/Thesis2/storage/regen_headline")
REL = Path("/home/frapercan/Thesis2/CAFA_forever/data/releases/Sep_2025_Mar_2026")
OBO = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
IA = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv"
TOI = str(REL / "groundtruth_terms_of_interest.txt")
PY = "/home/frapercan/Thesis2/repositories/PROTEA/.venv/bin/python"

# what the board published for us, to compare against rather than to assume
BOARD = {"NK": 0.3374, "LK": 0.4402, "PK": 0.21806691440311407}

t = pq.read_table(W / "rerank_out" / "eval_scores.parquet")
CAT = np.asarray(t.column("category").to_pylist())
ASP = np.asarray(t.column("aspect").to_pylist())
PROT = np.asarray(t.column("protein_accession").to_pylist())
GO = np.asarray(t.column("go_term_id").to_pylist())
RR = t.column("reranker_score").to_numpy(zero_copy_only=False).astype(np.float64)
print(f"{len(RR):,} scored rows | categories {sorted(set(CAT.tolist()))} | "
      f"{(RR <= 0).mean():.1%} non-positive overall", flush=True)

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


def score(cat, tau_pre):
    """One board-shaped evaluation: our submission for `cat`, against that cell's own gt."""
    keep = (CAT == cat.lower()) & (RR > tau_pre)
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "pd"
        d.mkdir()
        with (d / "predictions_protea.tsv").open("w") as fh:
            for p_, g_, v in zip(PROT[keep], GO[keep], RR[keep]):
                fh.write(f"{p_}\t{g_}\t{v:.6f}\n")
        raw = Path(td) / "r.json"
        drv = Path(td) / "d.py"
        drv.write_text(DRIVER.format(obo=OBO, pd=str(d), gt=str(REL / f"groundtruth_{cat}.tsv"),
                                     ia=IA, toi=TOI, o=str(raw)))
        r = subprocess.run([PY, str(drv)], capture_output=True, text=True, timeout=7200)
        if r.returncode != 0:
            print(f"  FAILED {cat} tau_pre={tau_pre}: {r.stderr[-300:]}", flush=True)
            return None
        rows = {}
        for rec in json.loads(raw.read_text()).get("f_micro_w", []):
            ns = rec.get("ns") or ""
            if rec.get("f_micro_w") is not None:
                rows[ns] = {k: round(float(rec[k]), 5) for k in
                            ("f_micro_w", "pr_micro_w", "rc_micro_w", "tau", "cov_w")
                            if rec.get(k) is not None}
        rows["_rows_submitted"] = int(keep.sum())
        return rows


t0 = time.time()
res = {"board_line": "cafaeval <obo> <predDir> <gt> -ia <IA> -out_dir <..> -toi <toi> "
                     "-prop fill -norm cafa -threads N -no_orphans   "
                     "(evaluation.nf:211-215; NO -max_terms, NO -th_step => defaults "
                     "max_terms=None -> VECTORISED parser, th_step=0.01)",
       "positivity_guard": "the vectorised parser has no positivity guard, so submitted "
                           "non-positive scores are STORED and block fill's ancestor "
                           "inheritance: measured cost 0.0933 on pk-bpo. Every arm here "
                           "submits score > tau_pre with tau_pre >= 0, so the guard always holds.",
       "board_published": BOARD,
       "arms": {}}

for tau_pre, label in ((0.0, "guard_only"), (0.4, "prefilter_tau_pre_0.4")):
    res["arms"][label] = {}
    for cat in ("NK", "LK", "PK"):
        r = score(cat, tau_pre)
        res["arms"][label][cat] = r
        bp = (r or {}).get("biological_process", {})
        print(f"  {label:22s} {cat}-BP  f={bp.get('f_micro_w')}  pr={bp.get('pr_micro_w')} "
              f"rc={bp.get('rc_micro_w')}  tau={bp.get('tau')}  rows={(r or {}).get('_rows_submitted'):,}"
              f"   ({time.time()-t0:.0f}s)", flush=True)
        json.dump(res, open(OUT / "build_submission.json", "w"), indent=1)

print(f"\n=== our recipe under the board's REAL line, vs what the board published ===", flush=True)
for cat in ("NK", "LK", "PK"):
    g = res["arms"]["guard_only"][cat].get("biological_process", {}).get("f_micro_w")
    p = res["arms"]["prefilter_tau_pre_0.4"][cat].get("biological_process", {}).get("f_micro_w")
    b = BOARD[cat]
    print(f"  {cat}-BP  guard_only {g}   prefilter@0.4 {p}   board published {b}", flush=True)
    if g is not None:
        print(f"        guard_only - board = {g - b:+.4f}", flush=True)
json.dump(res, open(OUT / "build_submission.json", "w"), indent=1)
print("DONE", flush=True)
