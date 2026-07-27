"""PKW.9. The two cafaeval parsers disagree by 0.0933 on our file, and the board uses the other one.

THE FACT, kill-tested with one variable (`parser_path_killtest.py`). `max_terms=10**9` is a cap
that can never bind, so it is semantically identical to no cap. Same file, same flags, only the
code path differs:

    legacy      (max_terms=10**9)  f_micro_w 0.21293  pr 0.180  cov_w 0.927  tau 0.398
    vectorised  (max_terms=None)   f_micro_w 0.11960  pr 0.083  cov_w 0.981  tau 0.004

**0.0933, purely the parser.** `_pred_parser_legacy` runs only when a cap is set; otherwise
`_pred_parser_vectorised` does. The board never passes `-max_terms`
(`evaluation.nf:211-215`; grep over modules/, workflows/, nextflow.config and main.nf finds
the flag nowhere), so **the board reads us through the vectorised path and every lab number in
this directory was measured through the legacy one.**

THE HYPOTHESIS THIS TESTS. The legacy parser stores a value only when `prob_f > old` with
`old = 0.0` (parser.py:230), so our non-positive scores are silently DROPPED and their cells
stay exactly zero, which is precisely the sentinel `fill` overwrites with a free ancestor. If
the vectorised path instead STORES them (group-max then scatter, no positivity guard), those
cells become small positives, `fill` protects them, inheritance is blocked, and precision
collapses. Coverage rising 0.927 -> 0.981 and precision falling 0.180 -> 0.083 both fit.

THE PREDICTION, and it is falsifiable. If the divergence is exactly the non-positive rows, then
removing them FROM THE FILE must make the vectorised path agree with the legacy one. If it does
not, my explanation is wrong and the divergence is something else I have not found.

  A  legacy,     full file                 expect 0.2129 (the anchor)
  B  vectorised, full file                 expect 0.1196 (the divergence)
  C  vectorised, rows with score > 0       THE TEST: does it return to ~0.2129?
  D  legacy,     rows with score > 0       control: must equal A, since legacy drops them anyway

GATE, naming the quantity (discipline #10): the quantity is |C - A|. Under 0.002 the non-positive
rows explain the divergence completely and the fix is to filter the submitted file. Above it,
something else is also wrong and this must not be reported as solved.

WHY IT MATTERS. If C confirms, then on the board's real code path our submitted negatives are
being stored and are blocking the ancestor inheritance, so the zero-clamp we believed was giving
us a free prefilter is NOT operating there. Filtering the file would then be worth roughly +0.09
on the board rather than the +0.0092 measured through the legacy path in PKW.8.
"""
import json, subprocess, tempfile, time
from pathlib import Path
import numpy as np, pyarrow.parquet as pq

W = Path("/home/frapercan/Thesis2/storage/cooc_experiment")
DS = Path("/home/frapercan/Thesis2/repositories/protea-reranker-lab/datasets/protst-global-train227-test230")
REL = Path("/home/frapercan/Thesis2/CAFA_forever/data/releases/Sep_2025_Mar_2026")
OBO = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
IA = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv"
TOI = str(REL / "groundtruth_terms_of_interest.txt")
PY = "/home/frapercan/Thesis2/repositories/PROTEA/.venv/bin/python"

t = pq.read_table(W / "rerank_out" / "eval_scores.parquet")
cat = np.asarray(t.column("category").to_pylist())
asp = np.asarray(t.column("aspect").to_pylist())
m = (cat == "pk") & (asp == "bpo")
PROT = np.asarray(t.column("protein_accession").to_pylist())[m]
GO = np.asarray(t.column("go_term_id").to_pylist())[m]
RR = t.column("reranker_score").to_numpy(zero_copy_only=False).astype(np.float64)[m]
print(f"{len(RR):,} pk-bpo rows | {(RR <= 0).mean():.1%} are non-positive", flush=True)

DRIVER = '''
import json, signal
from cafaeval.evaluation import cafa_eval
signal.signal(signal.SIGTERM, signal.SIG_DFL)
df,dfs=cafa_eval({args})
out={{}}
for k,v in dfs.items(): out[k]=v.reset_index().to_dict(orient="records")
json.dump(out,open("{o}","w"),default=str)
'''


def run(name, max_terms, positives_only):
    keep = RR > 0 if positives_only else np.ones(len(RR), dtype=bool)
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "pd"
        d.mkdir()
        with (d / "predictions_protea.tsv").open("w") as fh:
            for p_, g_, v in zip(PROT[keep], GO[keep], RR[keep]):
                fh.write(f"{p_}\t{g_}\t{v:.6f}\n")
        gt = Path(td) / "gt.tsv"
        gt.write_text((DS / "gt_pk_bp.tsv").read_text())
        raw = Path(td) / "r.json"
        args = (f'"{OBO}","{d}","{gt}",ia="{IA}",prop="fill",norm="cafa",no_orphans=True,'
                f'toi_file={repr(TOI)},max_terms={max_terms},th_step=0.001,'
                f'n_cpu=4,weighted_only=False')
        drv = Path(td) / "d.py"
        drv.write_text(DRIVER.format(args=args, o=str(raw)))
        r = subprocess.run([PY, str(drv)], capture_output=True, text=True, timeout=7200)
        if r.returncode != 0:
            print(f"  FAILED {name}: {r.stderr[-300:]}", flush=True)
            return None
        best = None
        for rec in json.loads(raw.read_text()).get("f_micro_w", []):
            if (rec.get("ns") or "") == "biological_process" and rec.get("f_micro_w") is not None:
                best = rec
        return ({k: round(float(best[k]), 5) for k in
                 ("f_micro_w", "pr_micro_w", "rc_micro_w", "tau", "cov_w") if best.get(k) is not None},
                int(keep.sum())) if best else (None, int(keep.sum()))


ARMS = [
    ("A_legacy_full_file", 10**9, False),
    ("B_vectorised_full_file", None, False),
    ("C_vectorised_positives_only", None, True),
    ("D_legacy_positives_only", 10**9, True),
]
t0 = time.time()
res = {"fact": "legacy (max_terms=10**9, a cap that never binds) 0.21293 vs vectorised "
               "(max_terms=None) 0.11960 on the same file and flags: 0.0933, purely the code path.",
       "the_board_uses": "the VECTORISED path: it never passes -max_terms (evaluation.nf:211-215).",
       "hypothesis": "legacy drops non-positive scores (parser.py:230, `if prob_f > old` from "
                     "old=0.0) so their cells stay the exact zero that `fill` overwrites with a "
                     "free ancestor; the vectorised path stores them, `fill` protects them, and "
                     "inheritance is blocked.",
       "gate": "|C - A| < 0.002 => the non-positive rows explain the divergence completely.",
       "arms": {}}
for name, mt, pos in ARMS:
    r, n = run(name, mt, pos)
    res["arms"][name] = {"result": r, "rows_submitted": n}
    print(f"  {name:30s} rows={n:>7,} -> {r}  ({time.time()-t0:.0f}s)", flush=True)
    json.dump(res, open(W / "parser_divergence.json", "w"), indent=1)

A = (res["arms"]["A_legacy_full_file"]["result"] or {}).get("f_micro_w")
C = (res["arms"]["C_vectorised_positives_only"]["result"] or {}).get("f_micro_w")
if A and C:
    res["delta_C_minus_A"] = round(C - A, 5)
    res["NONPOSITIVES_EXPLAIN_IT"] = abs(C - A) < 0.002
json.dump(res, open(W / "parser_divergence.json", "w"), indent=1)
print(f"\n=== Do the non-positive rows explain the parser divergence? ({time.time()-t0:.0f}s) ===", flush=True)
print(f"  A legacy, full file          = {A}", flush=True)
print(f"  C vectorised, positives only = {C}", flush=True)
if A and C:
    print(f"  |C - A| = {abs(C-A):.5f}  -> explained: {abs(C-A) < 0.002}", flush=True)
    print("  If explained: on the board's real path our negatives are STORED and block the", flush=True)
    print("  ancestor inheritance, so filtering the submitted file is worth ~+0.09 there,", flush=True)
    print("  not the +0.0092 PKW.8 measured through the legacy path.", flush=True)
print("DONE", flush=True)
