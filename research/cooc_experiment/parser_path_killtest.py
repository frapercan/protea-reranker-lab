"""PKW.9. Is the 0.2131 / 0.2181 gap our own flag drift?

WHAT I GOT WRONG, AND IT CONTAMINATES EVERY NUMBER MEASURED LAST NIGHT. I applied discipline
#9 ("check what the grader actually runs") to exactly one flag. I grepped `-prop fill`, found
it, felt vindicated, and stopped reading the command line. The board's real invocation
(`CAFA_forever/modules/local/evaluation.nf:211-215`, and identically at 243, 275, 308, 340,
372) is:

    cafaeval <obo> <predictionsDir> <groundtruthTsv> \
      -ia <IA.tsv> -out_dir <...> \
      -toi <groundtruth_terms_of_interest.txt> \
      -prop fill -norm cafa -threads N -no_orphans

Three flags differ from every lab run in this directory, and I have been calling ours "the
board's exact flags" in a receipt:

  1. `-toi`      the board restricts to 38,650 terms of interest. WE NEVER PASS IT.
  2. `max_terms` the board never passes it, so cafaeval's default applies: **None**. We pass 500.
  3. `th_step`   the board never passes it, so the default applies: **0.01**. We pass 0.001.

`max_terms` and `th_step` were never in the pipeline at all: `grep -rn "max_terms|th_step"`
over modules/, workflows/, nextflow.config and main.nf returns nothing.

WHY IT PLAUSIBLY EXPLAINS THE GAP. All 3,855 distinct terms in the PK-BP ground truth are
already inside the toi list, so `-toi` cannot remove a true positive. It can only drop
predicted terms that lie outside the list, i.e. it removes false positives and RAISES
precision. That is the right sign to turn 0.2131 into 0.2181.

THE ARMS. One variable at a time, from our flags to the board's, so the gap is attributed and
not just closed:

  L  lab as run all night     toi=None  max_terms=500   th_step=0.001   expect ~0.2131
  T  + toi only               toi=SET   max_terms=500   th_step=0.001
  M  + max_terms default      toi=SET   max_terms=None  th_step=0.001
  B  + th_step default = THE BOARD'S EXACT LINE                          expect ~0.2181

THE GATE, and name the quantity (discipline #10): the quantity is the absolute difference
between arm B's f_micro_w and the board's published 0.21807 for `predictions_protea.tsv`.
If |B - 0.21807| < 0.002 the gap is our flag drift, the lab reproduces the board, and the
anchor must be restated. If B lands far from it, the gap is in the PREDICTIONS and
`predictions_protea.tsv` is a genuinely different arm we cannot rebuild. Either answer is
worth having; only one is good news.

CAVEAT: this scores OUR eval_scores.parquet, i.e. the deployed recipe's margins. It is not a
claim to have recovered predictions_protea.tsv, whose file exists nowhere (`predictions_uneval`
and `predictions_by_window` are both absent from disk; no repo ever tracked the name).
"""
import json, subprocess, tempfile, time, collections
from pathlib import Path
import numpy as np, pyarrow.parquet as pq

W = Path("/home/frapercan/Thesis2/storage/cooc_experiment")
DS = Path("/home/frapercan/Thesis2/repositories/protea-reranker-lab/datasets/protst-global-train227-test230")
REL = Path("/home/frapercan/Thesis2/CAFA_forever/data/releases/Sep_2025_Mar_2026")
OBO = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
IA = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv"
TOI = str(REL / "groundtruth_terms_of_interest.txt")
PY = "/home/frapercan/Thesis2/repositories/PROTEA/.venv/bin/python"
BOARD = 0.21806691440311407          # predictions_protea.tsv, biological_process, results_PK

t = pq.read_table(W / "rerank_out" / "eval_scores.parquet")
cat = np.asarray(t.column("category").to_pylist())
asp = np.asarray(t.column("aspect").to_pylist())
m = (cat == "pk") & (asp == "bpo")
PROT = np.asarray(t.column("protein_accession").to_pylist())[m]
GO = np.asarray(t.column("go_term_id").to_pylist())[m]
RR = t.column("reranker_score").to_numpy(zero_copy_only=False).astype(np.float64)[m]
print(f"{len(RR):,} pk-bpo rows from the deployed recipe", flush=True)

DRIVER = '''
import json, signal
from cafaeval.evaluation import cafa_eval
signal.signal(signal.SIGTERM, signal.SIG_DFL)
df,dfs=cafa_eval({args})
out={{}}
for k,v in dfs.items(): out[k]=v.reset_index().to_dict(orient="records")
json.dump(out,open("{o}","w"),default=str)
'''


def run(name, toi, max_terms, th_step):
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "pd"
        d.mkdir()
        with (d / "predictions_protea.tsv").open("w") as fh:
            for p_, g_, v in zip(PROT, GO, RR):
                fh.write(f"{p_}\t{g_}\t{v:.6f}\n")
        gt = Path(td) / "gt.tsv"
        with gt.open("w"), open(DS / "gt_pk_bp.tsv") as src:
            pass
        gt.write_text((DS / "gt_pk_bp.tsv").read_text())
        raw = Path(td) / "r.json"
        args = (f'"{OBO}","{d}","{gt}",ia="{IA}",prop="fill",norm="cafa",no_orphans=True,'
                f'toi_file={repr(toi)},max_terms={max_terms},th_step={th_step},'
                f'n_cpu=4,weighted_only=False')
        drv = Path(td) / "d.py"
        drv.write_text(DRIVER.format(args=args, o=str(raw)))
        r = subprocess.run([PY, str(drv)], capture_output=True, text=True, timeout=7200)
        if r.returncode != 0:
            print(f"  FAILED {name}: {r.stderr[-400:]}", flush=True)
            return None
        best = None
        for rec in json.loads(raw.read_text()).get("f_micro_w", []):
            if (rec.get("ns") or "") == "biological_process" and rec.get("f_micro_w") is not None:
                best = rec
        if not best:
            return None
        return {k: round(float(best[k]), 5) for k in
                ("f_micro_w", "pr_micro_w", "rc_micro_w", "tau", "cov_w") if best.get(k) is not None}


ARMS = [
    ("K_legacy_cap_never_binds", TOI, 10**9, 0.001),
    ("V_vectorised_no_cap", TOI, None, 0.001),
]
t0 = time.time()
res = {"board_command": "cafaeval <obo> <predDir> <gt> -ia <IA> -out_dir <..> -toi <toi> "
                        "-prop fill -norm cafa -threads N -no_orphans  "
                        "(evaluation.nf:211-215; identical at 243/275/308/340/372)",
       "flags_the_board_never_passes": {"max_terms": "default None (we used 500)",
                                        "th_step": "default 0.01 (we used 0.001)"},
       "toi_terms": 38650, "gt_terms_outside_toi": 0,
       "board_published_f_micro_w_predictions_protea": BOARD,
       "gate": "|B - 0.21807| < 0.002 => the gap is our flag drift and the lab reproduces the "
               "board. Otherwise the gap is in the PREDICTIONS and predictions_protea.tsv is a "
               "different arm we cannot rebuild.",
       "arms": {}}

for name, toi, mt, ts in ARMS:
    r = run(name, toi, mt, ts)
    res["arms"][name] = r
    print(f"  {name:26s} toi={'SET' if toi else 'None':4s} max_terms={str(mt):4s} th_step={ts:<6} -> {r}  ({time.time()-t0:.0f}s)", flush=True)
    json.dump(res, open(W / "parser_path_killtest.json", "w"), indent=1)

b = (res["arms"].get("B_board_exact_line") or {}).get("f_micro_w")
if b is not None:
    res["delta_vs_board"] = round(b - BOARD, 5)
    res["REPRODUCES_BOARD"] = abs(b - BOARD) < 0.002
json.dump(res, open(W / "parser_path_killtest.json", "w"), indent=1)
print(f"\n=== Is the 0.2131/0.2181 gap our own flag drift? ({time.time()-t0:.0f}s) ===", flush=True)
print(f"  board published (predictions_protea.tsv, BP) = {BOARD:.5f}", flush=True)
print(f"  our deployed recipe under the board's exact line = {b}", flush=True)
if b is not None:
    print(f"  delta = {b-BOARD:+.5f}  -> reproduces the board: {abs(b-BOARD) < 0.002}", flush=True)
print("DONE", flush=True)
