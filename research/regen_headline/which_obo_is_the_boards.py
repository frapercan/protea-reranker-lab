"""Which information accretion table does the board score us with? One variable: the IA.

WHAT I GOT WRONG, AND THE REPO ALREADY KNEW IT. I concluded that we cannot reproduce the board
because our ontology is from the wrong month: `protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo`
declares `data-version: releases/2025-07-22` while the board's t0 is September 2025, so I wrote
that "the directory name lies". It does not.

`protea/core/band_registry.py` and `docs/source/architecture/evaluation.rst` pin band **v227** to
GOA release **227 (2025-09-04)** and ontology **`releases/2025-07-22`**, and a CI guard
(`scripts/check_band_registry.py`, wired into `lint.yml`) enforces it. The September GOA release
was built against the July ontology; that pairing is deliberate, documented and tested. The
directory is the LAFA t0 for September 2025 and it holds exactly the obo that t0 pinned. **My
discipline #14 was born of a misreading and the ontology is not the discrepancy.**

`docs/IA_PROVENANCE_v227.md` goes further and names the exact phenomenon we spent the night
rediscovering: mixing a snapshot or IA across bands "inflates a **phantom PROTEA-vs-LAFA gap**".
It compared three IA tables on 2026-06-06 and chose one, on the grounds that it is "the exact
table the deployed LAFA endpoint scored its predictions against".

SO THE QUESTION IS NARROW AND TESTABLE. `predictions_7401_reranked.tsv` is the one file that
exists both locally and on the board, which scores its PK-BP at **0.1170**. Our re-run on the
board's own line gives **0.20132**. If the IA is the difference, one of the three tables named in
that provenance note will produce the board's number.

  A  lafa_t0_Sep_2025/IA.tsv          39,906 terms   the authoritative v227 IA, what we have used all along
  B  IA-swissprot-exp-v227.txt        38,739 terms   the lab's independent recompute (r=0.982 against A)
  C  IA_cafa6.tsv                     40,122 terms   the generic CAFA6 corpus, explicitly REJECTED for v227

Everything else is held fixed: the same file, the same `groundtruth_PK.tsv`, the same obo, and the
board's exact line (`-toi`, `prop=fill`, `norm=cafa`, `no_orphans`, no `max_terms`, no `th_step`).

THE GATE, quantity named: the quantity is |f - 0.1170| for each IA. Under 0.002 identifies the
board's table and closes the frame question. If none of them lands there, **the IA is not the
discrepancy either**, and I say so rather than reaching for a fourth candidate. Discipline #4:
five tidy explanations have already died this run, and each one fitted the evidence.

Note on C: the provenance note rejects it for v227 on the grounds that it is a different corpus.
It is included here precisely because it was rejected: if the board turns out to have used it, the
rejection is the discrepancy, and that is worth knowing.
"""
import json, subprocess, tempfile, time
from pathlib import Path

REL = Path("/home/frapercan/Thesis2/CAFA_forever/data/releases/Sep_2025_Mar_2026")
SUB = Path("/home/frapercan/Thesis2/repositories/protea-reranker-lab/results/clean_227230/"
           "lafa_submission/predictions_7401_reranked.tsv")
OBO = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
TOI = str(REL / "groundtruth_terms_of_interest.txt")
PY = "/home/frapercan/Thesis2/repositories/PROTEA/.venv/bin/python"
BOARD = 0.1170

# The IA axis came back flat: all three tables give 0.201-0.204 against the board's 0.117, so the
# IA is NOT the discrepancy. The obo governs True-Path propagation, which decides the propagated
# ground truth AND the propagated prediction, so it is the next single variable. Four exist here.
IAS = {"A_lafa_t0_Sep_2025": "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv"}
OBOS = {
    "obo_2025-07-22_v227_pin": "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo",
    "obo_2025-06-01": "/home/frapercan/lafa_workdir/data/go-basic.obo",
    "obo_2026-01-23": "/home/frapercan/Thesis2/lafa-smoke/data/go-basic.obo",
    "obo_2026-03-25": "/home/frapercan/Thesis2/protea-neural-head/data/go-basic.obo",
}

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

t0 = time.time()
res = {"question": "which OBO does the board propagate with? (the IA axis came back FLAT: all three tables give 0.201-0.204)",
       "control_file": str(SUB),
       "board_PK_BP_for_this_file": BOARD,
       "what_the_repo_already_knew": "band_registry.py pins v227 to GOA 227 (2025-09-04) + obo "
                                     "releases/2025-07-22, CI-enforced. The July obo is CORRECT for "
                                     "v227; my 'the directory name lies' was a misreading. "
                                     "docs/IA_PROVENANCE_v227.md names the 'phantom PROTEA-vs-LAFA "
                                     "gap' from mixing IA across bands and chose A on 2026-06-06.",
       "gate": "|f - 0.1170| < 0.002 identifies the board's table. If none lands, the IA is NOT the "
               "discrepancy and no fourth candidate gets invented.",
       "arms": {}}

for name, OBO in OBOS.items():
    ia = IAS["A_lafa_t0_Sep_2025"]
    if not Path(OBO).exists():
        print(f"  {name}: MISSING {OBO}", flush=True)
        continue
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "pd"
        d.mkdir()
        (d / "predictions_7401_reranked.tsv").write_text(SUB.read_text())
        raw = Path(td) / "r.json"
        drv = Path(td) / "d.py"
        drv.write_text(DRIVER.format(obo=OBO, pd=str(d), gt=str(REL / "groundtruth_PK.tsv"),
                                     ia=ia, toi=TOI, o=str(raw)))
        r = subprocess.run([PY, str(drv)], capture_output=True, text=True, timeout=7200)
        if r.returncode != 0:
            print(f"  {name}: FAILED {r.stderr[-200:]}", flush=True)
            continue
        best = None
        for rec in json.loads(raw.read_text()).get("f_micro_w", []):
            if (rec.get("ns") or "") == "biological_process" and rec.get("f_micro_w") is not None:
                best = rec
        f = round(float(best["f_micro_w"]), 5)
        res["arms"][name] = {"obo": OBO, "f_micro_w": f,
                             "pr": round(float(best["pr_micro_w"]), 4),
                             "rc": round(float(best["rc_micro_w"]), 4),
                             "tau": round(float(best["tau"]), 3),
                             "cov_w": round(float(best["cov_w"]), 4),
                             "delta_vs_board": round(f - BOARD, 5),
                             "IS_THE_BOARDS": abs(f - BOARD) < 0.002}
        print(f"  {name:26s} f={f:.5f}  delta vs board {f-BOARD:+.5f}  "
              f"-> is the board's: {abs(f-BOARD) < 0.002}   ({time.time()-t0:.0f}s)", flush=True)
    json.dump(res, open(Path(__file__).parent / "which_obo_is_the_boards.json", "w"), indent=1)

hit = [k for k, v in res["arms"].items() if v.get("IS_THE_BOARDS")]
res["verdict"] = ("the board's IA is " + hit[0]) if hit else (
    "NONE of the three IA tables reproduces the board. The IA is not the discrepancy either.")
json.dump(res, open(Path(__file__).parent / "which_obo_is_the_boards.json", "w"), indent=1)
print(f"\n=== board PK-BP for this file: {BOARD} ===", flush=True)
print(f"  {res['verdict']}", flush=True)
print("DONE", flush=True)
