"""Validate the harness reproduces the published per-category LAFA numbers.
Build the prediction file from the EXISTING per-category eval_scores.parquet
(reranker_score, global min-max to (0,1]), restrict to the 7401 query set,
KNN fallback for the rest, score 3x. Expect:
  NK mfo0.602 bpo0.309 cco0.431 ; LK mfo0.519 bpo0.348 cco0.419 ;
  PK mfo0.235 bpo0.117 cco0.254.
"""
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

import lafa_harness as H

S = Path("/tmp/claude-1000/-home-frapercan-Thesis2/afd2c43a-ede7-46dc-94dd-9745808d2194/scratchpad")
EVAL_SCORES = S / "rerank_out" / "eval_scores.parquet"
WORK = S / "per_aspect" / "_validate"
WORK.mkdir(parents=True, exist_ok=True)

EXPECT = {
    "NK": {"mfo": 0.602, "bpo": 0.309, "cco": 0.431},
    "LK": {"mfo": 0.519, "bpo": 0.348, "cco": 0.419},
    "PK": {"mfo": 0.235, "bpo": 0.117, "cco": 0.254},
}


def main():
    t = pq.read_table(EVAL_SCORES)
    prot = np.asarray(t.column("protein_accession").to_pylist())
    go = np.asarray(t.column("go_term_id").to_pylist())
    rr = t.column("reranker_score").to_numpy(zero_copy_only=False).astype(np.float64)
    mn, mx = rr.min(), rr.max()
    norm = (rr - mn) / (mx - mn)
    queries = set(H.query_ids())
    keep = np.array([p in queries for p in prot])
    rows = zip(prot[keep], go[keep], norm[keep])
    pred_file = WORK / "pred.tsv"
    ncov, nfb = H.build_pred_file(rows, pred_file)
    print(f"covered proteins={ncov} fallback rows={nfb}", flush=True)
    res = H.score_all(pred_file, WORK)
    print(json.dumps(res, indent=1))
    print("\n=== baseline reproduction check ===")
    ok = True
    for cat in ["NK", "LK", "PK"]:
        for a in ["mfo", "bpo", "cco"]:
            got = res[cat].get(a)
            exp = EXPECT[cat][a]
            d = None if got is None else round(got - exp, 4)
            flag = "OK" if (got is not None and abs(got - exp) <= 0.003) else "DIFF"
            if flag == "DIFF":
                ok = False
            print(f"{cat}-{a}: got={got} exp={exp} d={d} {flag}")
    print("ALL MATCH" if ok else "MISMATCH - investigate")
    (WORK / "baseline_repro.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
