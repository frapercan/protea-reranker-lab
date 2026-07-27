"""Gate variant (ADR-D43): PK scored by raw-KNN (1 - distance) instead of the
collapsed PK reranker; NK/LK stay reranked. Only re-scores the 3 PK cells; NK/LK
cells are reused from the reranked run. Same LAFA-exact harness.
"""
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

import lafa_harness as H

S = Path("/tmp/claude-1000/-home-frapercan-Thesis2/afd2c43a-ede7-46dc-94dd-9745808d2194/scratchpad")
SCORES = S / "clfassoc_out" / "eval_scores.parquet"
WORK = S / "clfassoc_out" / "lafa_work_gatepk"
WORK.mkdir(parents=True, exist_ok=True)


def main():
    t = pq.read_table(SCORES)
    prot = np.asarray(t.column("protein_accession").to_pylist())
    go = np.asarray(t.column("go_term_id").to_pylist())
    cat = np.asarray(t.column("category").to_pylist())
    dist = t.column("distance").to_numpy(zero_copy_only=False).astype(np.float64)
    knn = 1.0 - dist

    # PK candidates only, scored by raw-KNN
    queries = set(H.query_ids())
    mpk = cat == "pk"
    keep = mpk & np.array([p in queries for p in prot])
    rows = list(zip(prot[keep], go[keep], knn[keep]))
    pred_file = WORK / "pred.tsv"
    # no KNN fallback append (PK fallback already KNN), but harness needs query set;
    # restrict fallback to PK-missing queries = none meaningful -> pass empty set
    ncov, nfb = H.build_pred_file(rows, pred_file, fallback_prots=set())
    print(f"PK rows={keep.sum()} covered_proteins={ncov}", flush=True)
    res = H.score_all(pred_file, WORK)
    pk = {a: res["PK"].get(a) for a in ["mfo", "bpo", "cco"]}
    print("GATE PK (raw-KNN):", json.dumps(pk, indent=1), flush=True)
    (WORK / "pk_gate.json").write_text(json.dumps(pk, indent=1))


if __name__ == "__main__":
    main()
