"""Compute term-space features on TEST prediction frames (full, all aspects)."""
import os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from features import build_term_features

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ns_map = C.load_obo_namespace()
    ia = C.load_ia()
    code_idx, codes = C.load_codes()
    depth = C.compute_depths()

    for cat, pred in [("lk", C.LK_PRED), ("pk", C.PK_PRED)]:
        t0 = time.time()
        df = pd.read_csv(pred, sep="\t", header=None,
                         names=["protein_accession", "go_term_id", "base_score"])
        bp = build_term_features(df, ns_map, code_idx, codes, depth, ia, topk=10)
        bp.to_parquet(os.path.join(HERE, f"test_{cat}_bp_feat.parquet"))
        # also persist the non-BP rows so we can reassemble the full pred file
        non_bp = df[~df["go_term_id"].map(ns_map).eq("biological_process")].copy()
        non_bp.to_parquet(os.path.join(HERE, f"test_{cat}_nonbp.parquet"))
        print(f"[{cat}] total={len(df)} BP={len(bp)} nonBP={len(non_bp)} "
              f"code_cov_bp={bp['has_code'].mean():.3f} ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
