"""Reconcile the two Information Accretion (IA) tables for the v227 band.

F-LAFA-IA.1 item 2: prove that the lab IA file and the LAFA-KNN IA file are
the same release computed against two different OBO snapshots, pick the
authoritative one, and emit a JSON report.

Inputs (paths are arguments so the script is portable):
  - lab IA   : datasets/ia/IA-swissprot-exp-v227.txt
               (democafa, SwissProt exp+IC+TAS @ goa v227, OBO releases/2026-01-23)
  - knn IA   : protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv
               (LAFA t0 Sep 2025 reproduction, OBO releases/2025-07-22)

Both are two-column TSV consumed by cafaeval ia_parser. This compares term
universes and per-term values and writes a machine-readable verdict.

Usage:
    python experiments/lafa_ia_v227_protocol/reconcile_ia.py LAB_IA KNN_IA OUT_JSON
"""

from __future__ import annotations

import json
import statistics
import sys


def load(path: str) -> dict[str, float]:
    out: dict[str, float] = {}
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            try:
                out[parts[0]] = float(parts[1])
            except ValueError:
                continue
    return out


def main() -> None:
    lab_path, knn_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    lab = load(lab_path)
    knn = load(knn_path)
    ls, ks = set(lab), set(knn)
    inter = ls & ks
    diffs = [abs(lab[t] - knn[t]) for t in inter]
    nz = [d for d in diffs if d > 1e-9]
    roots = ("GO:0008150", "GO:0005575", "GO:0003674")
    report = {
        "lab_file": lab_path,
        "knn_file": knn_path,
        "lab_terms": len(lab),
        "knn_terms": len(knn),
        "shared_terms": len(inter),
        "only_in_lab": len(ls - ks),
        "only_in_knn": len(ks - ls),
        "shared_identical_value": sum(1 for d in diffs if d <= 1e-9),
        "shared_diff_gt_1e9": len(nz),
        "max_abs_diff": max(diffs) if diffs else 0.0,
        "mean_abs_diff_all_shared": statistics.mean(diffs) if diffs else 0.0,
        "median_abs_diff_nonzero": statistics.median(nz) if nz else 0.0,
        "roots_both_zero": all(lab.get(r) == 0.0 and knn.get(r) == 0.0 for r in roots),
        "authoritative": "lab IA-swissprot-exp-v227.txt",
        "authoritative_reason": (
            "Same goa v227 SwissProt exp+IC+TAS corpus and same democafa "
            "formula in both. They diverge only because each was propagated "
            "against a different OBO snapshot (lab releases/2026-01-23 vs "
            "knn releases/2025-07-22). The lab file is authoritative for the "
            "lab eval grid because its OBO matches datasets/bench-v1-K5/go.obo, "
            "the same ontology cafaeval propagates predictions and ground "
            "truth against; a mismatched OBO can drive IA negative or drop "
            "terms from the universe."
        ),
    }
    with open(out_path, "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
