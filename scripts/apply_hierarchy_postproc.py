#!/usr/bin/env python3
"""Apply CondProbMod / soft-propagation hierarchy post-processing to predictions.

Thin entrypoint for the R2.1 ablation. It reads a CAFA-format prediction
file (``protein<TAB>go_term<TAB>score``, no header) plus a ``parent_map.json``
(from ``scripts/export_parent_map.py``), applies one or both hierarchy-aware
passes, and writes a new prediction file ready to feed to cafaeval / the lab's
evaluator.

Passes (composable, applied in this order when both requested):

``--condprobmod``
    Treat the input scores as conditional edge probabilities
    ``P(term | parent)`` and rebuild the marginal per-term probability via the
    parent-product recursion (:func:`reconstruct_marginal_probs`). Use this when
    the booster was *trained* on the CondProbMod conditional target (see
    :func:`build_conditional_mask`).

``--soft-prop``
    Apply the soft two-way Pmin/Pmax blend
    (:func:`soft_propagate_scores`), default 0.7/0.3, enforcing
    ``parent >= child`` consistency on the GO DAG.

Ablation usage (off the F0 validation frame), comparing IA-Fmax before/after::

    # 1. baseline predictions already exist as preds.baseline.tsv
    # 2. soft propagation pass:
    poetry run python scripts/apply_hierarchy_postproc.py \\
        --pred preds.baseline.tsv \\
        --parent-map datasets/<frame>/parent_map.json \\
        --soft-prop --out preds.softprop.tsv
    # 3. score both with cafaeval and read the IA-Fmax delta (target ~+0.04
    #    for CondProbMod per ProtBoost).

This script is pure numpy + stdlib (no torch / psycopg2): safe under the lab's
no-heavy-deps CI rule.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from protea_reranker_lab.condprobmod import reconstruct_marginal_probs
from protea_reranker_lab.propagation import load_parent_map
from protea_reranker_lab.soft_propagation import soft_propagate_scores


def _read_predictions(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read a ``protein<TAB>go<TAB>score`` file into parallel arrays."""
    proteins: list[str] = []
    go_terms: list[str] = []
    scores: list[float] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            proteins.append(parts[0])
            go_terms.append(parts[1])
            scores.append(float(parts[2]))
    return (
        np.array(proteins, dtype=object),
        np.array(go_terms, dtype=object),
        np.array(scores, dtype=np.float32),
    )


def _write_predictions(
    path: Path, proteins: np.ndarray, go_terms: np.ndarray, scores: np.ndarray,
) -> None:
    with open(path, "w") as f:
        for p, g, s in zip(proteins, go_terms, scores):
            f.write(f"{p}\t{g}\t{float(s):.6f}\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--pred", required=True, type=Path,
                    help="CAFA-format predictions (protein TAB go TAB score).")
    ap.add_argument("--parent-map", required=True, type=Path,
                    help="parent_map.json from scripts/export_parent_map.py.")
    ap.add_argument("--out", required=True, type=Path, help="output predictions path.")
    ap.add_argument("--condprobmod", action="store_true",
                    help="Rebuild marginals from conditional edge probabilities.")
    ap.add_argument("--soft-prop", action="store_true",
                    help="Apply the soft two-way Pmin/Pmax blend.")
    ap.add_argument("--pmax-weight", type=float, default=0.7,
                    help="Blend weight on the bottom-up Pmax direction (default 0.7).")
    args = ap.parse_args(argv)

    if not args.condprobmod and not args.soft_prop:
        print("[err] pass at least one of --condprobmod / --soft-prop",
              file=sys.stderr)
        return 2

    parent_map = load_parent_map(args.parent_map)
    proteins, go_terms, scores = _read_predictions(args.pred)
    print(f"[postproc] read {len(scores)} predictions, "
          f"{len(parent_map)} DAG terms")

    if args.condprobmod:
        scores = reconstruct_marginal_probs(proteins, go_terms, scores, parent_map)
        print("[postproc] applied CondProbMod marginal reconstruction")
    if args.soft_prop:
        scores = soft_propagate_scores(
            proteins, go_terms, scores, parent_map, pmax_weight=args.pmax_weight,
        )
        print(f"[postproc] applied soft Pmin/Pmax (pmax_weight={args.pmax_weight})")

    _write_predictions(args.out, proteins, go_terms, scores)
    print(f"[postproc] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
