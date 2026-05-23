"""Paired-bootstrap champion comparison against the KNN baseline.

This module is the public entry point for statistical comparison
between a trained reranker and the KNN-only ``vote_count`` baseline.
It re-exports :func:`bootstrap_paired_fmax` and :class:`BootstrapResult`
from :mod:`protea_reranker_lab.bootstrap` so that documentation and
scripts can import from a stable ``compare`` namespace.

Typical usage::

    from protea_reranker_lab.compare import bootstrap_paired_fmax, BootstrapResult

    result = bootstrap_paired_fmax(
        cell="nk-mfo",
        predictions_parquet=Path("runs/winner/predictions.parquet"),
        source_eval_parquet=Path("datasets/bench-v1-K{k}-v{val_band}-lineage-{plm_short}/eval.parquet"),
        eval_snapshot_pair=None,
        workdir=Path("/tmp/bootstrap_work"),
    )
    print(result.to_json())

**Published champion (multi-seed, leakage-fixed, 2026-05-17):**
Selective average cafaeval Fmax **0.6215 +/- 0.0014** on the
``bench-v1-K{k}-v{val_band}-lineage-{plm_short}`` family
(``k=5``, ``val_band=226``, leakage-fixed feature set), NK+LK cells
only. Six NK+LK confidence intervals are all strictly positive at the
95% level (N=10000 bootstrap iterations). This is the number cited in
Chapter 6 of the doctoral thesis.

See memory ``project_lb2_leakage_fixed_champion`` and
``project_lb3_paired_ci_2026_05_18`` for full details.
"""

from __future__ import annotations

from .bootstrap import BootstrapResult, bootstrap_paired_fmax

__all__ = ["BootstrapResult", "bootstrap_paired_fmax"]
