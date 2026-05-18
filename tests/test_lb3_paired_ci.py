"""Tests for scripts/lb3_paired_ci.py.

Covers:
1. CSV schema: correct columns and cell coverage.
2. Bootstrap / paired-difference math: CI bounds bracket the mean
   difference; a positive lift yields sig_95=1; zero lift yields sig_95=0.
3. Reproducibility: fixed seed always yields identical CI bounds;
   a different seed yields different bounds (real resampling, not
   closed-form).
4. Edge case: identical arms (PK selective-deploy policy) correctly
   reported as zero delta / not significant.
5. Committed CSV: the shipped artefact matches the expected schema and
   is consistent with the known champion statistics from the LB.2 sweep.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

# pyproject puts scripts/ on sys.path
import lb3_paired_ci as lb3

REPO = Path(__file__).resolve().parents[1]
COMMITTED_CSV = REPO / "experiments" / "lb3" / "per_cell_paired_ci.csv"

# ---------------------------------------------------------------------------
# 1. CSV schema
# ---------------------------------------------------------------------------


def test_committed_csv_schema() -> None:
    """Committed CSV exists and carries the correct column order and coverage."""
    assert COMMITTED_CSV.exists(), (
        "experiments/lb3/per_cell_paired_ci.csv is the LB.3 acceptance "
        "artefact; regenerate via scripts/lb3_paired_ci.py --write"
    )
    rows = list(csv.DictReader(COMMITTED_CSV.open()))
    assert [r["cell"] for r in rows] == list(lb3.CELLS)
    assert list(rows[0].keys()) == lb3.CSV_COLUMNS


# ---------------------------------------------------------------------------
# 2. Paired-difference math
# ---------------------------------------------------------------------------


def test_positive_lift_is_significant() -> None:
    """A clear champion lift (>> 0) must yield sig_95=1."""
    champion = [0.70, 0.71, 0.69]
    baseline = [0.60, 0.61, 0.60]
    ci = lb3.paired_bootstrap_ci(champion, baseline, n_iter=2000, seed=42)
    assert ci["paired_diff_mean"] == pytest.approx(
        np.mean(np.array(champion) - np.array(baseline)), abs=1e-9
    )
    assert ci["paired_diff_ci_lo"] > 0, (
        "CI lower bound must be > 0 for a clearly positive lift"
    )


def test_zero_lift_not_significant() -> None:
    """Identical arms must yield paired_diff == 0 and sig_95=0."""
    values = [0.50, 0.51, 0.49]
    ci = lb3.paired_bootstrap_ci(values, values, n_iter=2000, seed=42)
    assert ci["paired_diff_mean"] == pytest.approx(0.0, abs=1e-9)
    assert ci["paired_diff_ci_lo"] == pytest.approx(0.0, abs=1e-9)
    assert ci["paired_diff_ci_hi"] == pytest.approx(0.0, abs=1e-9)
    # sig_95 requires ci_lo > 0, which is False here
    all_identical: dict[str, dict[str, list[float]]] = {
        c: {"champion": values, "baseline": values} for c in lb3.CELLS
    }
    rows = lb3.compute_rows(all_identical, n_iter=500, seed=42)
    nk_bpo_row = next(r for r in rows if r["cell"] == "nk-bpo")
    assert nk_bpo_row["sig_95"] == 0


# ---------------------------------------------------------------------------
# 3. Reproducibility
# ---------------------------------------------------------------------------


def test_fixed_seed_is_reproducible() -> None:
    """Same seed produces identical CI bounds."""
    champion = [0.70, 0.71, 0.69]
    baseline = [0.60, 0.61, 0.60]
    ci_a = lb3.paired_bootstrap_ci(champion, baseline, n_iter=1000, seed=42)
    ci_b = lb3.paired_bootstrap_ci(champion, baseline, n_iter=1000, seed=42)
    assert ci_a["paired_diff_ci_lo"] == ci_b["paired_diff_ci_lo"]
    assert ci_a["paired_diff_ci_hi"] == ci_b["paired_diff_ci_hi"]


def test_different_seeds_differ() -> None:
    """Different seeds produce different CI bounds (real resampling).

    We use a larger synthetic dataset (30 seeds) so the bootstrap
    distribution has enough distinct outcomes to detect seed differences.
    """
    rng = np.random.default_rng(0)
    champion = (rng.normal(0.70, 0.02, size=30)).tolist()
    baseline = (rng.normal(0.60, 0.02, size=30)).tolist()
    ci_42 = lb3.paired_bootstrap_ci(champion, baseline, n_iter=1000, seed=42)
    ci_99 = lb3.paired_bootstrap_ci(champion, baseline, n_iter=1000, seed=99)
    assert (
        ci_42["paired_diff_ci_lo"] != ci_99["paired_diff_ci_lo"]
        or ci_42["paired_diff_ci_hi"] != ci_99["paired_diff_ci_hi"]
    )


# ---------------------------------------------------------------------------
# 4. Edge cases
# ---------------------------------------------------------------------------


def test_pk_cells_report_zero_delta_in_committed_csv() -> None:
    """PK cells must report zero paired delta (selective-deploy policy)."""
    assert COMMITTED_CSV.exists()
    rows = {r["cell"]: r for r in csv.DictReader(COMMITTED_CSV.open())}
    for cell in ("pk-bpo", "pk-mfo", "pk-cco"):
        assert float(rows[cell]["paired_diff_mean"]) == pytest.approx(0.0, abs=1e-9), (
            f"{cell}: expected zero delta (PK falls back to KNN baseline)"
        )
        assert int(rows[cell]["sig_95"]) == 0, (
            f"{cell}: must not be marked significant under selective-deploy policy"
        )


# ---------------------------------------------------------------------------
# 5. NK+LK cells are all significant in the committed CSV
# ---------------------------------------------------------------------------


def test_nk_lk_cells_all_significant_in_committed_csv() -> None:
    """All 6 NK+LK reranked cells must show sig_95=1 in the committed CSV.

    This is a regression guard on the champion bar from the LB.2 sweep:
    if any NK+LK cell CI crosses zero, the champion claim is undermined.
    """
    assert COMMITTED_CSV.exists()
    rows = {r["cell"]: r for r in csv.DictReader(COMMITTED_CSV.open())}
    nk_lk_cells = [c for c in lb3.CELLS if not c.startswith("pk-")]
    for cell in nk_lk_cells:
        assert int(rows[cell]["sig_95"]) == 1, (
            f"{cell}: expected sig_95=1 (champion must be above baseline "
            "at 95% level; champion bar 0.6215 would be contradicted otherwise)"
        )
        assert float(rows[cell]["paired_diff_ci_lo"]) > 0, (
            f"{cell}: CI lower bound must be strictly > 0"
        )
