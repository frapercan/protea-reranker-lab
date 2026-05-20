"""Tests for scripts/lr4_v18_selective.py.

Covers the LR.4 closure: the leakage-free re-run of the selective-rerank
policy on bench-v1-K5-v226-lineage-prostt5. The committed CSV under
``experiments/lr4/v18_selective_delta.csv`` is the acceptance artefact;
the script regenerates it from ``runs/lb2_multiseed/cis.json`` when
present, or from documented LB.2 multi-seed sweep numbers otherwise.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

# pyproject's pytest config puts scripts/ on the pythonpath.
import lr4_v18_selective as ls

REPO = Path(__file__).resolve().parents[1]
COMMITTED_CSV = REPO / "experiments" / "lr4" / "v18_selective_delta.csv"


def test_documented_aggregate_matches_lb2_champion() -> None:
    """The aggregate selective avg must equal the documented champion 0.6215.

    Source of truth: project_lb2_leakage_fixed_champion memory and
    EXPERIMENTS.md ``LB.2 multi-seed sweep`` section. Any drift means
    the canonical constants have diverged from the documented sweep.
    """
    rows = ls._rows(cis_payload=None)
    agg = ls._aggregate(rows)
    assert pytest.approx(agg["selective_avg_fmax"], abs=1e-4) == 0.6215
    assert pytest.approx(agg["all_baseline_avg_fmax"], abs=1e-4) == 0.5818
    assert pytest.approx(agg["delta_vs_all_baseline"], abs=1e-4) == 0.0397
    # Delta vs leaky 0.4562 (memory-only pre-leakage-fix record).
    assert pytest.approx(agg["delta_vs_leaky"], abs=1e-4) == 0.1653


def test_selective_policy_applies_reranker_on_nklk_only() -> None:
    rows = ls._rows(cis_payload=None)
    by_cell = {r["cell"]: r for r in rows}
    for cell in ("nk-bpo", "nk-mfo", "nk-cco",
                 "lk-bpo", "lk-mfo", "lk-cco"):
        assert by_cell[cell]["policy"] == "reranker"
        assert by_cell[cell]["source"] == "EXPERIMENTS.md"
    for cell in ("pk-bpo", "pk-mfo", "pk-cco"):
        assert by_cell[cell]["policy"] == "baseline"
        # PK cells fall back to baseline; selective equals baseline,
        # delta against baseline must be exactly zero.
        assert by_cell[cell]["selective_fmax"] == by_cell[cell]["baseline_fmax"]
        assert by_cell[cell]["delta_vs_baseline"] == 0.0


def test_cis_json_overrides_documented_values(tmp_path: Path) -> None:
    """When ``runs/lb2_multiseed/cis.json`` is present, use it as source."""
    cis = {
        "nk-bpo": {"mean": 0.99, "ci_lo": 0.98, "ci_hi": 1.00},
        "nk-mfo": {"mean": 0.50, "ci_lo": 0.48, "ci_hi": 0.52},
    }
    cis_path = tmp_path / "cis.json"
    cis_path.write_text(json.dumps(cis))
    payload = ls._load_cis(cis_path)
    rows = ls._rows(payload)
    by_cell = {r["cell"]: r for r in rows}
    # nk-bpo: override from cis.json
    assert by_cell["nk-bpo"]["selective_fmax"] == 0.99
    assert by_cell["nk-bpo"]["source"] == "lb2_multiseed/cis.json"
    # The CI half-width is derived from ci_hi - ci_lo / 2.
    assert pytest.approx(by_cell["nk-bpo"]["ci_half"], abs=1e-9) == 0.01
    # Cells not in cis.json fall back to documented values.
    assert by_cell["nk-cco"]["source"] == "EXPERIMENTS.md"


def test_missing_cis_uses_documented_fallback(tmp_path: Path) -> None:
    """A missing ``cis.json`` is not an error: documented values win."""
    missing = tmp_path / "does_not_exist.json"
    payload = ls._load_cis(missing)
    assert payload is None
    rows = ls._rows(payload)
    by_cell = {r["cell"]: r for r in rows}
    assert by_cell["nk-bpo"]["source"] == "EXPERIMENTS.md"


def test_csv_round_trip(tmp_path: Path) -> None:
    rows = ls._rows(cis_payload=None)
    agg = ls._aggregate(rows)
    out = tmp_path / "delta.csv"
    ls._write_csv(rows, agg, out)
    read = list(csv.DictReader(out.open()))
    # Filter out the blank separator row and the aggregate footer.
    data_rows = [r for r in read if r["cell"] in by_cell_helper()]
    assert {r["cell"] for r in data_rows} == set(ls.CELLS)


def by_cell_helper() -> set[str]:
    return set(ls.CELLS)


def test_committed_csv_matches_canonical_shape() -> None:
    """The committed artefact pins schema + cell coverage for LR.4.

    Mirrors the LR.1 acceptance pattern in
    ``tests/test_lr1_lineage_delta.py``: a future regression cannot
    silently change column order or cell coverage.
    """
    assert COMMITTED_CSV.exists(), (
        "experiments/lr4/v18_selective_delta.csv is the LR.4 "
        "acceptance artefact; regenerate via "
        "scripts/lr4_v18_selective.py --write"
    )
    rows = list(csv.DictReader(COMMITTED_CSV.open()))
    data_rows = [r for r in rows if r["cell"] in ls.CELLS]
    assert [r["cell"] for r in data_rows] == list(ls.CELLS)
    expected_cols = [
        "cell", "tier", "aspect", "policy",
        "selective_fmax", "ci_half", "baseline_fmax",
        "delta_vs_baseline", "source",
    ]
    assert list(data_rows[0].keys()) == expected_cols
