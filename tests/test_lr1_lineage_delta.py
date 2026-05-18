"""Tests for scripts/lr1_lineage_delta.py.

Covers the LR.1 delta regenerator: pointing the script at a synthetic
runs tree should produce the right per-cell delta and the CSV that
ships under ``experiments/lr1/lineage_delta.csv`` should match the
script's stdout.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

# pyproject's pytest config puts scripts/ on the pythonpath.
import lr1_lineage_delta as ld

REPO = Path(__file__).resolve().parents[1]
COMMITTED_CSV = REPO / "experiments" / "lr1" / "lineage_delta.csv"


def _write_run(run_dir: Path, fmax: float, *, status: str = "ok") -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run.json").write_text(json.dumps({
        "status": status,
        "metrics": {"test_fmax": fmax},
    }))


def test_rows_compute_per_cell_delta(tmp_path: Path) -> None:
    reranker_runs = tmp_path / "study_v23"
    baseline_runs = tmp_path / "study_v24_no_lineage"
    # Two cells with deterministic deltas.
    _write_run(reranker_runs / "bench-v1-K5-v226-lineage_nk-bpo", 0.50)
    _write_run(baseline_runs / "bench-v1-K5-v226-nolineage_nk-bpo", 0.40)
    _write_run(reranker_runs / "bench-v1-K5-v226-lineage_pk-mfo", 0.30)
    _write_run(baseline_runs / "bench-v1-K5-v226-nolineage_pk-mfo", 0.20)
    rows = ld._rows(reranker_runs, baseline_runs)
    by_cell = {r["cell"]: r for r in rows}
    assert pytest.approx(by_cell["nk-bpo"]["delta"], abs=1e-9) == 0.10
    assert pytest.approx(by_cell["pk-mfo"]["delta"], abs=1e-9) == 0.10
    # Missing cells stay None (script tolerates absent runs).
    assert by_cell["lk-cco"]["reranker_lab_fmax"] is None
    assert by_cell["lk-cco"]["baseline_lab_fmax"] is None
    assert by_cell["lk-cco"]["delta"] is None


def test_failed_run_status_treated_as_missing(tmp_path: Path) -> None:
    reranker_runs = tmp_path / "study_v23"
    baseline_runs = tmp_path / "study_v24_no_lineage"
    _write_run(
        reranker_runs / "bench-v1-K5-v226-lineage_nk-bpo", 0.99,
        status="error",
    )
    _write_run(baseline_runs / "bench-v1-K5-v226-nolineage_nk-bpo", 0.40)
    rows = ld._rows(reranker_runs, baseline_runs)
    by_cell = {r["cell"]: r for r in rows}
    # The reranker arm was an error status, so its fmax does not flow
    # through and the delta drops to None.
    assert by_cell["nk-bpo"]["reranker_lab_fmax"] is None
    assert by_cell["nk-bpo"]["delta"] is None


def test_csv_round_trip(tmp_path: Path) -> None:
    reranker_runs = tmp_path / "study_v23"
    baseline_runs = tmp_path / "study_v24_no_lineage"
    _write_run(reranker_runs / "bench-v1-K5-v226-lineage_nk-bpo", 0.5)
    _write_run(baseline_runs / "bench-v1-K5-v226-nolineage_nk-bpo", 0.4)
    rows = ld._rows(reranker_runs, baseline_runs)
    out = tmp_path / "delta.csv"
    ld._write_csv(rows, out)
    read = list(csv.DictReader(out.open()))
    nk_bpo = next(r for r in read if r["cell"] == "nk-bpo")
    assert nk_bpo["reranker_lab_fmax"] == "0.5000"
    assert nk_bpo["baseline_lab_fmax"] == "0.4000"
    assert nk_bpo["delta"] == "0.1000"


def test_committed_csv_matches_canonical_shape() -> None:
    # The committed artefact is the source of truth for LR.1
    # acceptance; this test pins its schema so a future regression
    # cannot silently change the column order or the cell coverage.
    assert COMMITTED_CSV.exists(), (
        "experiments/lr1/lineage_delta.csv is the LR.1 acceptance "
        "artefact; regenerate via scripts/lr1_lineage_delta.py --write"
    )
    rows = list(csv.DictReader(COMMITTED_CSV.open()))
    assert [r["cell"] for r in rows] == list(ld.CELLS)
    expected_cols = [
        "cell", "category", "aspect",
        "reranker_lab_fmax", "baseline_lab_fmax", "delta",
    ]
    assert list(rows[0].keys()) == expected_cols
