"""Tests for FARM-EXP.9 partial pass deliverables.

Validates:
1. Schema of cells_to_rerun.csv (source of truth for all FARM-EXP.9 work).
2. Schema of any completed run.json files under runs/transversal/farm_exp_9_rep_*.
3. Schema of partial_ci.csv (if present).
4. Stop-condition guard: no champion-cell CI fully below zero.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CELLS_CSV = REPO / "experiments" / "farm_exp_9" / "cells_to_rerun.csv"
TRANSVERSAL_DIR = REPO / "runs" / "transversal"
PARTIAL_CI_CSV = REPO / "experiments" / "farm_exp_9" / "partial_ci.csv"

REQUIRED_CELLS_CSV_COLUMNS = {
    "cell_id",
    "source",
    "axis_tuple",
    "cell",
    "seed",
    "pre_leakage_fmax",
    "pre_leakage_dataset",
    "target_eval_set",
    "estimated_runtime_minutes",
    "priority",
    "status",
    "notes",
}

VALID_STATUS_VALUES = {"pending", "done", "error", "skipped"}
VALID_SOURCES = {
    "study_v9_replication",
    "study_v9_ablation",
    "study_v9_hparam",
    "standalone",
}
CHAMPION_CELLS = {"nk-bpo", "nk-mfo", "nk-cco", "lk-bpo", "lk-mfo", "lk-cco"}


class TestCellsToRerunCSV:
    """Validate cells_to_rerun.csv — the canonical cell list."""

    def test_file_exists(self):
        assert CELLS_CSV.exists(), f"cells_to_rerun.csv missing: {CELLS_CSV}"

    def test_required_columns_present(self):
        with CELLS_CSV.open() as fh:
            reader = csv.DictReader(fh)
            cols = set(reader.fieldnames or [])
        missing = REQUIRED_CELLS_CSV_COLUMNS - cols
        assert not missing, f"Missing columns in cells_to_rerun.csv: {missing}"

    def test_row_count_in_expected_range(self):
        """Expect ~80-100 pre-leakage cells (axis-map estimate)."""
        with CELLS_CSV.open() as fh:
            rows = list(csv.DictReader(fh))
        assert 80 <= len(rows) <= 120, (
            f"Expected 80-120 rows, got {len(rows)}. "
            "Check axis-map §Pre-computed cells in snapshot."
        )

    def test_all_status_values_valid(self):
        with CELLS_CSV.open() as fh:
            rows = list(csv.DictReader(fh))
        invalid = {r["status"] for r in rows if r["status"] not in VALID_STATUS_VALUES}
        assert not invalid, f"Invalid status values: {invalid}"

    def test_target_eval_set_is_filtered(self):
        """All cells must target bench-v1-K5-filtered (the leakage-free eval)."""
        with CELLS_CSV.open() as fh:
            rows = list(csv.DictReader(fh))
        bad = [r["cell_id"] for r in rows if r["target_eval_set"] != "bench-v1-K5-filtered"]
        assert not bad, f"Cells with wrong target_eval_set: {bad[:5]}"

    def test_pre_leakage_dataset_is_bench_v1_k5(self):
        """Source dataset must be bench-v1-K5 (pre-leakage)."""
        with CELLS_CSV.open() as fh:
            rows = list(csv.DictReader(fh))
        bad = [
            r["cell_id"]
            for r in rows
            if r.get("pre_leakage_dataset") not in ("bench-v1-K5", "")
        ]
        assert not bad, f"Cells with unexpected pre_leakage_dataset: {bad[:5]}"

    def test_all_sources_are_known(self):
        with CELLS_CSV.open() as fh:
            rows = list(csv.DictReader(fh))
        unknown = {r["source"] for r in rows if r["source"] not in VALID_SOURCES}
        assert not unknown, f"Unknown source values: {unknown}"

    def test_priority_field_is_integer(self):
        with CELLS_CSV.open() as fh:
            rows = list(csv.DictReader(fh))
        for r in rows:
            try:
                int(r["priority"])
            except (ValueError, KeyError):
                pytest.fail(f"Non-integer priority in row {r['cell_id']}: {r.get('priority')}")

    def test_axis_tuple_present_and_nonempty(self):
        with CELLS_CSV.open() as fh:
            rows = list(csv.DictReader(fh))
        bad = [r["cell_id"] for r in rows if not r.get("axis_tuple", "").strip()]
        assert not bad, f"Cells missing axis_tuple: {bad[:5]}"


class TestCompletedRunJSON:
    """Validate run.json schema for any completed farm_exp_9 cells."""

    def _get_rep_run_jsons(self):
        """Return run.json paths for all farm_exp_9_rep_* runs with status=ok."""
        if not TRANSVERSAL_DIR.exists():
            return []
        result = []
        for d in TRANSVERSAL_DIR.iterdir():
            if not d.name.startswith("farm_exp_9_rep_"):
                continue
            rj = d / "run.json"
            if not rj.exists():
                continue
            with rj.open() as fh:
                data = json.load(fh)
            # Only test completed runs; in-progress runs are skipped here
            if data.get("status") == "ok":
                result.append(rj)
        return result

    def test_at_least_one_completed_run_exists(self):
        runs = self._get_rep_run_jsons()
        assert runs, (
            "No completed farm_exp_9_rep_*/run.json found under runs/transversal/. "
            "At least 1 replication cell must complete for a valid partial pass."
        )

    def test_run_json_required_fields(self):
        REQUIRED = {
            "run_id", "status", "started_at", "spec_name", "spec_hash",
            "output_dir", "dataset", "metrics",
        }
        for run_json in self._get_rep_run_jsons():
            with run_json.open() as fh:
                d = json.load(fh)
            missing = REQUIRED - set(d)
            assert not missing, f"{run_json}: missing fields {missing}"

    def test_run_json_status_ok(self):
        for run_json in self._get_rep_run_jsons():
            with run_json.open() as fh:
                d = json.load(fh)
            assert d["status"] == "ok", f"{run_json}: status={d['status']!r}"

    def test_run_json_eval_set_is_filtered(self):
        for run_json in self._get_rep_run_jsons():
            with run_json.open() as fh:
                d = json.load(fh)
            ds_name = d.get("dataset", {}).get("name", "")
            assert ds_name == "bench-v1-K5-filtered", (
                f"{run_json}: dataset.name={ds_name!r} (expected bench-v1-K5-filtered)"
            )

    def test_run_json_has_test_fmax(self):
        for run_json in self._get_rep_run_jsons():
            with run_json.open() as fh:
                d = json.load(fh)
            fmax = d.get("metrics", {}).get("test_fmax")
            assert fmax is not None, f"{run_json}: metrics.test_fmax missing"
            assert 0.0 <= fmax <= 1.0, f"{run_json}: test_fmax={fmax} out of [0,1]"

    def test_run_json_farm_exp_9_tag_present(self):
        for run_json in self._get_rep_run_jsons():
            with run_json.open() as fh:
                d = json.load(fh)
            tags = d.get("spec_tags", [])
            assert "farm_exp_9" in tags, f"{run_json}: 'farm_exp_9' tag missing. tags={tags}"

    def test_no_leakage_features_in_run(self):
        """Verify leakage features (anc2vec_query_known_*) are not in the feature set."""
        LEAKAGE = {"anc2vec_query_known_cos", "anc2vec_query_known_maxcos", "anc2vec_query_known_count"}
        for run_json in self._get_rep_run_jsons():
            with run_json.open() as fh:
                d = json.load(fh)
            feat_cols = set(d.get("features", {}).get("feature_columns", []))
            overlap = feat_cols & LEAKAGE
            assert not overlap, (
                f"{run_json}: leakage features present in run: {overlap}"
            )


class TestChampionFlipStopCondition:
    """STOP condition guard: no champion cell's CI should cross zero in the wrong direction."""

    def test_no_champion_cell_ci_fully_below_zero(self):
        """Paired diff CI of a champion cell must not be entirely negative.

        If a champion cell's new fmax is significantly LOWER than its
        pre-leakage fmax (CI fully below zero), that is anomalous and
        requires human review before continuing FARM-EXP.9.

        Note: The comparison is across incompatible eval sets, so this
        test guards against gross anomalies only, not fine-grained diffs.
        """
        if not PARTIAL_CI_CSV.exists():
            pytest.skip("partial_ci.csv not yet written — run with --write flag")
        with PARTIAL_CI_CSV.open() as fh:
            rows = list(csv.DictReader(fh))
        for r in rows:
            if r["cell"] not in CHAMPION_CELLS:
                continue
            try:
                ci_lo = float(r["paired_diff_ci_lo"])
                ci_hi = float(r["paired_diff_ci_hi"])
            except (ValueError, KeyError):
                continue
            assert not (ci_lo < 0 and ci_hi < 0), (
                f"STOP CONDITION: champion cell {r['cell']} CI [{ci_lo:.4f}, {ci_hi:.4f}] "
                "is entirely negative. Leakage fix may have degraded a champion. "
                "Surface for human review before continuing FARM-EXP.9."
            )
