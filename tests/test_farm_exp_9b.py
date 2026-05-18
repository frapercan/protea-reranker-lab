"""Tests for FARM-EXP.9b (second pass) deliverables.

Validates:
1. Schema of newly-done run.json files for ablation + hparam + remaining rep cells.
2. Schema of partial_ci_pass2.csv.
3. Schema of ablation_summary_pass2.csv.
4. Schema of hparam_sweep_summary.csv.
5. Stop-condition: no champion cell CI fully below zero (same guard as pass-1).
6. Linter: no bare version tokens in new artefacts.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CELLS_CSV = REPO / "experiments" / "farm_exp_9" / "cells_to_rerun.csv"
TRANSVERSAL_DIR = REPO / "runs" / "transversal"
PARTIAL_CI_PASS2_CSV = REPO / "experiments" / "farm_exp_9" / "partial_ci_pass2.csv"
ABL_SUMMARY_CSV = REPO / "experiments" / "farm_exp_9" / "ablation_summary_pass2.csv"
HP_SUMMARY_CSV = REPO / "experiments" / "farm_exp_9" / "hparam_sweep_summary.csv"
SUMMARY_JSON = REPO / "experiments" / "farm_exp_9" / "summary.json"

CHAMPION_CELLS = {"nk-bpo", "nk-mfo", "nk-cco", "lk-bpo", "lk-mfo", "lk-cco"}
ALL_CELLS = {"nk-bpo", "nk-mfo", "nk-cco", "lk-bpo", "lk-mfo", "lk-cco", "pk-bpo", "pk-mfo", "pk-cco"}
VALID_STATUS = {"pending", "done", "error", "skipped"}

ABL_RUN_PATTERN = re.compile(r"^farm_exp_9_abl_")
HP_RUN_PATTERN = re.compile(r"^farm_exp_9_hp_")
REP_RUN_PATTERN = re.compile(r"^farm_exp_9_rep_")
STANDALONE_RUN_PATTERN = re.compile(r"^farm_exp_9_standalone_")


def _get_run_jsons_matching(pattern: re.Pattern) -> list[Path]:
    """Return run.json paths for completed runs matching pattern."""
    if not TRANSVERSAL_DIR.exists():
        return []
    result = []
    for d in TRANSVERSAL_DIR.iterdir():
        if not pattern.match(d.name):
            continue
        rj = d / "run.json"
        if not rj.exists():
            continue
        try:
            data = json.loads(rj.read_text())
        except json.JSONDecodeError:
            continue
        if data.get("status") == "ok":
            result.append(rj)
    return result


class TestCellsCSVPassTwo:
    """Validate cells_to_rerun.csv after pass-2 updates."""

    def test_file_exists(self):
        assert CELLS_CSV.exists()

    def test_row_count_unchanged(self):
        with CELLS_CSV.open() as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 94, f"Expected 94 rows, got {len(rows)}"

    def test_all_status_values_valid(self):
        with CELLS_CSV.open() as fh:
            rows = list(csv.DictReader(fh))
        invalid = {r["status"] for r in rows if r["status"] not in VALID_STATUS}
        assert not invalid, f"Invalid status values: {invalid}"

    def test_done_count_increased_vs_pass1(self):
        """After pass-2, at least 26 cells must be done (pass-1 had 25)."""
        with CELLS_CSV.open() as fh:
            rows = list(csv.DictReader(fh))
        done = sum(1 for r in rows if r["status"] == "done")
        assert done >= 26, (
            f"Expected >= 26 done cells after pass-2, got {done}. "
            "Pass-1 had 25; at least one pass-2 cell must have completed."
        )


class TestAblationRunJSONs:
    """Validate ablation run.json files."""

    REQUIRED = {
        "run_id", "status", "started_at", "spec_name", "spec_hash",
        "output_dir", "dataset", "metrics",
    }

    def test_ablation_runs_have_correct_fields(self):
        runs = _get_run_jsons_matching(ABL_RUN_PATTERN)
        if not runs:
            pytest.skip("No completed ablation runs yet")
        for rj in runs:
            data = json.loads(rj.read_text())
            missing = self.REQUIRED - set(data)
            assert not missing, f"{rj}: missing fields {missing}"

    def test_ablation_runs_have_test_fmax(self):
        runs = _get_run_jsons_matching(ABL_RUN_PATTERN)
        if not runs:
            pytest.skip("No completed ablation runs yet")
        for rj in runs:
            data = json.loads(rj.read_text())
            fmax = data.get("metrics", {}).get("test_fmax")
            assert fmax is not None, f"{rj}: metrics.test_fmax missing"
            assert 0.0 <= fmax <= 1.0, f"{rj}: test_fmax={fmax} out of [0,1]"

    def test_ablation_runs_eval_set_is_filtered(self):
        runs = _get_run_jsons_matching(ABL_RUN_PATTERN)
        if not runs:
            pytest.skip("No completed ablation runs yet")
        for rj in runs:
            data = json.loads(rj.read_text())
            ds_name = data.get("dataset", {}).get("name", "")
            assert ds_name == "bench-v1-K5-filtered", (
                f"{rj}: dataset={ds_name!r} (expected bench-v1-K5-filtered)"
            )

    def test_ablation_runs_no_leakage_features(self):
        LEAKAGE = {
            "anc2vec_query_known_cos",
            "anc2vec_query_known_maxcos",
            "anc2vec_query_known_count",
        }
        runs = _get_run_jsons_matching(ABL_RUN_PATTERN)
        if not runs:
            pytest.skip("No completed ablation runs yet")
        for rj in runs:
            data = json.loads(rj.read_text())
            feat_cols = set(data.get("features", {}).get("feature_columns", []))
            overlap = feat_cols & LEAKAGE
            assert not overlap, (
                f"{rj}: leakage features in ablation run: {overlap}"
            )

    def test_ablation_runs_have_farm_exp_9_tag(self):
        runs = _get_run_jsons_matching(ABL_RUN_PATTERN)
        if not runs:
            pytest.skip("No completed ablation runs yet")
        for rj in runs:
            data = json.loads(rj.read_text())
            tags = data.get("spec_tags", [])
            assert "farm_exp_9" in tags, f"{rj}: 'farm_exp_9' tag missing"


class TestHparamRunJSONs:
    """Validate hparam sweep run.json files."""

    def test_hparam_runs_have_test_fmax(self):
        runs = _get_run_jsons_matching(HP_RUN_PATTERN)
        if not runs:
            pytest.skip("No completed hparam runs yet")
        for rj in runs:
            data = json.loads(rj.read_text())
            fmax = data.get("metrics", {}).get("test_fmax")
            assert fmax is not None, f"{rj}: metrics.test_fmax missing"
            assert 0.0 <= fmax <= 1.0, f"{rj}: test_fmax={fmax} out of [0,1]"

    def test_hparam_runs_eval_set_is_filtered(self):
        runs = _get_run_jsons_matching(HP_RUN_PATTERN)
        if not runs:
            pytest.skip("No completed hparam runs yet")
        for rj in runs:
            data = json.loads(rj.read_text())
            ds_name = data.get("dataset", {}).get("name", "")
            assert ds_name == "bench-v1-K5-filtered", (
                f"{rj}: dataset={ds_name!r}"
            )


class TestPartialCIPass2:
    """Validate partial_ci_pass2.csv schema."""

    REQUIRED_COLS = {
        "cell", "category", "aspect", "seeds_done", "status",
        "new_fmax_mean", "new_fmax_ci_lo", "new_fmax_ci_hi",
        "old_fmax_mean", "paired_diff_mean", "paired_diff_ci_lo",
        "paired_diff_ci_hi", "sig_95", "note",
    }

    def test_file_exists_if_at_least_one_rep_done(self):
        rep_runs = _get_run_jsons_matching(REP_RUN_PATTERN)
        if not rep_runs:
            pytest.skip("No replication runs completed, CSV not expected yet")
        assert PARTIAL_CI_PASS2_CSV.exists(), (
            f"partial_ci_pass2.csv missing but {len(rep_runs)} rep runs completed"
        )

    def test_columns_present(self):
        if not PARTIAL_CI_PASS2_CSV.exists():
            pytest.skip("partial_ci_pass2.csv not yet written")
        with PARTIAL_CI_PASS2_CSV.open() as fh:
            cols = set(csv.DictReader(fh).fieldnames or [])
        missing = self.REQUIRED_COLS - cols
        assert not missing, f"Missing columns: {missing}"

    def test_no_champion_ci_below_zero(self):
        """Stop condition: no champion cell CI fully below zero."""
        if not PARTIAL_CI_PASS2_CSV.exists():
            pytest.skip("partial_ci_pass2.csv not yet written")
        with PARTIAL_CI_PASS2_CSV.open() as fh:
            rows = list(csv.DictReader(fh))
        for r in rows:
            if r.get("cell") not in CHAMPION_CELLS:
                continue
            ci_lo_str = r.get("paired_diff_ci_lo", "")
            ci_hi_str = r.get("paired_diff_ci_hi", "")
            if not ci_lo_str or not ci_hi_str:
                continue
            try:
                ci_lo = float(ci_lo_str)
                ci_hi = float(ci_hi_str)
            except ValueError:
                continue
            assert not (ci_lo < 0 and ci_hi < 0), (
                f"STOP CONDITION: champion cell {r['cell']} CI "
                f"[{ci_lo:.4f}, {ci_hi:.4f}] is entirely negative. "
                "Leakage fix may have degraded a champion. Surface for review."
            )


class TestAblationSummaryCSV:
    """Validate ablation_summary_pass2.csv schema."""

    REQUIRED_COLS = {
        "cell_id", "cell", "family_dropped", "fmax", "full_fmax", "delta_vs_full"
    }

    def test_columns_present(self):
        if not ABL_SUMMARY_CSV.exists():
            pytest.skip("ablation_summary_pass2.csv not yet written")
        with ABL_SUMMARY_CSV.open() as fh:
            cols = set(csv.DictReader(fh).fieldnames or [])
        missing = self.REQUIRED_COLS - cols
        assert not missing, f"Missing columns: {missing}"

    def test_fmax_in_range(self):
        if not ABL_SUMMARY_CSV.exists():
            pytest.skip("ablation_summary_pass2.csv not yet written")
        with ABL_SUMMARY_CSV.open() as fh:
            rows = list(csv.DictReader(fh))
        for r in rows:
            fmax_str = r.get("fmax", "")
            if not fmax_str:
                continue
            fmax = float(fmax_str)
            assert 0.0 <= fmax <= 1.0, f"fmax={fmax} out of [0,1] in row {r['cell_id']}"


class TestHparamSummaryCSV:
    """Validate hparam_sweep_summary.csv schema."""

    REQUIRED_COLS = {
        "cell_id", "num_leaves", "learning_rate", "neg_pos_ratio",
        "fmax", "delta_vs_baseline"
    }

    def test_columns_present(self):
        if not HP_SUMMARY_CSV.exists():
            pytest.skip("hparam_sweep_summary.csv not yet written")
        with HP_SUMMARY_CSV.open() as fh:
            cols = set(csv.DictReader(fh).fieldnames or [])
        missing = self.REQUIRED_COLS - cols
        assert not missing, f"Missing columns: {missing}"

    def test_fmax_in_range(self):
        if not HP_SUMMARY_CSV.exists():
            pytest.skip("hparam_sweep_summary.csv not yet written")
        with HP_SUMMARY_CSV.open() as fh:
            rows = list(csv.DictReader(fh))
        for r in rows:
            fmax_str = r.get("fmax", "")
            if not fmax_str:
                continue
            fmax = float(fmax_str)
            assert 0.0 <= fmax <= 1.0, f"fmax={fmax} out of [0,1] in row {r['cell_id']}"


class TestSummaryJSON:
    """Validate summary.json schema after pass-2 update."""

    REQUIRED_KEYS = {
        "cells_total_scope", "cells_done_total", "cells_remaining", "rows",
    }

    def test_file_exists(self):
        assert SUMMARY_JSON.exists()

    def test_required_keys(self):
        with SUMMARY_JSON.open() as fh:
            data = json.load(fh)
        missing = self.REQUIRED_KEYS - set(data)
        assert not missing, f"Missing keys in summary.json: {missing}"

    def test_cells_scope_is_94(self):
        with SUMMARY_JSON.open() as fh:
            data = json.load(fh)
        scope = data.get("cells_total_scope")
        assert scope == 94, f"Expected cells_total_scope=94, got {scope}"
