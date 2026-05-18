"""Tests for scripts/lm3_feature_importance.py.

Covers: CSV schema validation, importance-extraction math against a tiny
synthetic LightGBM model, and reproducibility under a fixed seed.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pytest

import lm3_feature_importance as lm3

REPO = Path(__file__).resolve().parents[1]
COMMITTED_CSV = REPO / "experiments" / "lm3" / "feature_importance_per_aspect.csv"
EXPECTED_COLS = ["cell", "category", "aspect", "feature", "importance", "rank"]
CELLS = list(lm3.CELLS)
ASPECTS = list(lm3.ASPECTS)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_run(run_dir: Path, fi: dict[str, float], *, status: str = "ok") -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run.json").write_text(json.dumps({
        "status": status,
        "feature_importance": fi,
    }))


def _make_cell_dir(base: Path, cell: str) -> Path:
    return base / f"bench-v1-K5-v226-lineage_{cell}"


# ---------------------------------------------------------------------------
# 1. CSV schema: columns, cell coverage, rank monotonicity
# ---------------------------------------------------------------------------

def test_committed_csv_schema() -> None:
    """Committed CSV has the required columns and covers all nine cells."""
    assert COMMITTED_CSV.exists(), (
        "experiments/lm3/feature_importance_per_aspect.csv is the LM.3 acceptance "
        "artefact; regenerate via scripts/lm3_feature_importance.py --write"
    )
    rows = list(csv.DictReader(COMMITTED_CSV.open()))
    assert len(rows) > 0
    assert list(rows[0].keys()) == EXPECTED_COLS


def test_committed_csv_cell_coverage() -> None:
    """Each of the nine cells appears at least once in the committed CSV."""
    rows = list(csv.DictReader(COMMITTED_CSV.open()))
    cells_present = {r["cell"] for r in rows}
    for cell in CELLS:
        assert cell in cells_present, f"cell {cell} missing from committed CSV"


def test_committed_csv_rank_monotonicity() -> None:
    """Within each cell, ranks form a gapless sequence starting at 1."""
    rows = list(csv.DictReader(COMMITTED_CSV.open()))
    from collections import defaultdict
    cell_ranks: dict[str, list[int]] = defaultdict(list)
    for r in rows:
        cell_ranks[r["cell"]].append(int(r["rank"]))
    for cell, ranks in cell_ranks.items():
        assert sorted(ranks) == list(range(1, len(ranks) + 1)), (
            f"cell {cell}: ranks are not a gapless 1..N sequence"
        )


# ---------------------------------------------------------------------------
# 2. Importance-extraction math against a tiny synthetic LightGBM model
# ---------------------------------------------------------------------------

def _build_tiny_booster(seed: int = 42) -> tuple[lgb.Booster, dict[str, float]]:
    """Train a tiny LightGBM booster on synthetic data; return booster + expected gain."""
    rng = np.random.default_rng(seed)
    n = 200
    X = rng.standard_normal((n, 3)).astype(np.float32)
    # Binary labels correlated with first feature
    y = (X[:, 0] > 0).astype(np.float32)
    ds = lgb.Dataset(X, label=y, feature_name=["feat_a", "feat_b", "feat_c"], free_raw_data=False)
    params = {"objective": "binary", "num_leaves": 4, "seed": seed, "verbose": -1}
    booster = lgb.train(params, ds, num_boost_round=20)
    fi = dict(zip(
        booster.feature_name(),
        booster.feature_importance(importance_type="gain"),
        strict=False,
    ))
    return booster, fi


def test_gain_metric_order_matches_feature_importance_type() -> None:
    """The feature ranked #1 by gain should be the one with the highest gain value."""
    _, fi = _build_tiny_booster(seed=42)
    ranked = sorted(fi.items(), key=lambda kv: -kv[1])
    top_feat = ranked[0][0]
    assert fi[top_feat] == max(fi.values()), "top-ranked feature does not have highest gain"


def test_build_rows_extracts_ranks_correctly(tmp_path: Path) -> None:
    """build_rows assigns rank=1 to the highest-gain feature in each cell."""
    runs_dir = tmp_path / "study_v23"
    fi_nk_bpo = {"feat_x": 500.0, "feat_y": 200.0, "feat_z": 50.0}
    _write_run(_make_cell_dir(runs_dir, "nk-bpo"), fi_nk_bpo)

    rows = lm3.build_rows(runs_dir)
    nk_bpo_rows = [r for r in rows if r["cell"] == "nk-bpo"]
    rank1 = next(r for r in nk_bpo_rows if r["rank"] == 1)
    assert rank1["feature"] == "feat_x", "rank=1 should be the highest-gain feature"
    assert float(rank1["importance"]) == pytest.approx(500.0)


def test_build_rows_failed_status_skipped(tmp_path: Path) -> None:
    """A run.json with status!='ok' is skipped; no rows emitted for that cell."""
    runs_dir = tmp_path / "study_v23"
    _write_run(_make_cell_dir(runs_dir, "nk-bpo"), {"feat_x": 1.0}, status="failed")
    rows = lm3.build_rows(runs_dir)
    cells = {r["cell"] for r in rows}
    assert "nk-bpo" not in cells, "failed run should produce no rows"


# ---------------------------------------------------------------------------
# 3. Aggregate function and reproducibility under fixed seed
# ---------------------------------------------------------------------------

def test_aggregate_by_aspect_mean_rank(tmp_path: Path) -> None:
    """aggregate_by_aspect returns entries sorted by mean rank ascending."""
    runs_dir = tmp_path / "study_v23"
    # Write three bpo cells with known importances
    for cat, fi in [
        ("nk", {"feat_a": 300.0, "feat_b": 100.0}),
        ("lk", {"feat_a": 200.0, "feat_b": 400.0}),
        ("pk", {"feat_a": 150.0, "feat_b": 50.0}),
    ]:
        _write_run(_make_cell_dir(runs_dir, f"{cat}-bpo"), fi)

    rows = lm3.build_rows(runs_dir)
    agg = lm3.aggregate_by_aspect(rows)

    # feat_a ranks: nk=1, lk=2, pk=1 → mean=1.33
    # feat_b ranks: nk=2, lk=1, pk=2 → mean=1.67
    bpo = {feat: mean_rank for feat, mean_rank, _ in agg["bpo"]}
    assert bpo["feat_a"] == pytest.approx(4 / 3, abs=1e-6)
    assert bpo["feat_b"] == pytest.approx(5 / 3, abs=1e-6)
    # sorted by mean_rank ascending: feat_a first
    assert agg["bpo"][0][0] == "feat_a"


def test_build_rows_reproducible_under_fixed_seed() -> None:
    """build_rows on the committed run.json artefacts is deterministic."""
    runs_dir = REPO / "runs" / "study_v23"
    if not runs_dir.exists():
        pytest.skip("study_v23 run artefacts not present in this worktree (gitignored)")

    rows1 = lm3.build_rows(runs_dir)
    rows2 = lm3.build_rows(runs_dir)
    assert rows1 == rows2, "build_rows is not deterministic across two calls"


def test_write_csv_round_trip(tmp_path: Path) -> None:
    """write_csv produces a file that round-trips back to equivalent float values."""
    runs_dir = tmp_path / "study_v23"
    fi = {"feat_a": 1234.5678, "feat_b": 0.0}
    _write_run(_make_cell_dir(runs_dir, "nk-bpo"), fi)

    rows = lm3.build_rows(runs_dir)
    out = tmp_path / "out.csv"
    lm3.write_csv(rows, out)

    read = list(csv.DictReader(out.open()))
    assert read[0]["cell"] == "nk-bpo"
    assert read[0]["feature"] == "feat_a"
    assert float(read[0]["importance"]) == pytest.approx(1234.5678, abs=1e-3)
    assert int(read[0]["rank"]) == 1
    assert list(read[0].keys()) == EXPECTED_COLS
