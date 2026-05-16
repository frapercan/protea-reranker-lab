"""Tests for FARM-EXP.3 scripts/bootstrap_cis.py.

Covers:

1. CLI ``--group-by`` parsing (validation, sentinel, dedup).
2. Grouping shape: 2 PLMs x 3 K values => 6 fake runs; each grouping
   axis produces the expected row count.
3. CI math: bounds bracket the point estimate, seed is honored
   (determinism), and a real resampling (not a normal-approx)
   computes the CI.
4. Filename convention matches the documented form.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

# pyproject's pytest config puts scripts/ on the pythonpath.
import bootstrap_cis as bc

REPO = Path(__file__).resolve().parents[1]


# ---------------------------------------------------- fixture builder


def _synthesize_fixture(out_dir: Path, *, samples_per_run: int = 200) -> None:
    """Write 6 fake run records spanning 2 PLMs x 3 K values.

    Each run carries a known-mean Beta-distributed fmax sample array
    so CI bounds are predictable. esm2_3b cells are centered above
    ankh_base cells by ~0.05, so grouping by plm yields a clear
    signed gap.
    """
    plms = ["esm2_3b", "ankh_base"]
    ks = [5, 10, 15]
    plm_offset = {"esm2_3b": 0.05, "ankh_base": 0.00}
    rng_master = np.random.default_rng(123)
    out_dir.mkdir(parents=True, exist_ok=True)
    for plm in plms:
        for k in ks:
            mean = 0.45 + plm_offset[plm] + 0.005 * (k - 10) / 5
            # Beta with mean=mean, fairly tight (alpha+beta=200).
            alpha = mean * 200
            beta = (1 - mean) * 200
            local_rng = np.random.default_rng(rng_master.integers(0, 2**31 - 1))
            samples = local_rng.beta(alpha, beta, size=samples_per_run)
            baseline_samples = local_rng.beta(
                alpha * 0.1, beta, size=samples_per_run
            )
            payload = {
                "run_id": f"{plm}__k{k}__run01",
                "axis": {
                    "plm": plm,
                    "k": k,
                    "reranker": "lgbm.per_cell_9",
                    "features": "v6",
                    "eval_set": "bench-v1-K5-filtered",
                    "propagation": "tpr_pred",
                    "ensemble": "none",
                },
                "shortid": f"{plm[:4]}{k:02d}faaa",
                "cell": "nk-bpo",
                "fmax_samples": samples.tolist(),
                "fmax_point": float(samples.mean()),
                "baseline_fmax_samples": baseline_samples.tolist(),
                "baseline_fmax_point": float(baseline_samples.mean()),
            }
            path = out_dir / f"{plm}__k{k}.json"
            path.write_text(json.dumps(payload))


# ---------------------------------------------------- parse_group_by


def test_parse_group_by_sentinel_empty() -> None:
    assert bc.parse_group_by(None) == ()
    assert bc.parse_group_by("") == ()
    assert bc.parse_group_by("all") == ()


def test_parse_group_by_single() -> None:
    assert bc.parse_group_by("plm") == ("plm",)


def test_parse_group_by_multi_preserves_order() -> None:
    assert bc.parse_group_by("plm,k,reranker") == ("plm", "k", "reranker")


def test_parse_group_by_deduplicates() -> None:
    assert bc.parse_group_by("plm,k,plm") == ("plm", "k")


def test_parse_group_by_rejects_unknown_axis() -> None:
    with pytest.raises(ValueError) as exc:
        bc.parse_group_by("plm,does_not_exist")
    assert "does_not_exist" in str(exc.value)


def test_parse_group_by_strips_whitespace() -> None:
    assert bc.parse_group_by("  plm , k  ") == ("plm", "k")


# ---------------------------------------------------- canonical axes


def test_canonical_axes_match_catalog_vocabulary() -> None:
    # Mirrors the keys emitted by scripts/build_study_specs.py for
    # the FARM-EXP.2 transversal catalog stanzas.
    expected = {
        "plm", "k", "reranker", "features",
        "eval_set", "propagation", "ensemble",
    }
    assert set(bc.CANONICAL_AXES) == expected


# ---------------------------------------------------- group_signature


def test_group_signature_flat() -> None:
    assert bc.group_signature((), ()) == "group=all"


def test_group_signature_single_axis() -> None:
    assert bc.group_signature(("plm",), ("esm2_3b",)) == "group=plm=esm2_3b"


def test_group_signature_multi_axis() -> None:
    sig = bc.group_signature(("plm", "k"), ("esm2_3b", 5))
    assert sig == "group=plm=esm2_3b__k=5"


# ---------------------------------------------------- grouping shape


def test_group_by_plm_yields_two_buckets(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    _synthesize_fixture(runs_dir)
    records = bc.load_runs(runs_dir)
    assert len(records) == 6

    buckets = bc.group_runs(records, ("plm",))
    assert set(buckets.keys()) == {("esm2_3b",), ("ankh_base",)}
    for key, bucket in buckets.items():
        assert len(bucket) == 3, f"{key} expected 3 runs (one per K)"


def test_group_by_k_yields_three_buckets(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    _synthesize_fixture(runs_dir)
    records = bc.load_runs(runs_dir)
    buckets = bc.group_runs(records, ("k",))
    assert set(buckets.keys()) == {(5,), (10,), (15,)}
    for bucket in buckets.values():
        assert len(bucket) == 2  # one per PLM


def test_group_by_plm_k_yields_six_singleton_buckets(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    _synthesize_fixture(runs_dir)
    records = bc.load_runs(runs_dir)
    buckets = bc.group_runs(records, ("plm", "k"))
    assert len(buckets) == 6
    for bucket in buckets.values():
        assert len(bucket) == 1


def test_group_by_all_yields_one_bucket(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    _synthesize_fixture(runs_dir)
    records = bc.load_runs(runs_dir)
    buckets = bc.group_runs(records, ())
    assert list(buckets.keys()) == [()]
    assert len(buckets[()]) == 6


# ---------------------------------------------------- CI math


def test_bootstrap_group_ci_brackets_point_estimate(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    _synthesize_fixture(runs_dir)
    records = bc.load_runs(runs_dir)
    ci = bc.bootstrap_group_ci(records, n_iter=500, seed=42)
    # CI is on the *mean*, very tight around the pooled mean, so
    # the pooled mean must lie inside.
    assert ci.fmax_ci_lo <= ci.fmax_mean <= ci.fmax_ci_hi
    # 95% CI half-width should be small relative to the spread.
    assert ci.fmax_ci_hi - ci.fmax_ci_lo > 0.0
    assert ci.fmax_ci_hi - ci.fmax_ci_lo < 0.05  # tight on a pooled mean
    # n_iter stamped correctly.
    assert ci.n_iter == 500


def test_bootstrap_group_ci_is_deterministic(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    _synthesize_fixture(runs_dir)
    records = bc.load_runs(runs_dir)
    ci_a = bc.bootstrap_group_ci(records, n_iter=500, seed=42)
    ci_b = bc.bootstrap_group_ci(records, n_iter=500, seed=42)
    assert ci_a.fmax_ci_lo == ci_b.fmax_ci_lo
    assert ci_a.fmax_ci_hi == ci_b.fmax_ci_hi
    # And a different seed gives a different CI (real resampling,
    # not a closed-form expression).
    ci_c = bc.bootstrap_group_ci(records, n_iter=500, seed=99)
    assert (ci_a.fmax_ci_lo != ci_c.fmax_ci_lo) or (
        ci_a.fmax_ci_hi != ci_c.fmax_ci_hi
    )


def test_bootstrap_group_ci_paired_diff_present(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    _synthesize_fixture(runs_dir)
    records = bc.load_runs(runs_dir)
    ci = bc.bootstrap_group_ci(records, n_iter=500, seed=42)
    assert ci.paired_diff_mean is not None
    assert ci.paired_diff_ci_lo is not None
    assert ci.paired_diff_ci_hi is not None
    # Candidate runs were synthesized above baseline by construction,
    # so the paired difference CI excludes zero on the low side.
    assert ci.paired_diff_ci_lo > 0.0


def test_bootstrap_group_ci_plm_gap_is_signed(tmp_path: Path) -> None:
    # esm2_3b runs were centered ~0.05 above ankh_base runs. The
    # grouped CI on each PLM should reflect that ordering.
    runs_dir = tmp_path / "runs"
    _synthesize_fixture(runs_dir, samples_per_run=400)
    records = bc.load_runs(runs_dir)
    buckets = bc.group_runs(records, ("plm",))
    cis = {
        key[0]: bc.bootstrap_group_ci(records, n_iter=500, seed=42)
        for key, records in buckets.items()
    }
    # esm2_3b mean must be above ankh_base mean by a comfortable
    # margin (synthetic gap was 0.05).
    assert cis["esm2_3b"].fmax_mean > cis["ankh_base"].fmax_mean + 0.02


# ---------------------------------------------------- end-to-end run


def test_run_emits_csv_per_grouping(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    out_dir = tmp_path / "out"
    _synthesize_fixture(runs_dir)
    written = bc.run(
        runs_dir=runs_dir,
        out_dir=out_dir,
        group_by=("plm", "k"),
        n_iter=200,
        seed=42,
        strict=False,
        emit_plot=False,
    )
    assert len(written) == 6
    for csv_path in written:
        assert csv_path.exists()
        rows = list(csv.DictReader(csv_path.open()))
        assert len(rows) == 1
        row = rows[0]
        assert row["plm"] in {"esm2_3b", "ankh_base"}
        assert row["k"] in {"5", "10", "15"}
        assert float(row["fmax_ci_lo"]) <= float(row["fmax_mean"])
        assert float(row["fmax_mean"]) <= float(row["fmax_ci_hi"])
        assert int(row["n_iter"]) == 200
        assert int(row["n_runs"]) == 1


def test_run_emits_plot_when_requested(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    out_dir = tmp_path / "out"
    _synthesize_fixture(runs_dir)
    bc.run(
        runs_dir=runs_dir,
        out_dir=out_dir,
        group_by=("plm",),
        n_iter=200,
        seed=42,
        strict=False,
        emit_plot=True,
    )
    pngs = sorted(out_dir.glob("*.png"))
    assert len(pngs) == 2
    for p in pngs:
        # Files must exist with non-trivial size.
        assert p.stat().st_size > 1000


def test_run_flat_grouping_emits_single_csv(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    out_dir = tmp_path / "out"
    _synthesize_fixture(runs_dir)
    written = bc.run(
        runs_dir=runs_dir,
        out_dir=out_dir,
        group_by=(),
        n_iter=200,
        seed=42,
        strict=False,
        emit_plot=False,
    )
    assert len(written) == 1
    assert written[0].name == "group=all.csv"
    rows = list(csv.DictReader(written[0].open()))
    assert len(rows) == 1
    assert int(rows[0]["n_runs"]) == 6


# ---------------------------------------------------- tolerant mode


def test_records_missing_axis_grouped_under_unknown(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    # Two well-formed runs + one with no axis block.
    rng = np.random.default_rng(0)
    for i, plm in enumerate(["esm2_3b", "ankh_base"]):
        runs_dir.joinpath(f"good_{i}.json").write_text(json.dumps({
            "run_id": f"good_{i}",
            "axis": {"plm": plm, "k": 5, "reranker": "none",
                     "features": "knn-only", "eval_set": "bench-v1-K5-filtered",
                     "propagation": "tpr_pred", "ensemble": "none"},
            "fmax_samples": rng.beta(45, 55, size=100).tolist(),
        }))
    runs_dir.joinpath("noaxis.json").write_text(json.dumps({
        "run_id": "noaxis",
        "fmax_samples": rng.beta(45, 55, size=100).tolist(),
    }))
    records = bc.load_runs(runs_dir)
    buckets = bc.group_runs(records, ("plm",), strict=False)
    keys = set(buckets.keys())
    assert ("esm2_3b",) in keys
    assert ("ankh_base",) in keys
    assert (bc.AXIS_UNKNOWN_LABEL,) in keys


def test_strict_mode_raises_on_missing_axis(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    rng = np.random.default_rng(0)
    runs_dir.joinpath("noaxis.json").write_text(json.dumps({
        "run_id": "x",
        "fmax_samples": rng.beta(45, 55, size=100).tolist(),
    }))
    records = bc.load_runs(runs_dir)
    with pytest.raises(KeyError):
        bc.group_runs(records, ("plm",), strict=True)
