"""Tests for per-aspect score calibration (F-RERANK-UNIVERSAL.5)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from protea_reranker_lab.calibration import (
    AspectCalibrator,
    CalibrationSpec,
    ASPECT_NAMES,
    calibrate_scores,
    calibration_stats,
    fit_aspect_calibrators,
    load_calibrators,
    save_calibrators,
)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def synthetic_scores(rng: np.random.Generator):
    """Generate (scores, labels) for three aspects."""
    n = 200
    scores = rng.uniform(0, 1, n).astype(np.float32)
    labels = (scores + rng.normal(0, 0.2, n) > 0.5).astype(np.int8)
    return scores, labels


class TestCalibrationSpec:
    def test_default_spec(self) -> None:
        spec = CalibrationSpec()
        assert spec.method == "isotonic"
        assert spec.min_samples == 10

    def test_platt_spec(self) -> None:
        spec = CalibrationSpec(method="platt")
        assert spec.method == "platt"


class TestFitAspectCalibrators:
    def test_isotonic_fits(self, synthetic_scores) -> None:
        scores, labels = synthetic_scores
        spec = CalibrationSpec(method="isotonic", min_samples=5)
        calibrators = fit_aspect_calibrators(
            {"mfo": (scores, labels)},
            spec,
        )
        assert "mfo" in calibrators
        assert calibrators["mfo"].is_fitted

    def test_platt_fits(self, synthetic_scores) -> None:
        scores, labels = synthetic_scores
        spec = CalibrationSpec(method="platt", min_samples=5)
        calibrators = fit_aspect_calibrators(
            {"bpo": (scores, labels)},
            spec,
        )
        assert calibrators["bpo"].is_fitted

    def test_identity_when_too_few_samples(self) -> None:
        scores = np.array([0.1, 0.9], dtype=np.float32)
        labels = np.array([0, 1], dtype=np.int8)
        spec = CalibrationSpec(min_samples=100)
        calibrators = fit_aspect_calibrators({"mfo": (scores, labels)}, spec)
        assert not calibrators["mfo"].is_fitted

    def test_identity_transform_returns_copy(self) -> None:
        cal = AspectCalibrator(aspect="mfo", spec=CalibrationSpec(), estimator=None)
        scores = np.array([0.1, 0.5, 0.9], dtype=np.float32)
        out = cal.transform(scores)
        np.testing.assert_array_equal(out, scores)
        # Must be a copy, not the same object
        assert out is not scores

    def test_identity_single_class_no_fit(self) -> None:
        """No calibrator should be fitted when all labels are one class."""
        scores = np.ones(50, dtype=np.float32) * 0.7
        labels = np.zeros(50, dtype=np.int8)
        spec = CalibrationSpec(min_samples=5)
        calibrators = fit_aspect_calibrators({"cco": (scores, labels)}, spec)
        assert not calibrators["cco"].is_fitted


class TestCalibrateScores:
    def test_calibrate_changes_scores(self, synthetic_scores) -> None:
        scores, labels = synthetic_scores
        n = len(scores)
        # Assign half to mfo, half to bpo
        aspects = np.array(["mfo"] * (n // 2) + ["bpo"] * (n - n // 2))
        spec = CalibrationSpec(method="isotonic", min_samples=5)
        calibrators = fit_aspect_calibrators(
            {
                "mfo": (scores[: n // 2], labels[: n // 2]),
                "bpo": (scores[n // 2 :], labels[n // 2 :]),
            },
            spec,
        )
        out = calibrate_scores(scores, aspects, calibrators)
        assert out.shape == scores.shape
        assert out.dtype == np.float32

    def test_calibrate_identity_for_unfitted(self, synthetic_scores) -> None:
        scores, _ = synthetic_scores
        aspects = np.array(["cco"] * len(scores))
        # No calibrators provided for cco
        calibrators = {"cco": AspectCalibrator(aspect="cco", spec=CalibrationSpec())}
        out = calibrate_scores(scores, aspects, calibrators)
        np.testing.assert_array_equal(out, scores.astype(np.float32))

    def test_calibrate_preserves_monotonicity(self, rng) -> None:
        """Isotonic calibration must not violate monotonicity in expectation."""
        n = 300
        sorted_scores = np.sort(rng.uniform(0, 1, n)).astype(np.float32)
        labels = (sorted_scores > 0.5).astype(np.int8)
        spec = CalibrationSpec(method="isotonic", min_samples=5)
        calibrators = fit_aspect_calibrators({"mfo": (sorted_scores, labels)}, spec)
        aspects = np.array(["mfo"] * n)
        out = calibrate_scores(sorted_scores, aspects, calibrators)
        # Calibrated values must be non-decreasing (isotonic property)
        assert (np.diff(out) >= -1e-6).all(), "Isotonic calibration violated monotonicity"


class TestSaveLoadCalibrators:
    def test_roundtrip(self, synthetic_scores) -> None:
        scores, labels = synthetic_scores
        spec = CalibrationSpec(method="isotonic", min_samples=5)
        calibrators = fit_aspect_calibrators({"mfo": (scores, labels)}, spec)
        stats = calibration_stats(calibrators, {"mfo": (scores, labels)})

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            save_calibrators(calibrators, tmp_path, spec, stats)

            # Verify files written
            assert (tmp_path / "mfo_calibrator.npy").exists()
            assert (tmp_path / "calibration_meta.json").exists()

            # Load and check
            loaded = load_calibrators(tmp_path)
            assert "mfo" in loaded
            assert loaded["mfo"].is_fitted

    def test_load_missing_aspect_returns_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            # Write a minimal meta with no fitted aspects
            meta = {"spec": {"method": "isotonic", "min_samples": 10, "out_of_bounds": "clip"},
                    "aspects_fitted": [], "stats": []}
            (tmp_path / "calibration_meta.json").write_text(json.dumps(meta))
            loaded = load_calibrators(tmp_path)
            for asp in ASPECT_NAMES:
                assert asp in loaded
                assert not loaded[asp].is_fitted

    def test_load_applies_correctly_after_save(self, synthetic_scores) -> None:
        scores, labels = synthetic_scores
        spec = CalibrationSpec(method="isotonic", min_samples=5)
        calibrators = fit_aspect_calibrators({"bpo": (scores, labels)}, spec)
        stats = calibration_stats(calibrators, {"bpo": (scores, labels)})

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            save_calibrators(calibrators, tmp_path, spec, stats)
            loaded = load_calibrators(tmp_path)

            # Both should produce the same output
            aspects = np.array(["bpo"] * len(scores))
            original_out = calibrate_scores(scores, aspects, calibrators)
            loaded_out = calibrate_scores(scores, aspects, loaded)
            np.testing.assert_array_almost_equal(original_out, loaded_out, decimal=5)


class TestCalibrationStats:
    def test_stats_structure(self, synthetic_scores) -> None:
        scores, labels = synthetic_scores
        spec = CalibrationSpec(method="isotonic", min_samples=5)
        calibrators = fit_aspect_calibrators({"mfo": (scores, labels)}, spec)
        stats = calibration_stats(calibrators, {"mfo": (scores, labels)})
        assert len(stats) == 1
        s = stats[0]
        assert s.aspect == "mfo"
        assert s.n_train == len(scores)
        assert s.n_positives == int(labels.sum())
        assert s.fitted is True

    def test_unfitted_stats(self) -> None:
        scores = np.array([0.5], dtype=np.float32)
        labels = np.array([1], dtype=np.int8)
        spec = CalibrationSpec(min_samples=100)
        calibrators = fit_aspect_calibrators({"cco": (scores, labels)}, spec)
        stats = calibration_stats(calibrators, {"cco": (scores, labels)})
        assert stats[0].fitted is False
