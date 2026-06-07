"""Per-aspect score calibration for the universal booster.

Fits one calibrator per GO namespace (MFO, BPO, CCO) on the VALID window
predictions, then applies it to score arrays before final reporting.

Two calibration methods:
- ``"isotonic"``: sklearn.isotonic.IsotonicRegression (non-parametric, monotone).
  Preferred for ranking outputs where the raw score ordering is correct but the
  scale is arbitrary.
- ``"platt"``: logistic regression on a single scalar feature (Platt scaling).
  Useful when the sigmoid-shaped transformation is appropriate (e.g. GBDT
  outputs).

The calibrator is fitted PER-ASPECT on a flat array of (score, label) pairs
from the VALID window. The fitted mapping is then applied to raw booster scores
before any downstream hierarchical correction step.

Usage::

    from pathlib import Path
    import numpy as np
    from protea_reranker_lab.calibration import (
        CalibrationSpec, fit_aspect_calibrators, calibrate_scores,
        save_calibrators, load_calibrators,
    )

    spec = CalibrationSpec(method="isotonic")
    calibrators = fit_aspect_calibrators(
        scores_per_aspect={"mfo": (scores_mfo, labels_mfo)},
        spec=spec,
    )
    calibrated = calibrate_scores(raw_scores, aspects, calibrators)

Serialization uses numpy-pickle via ``np.save``/``np.load``. A JSON sidecar
records the method, coverage, and per-aspect calibration stats.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np


ASPECT_NAMES: tuple[str, ...] = ("mfo", "bpo", "cco")

CalibrationMethod = Literal["isotonic", "platt"]


@dataclass(frozen=True)
class CalibrationSpec:
    """Configuration for per-aspect score calibration.

    Attributes:
        method:       ``"isotonic"`` (default) or ``"platt"`` (sigmoid).
        min_samples:  minimum samples required to fit a calibrator for an
                      aspect; falls back to identity if fewer samples present.
        out_of_bounds: handling for ``IsotonicRegression`` when prediction is
                      outside the training range (``"clip"`` or ``"nan"``).
    """

    method: CalibrationMethod = "isotonic"
    min_samples: int = 10
    out_of_bounds: str = "clip"


@dataclass
class CalibrationStats:
    """Per-aspect calibration diagnostics.

    Attributes:
        aspect:        GO namespace abbreviation.
        method:        calibration method used.
        n_train:       number of (score, label) pairs used for fitting.
        n_positives:   positives in the calibration set.
        fitted:        whether calibration was actually fitted (False if
                       ``min_samples`` threshold not met).
        score_min:     minimum raw score in the calibration set.
        score_max:     maximum raw score in the calibration set.
    """

    aspect: str
    method: str
    n_train: int
    n_positives: int
    fitted: bool
    score_min: float
    score_max: float


class AspectCalibrator:
    """Wrapper around a fitted per-aspect calibrator.

    Exposes a ``transform(scores)`` method that maps raw booster scores
    to calibrated probabilities (or a monotone equivalent).
    """

    def __init__(
        self,
        aspect: str,
        spec: CalibrationSpec,
        estimator: object | None = None,
    ) -> None:
        self.aspect = aspect
        self.spec = spec
        self._estimator = estimator

    @property
    def is_fitted(self) -> bool:
        """True when an estimator is attached (i.e. calibration was fitted)."""
        return self._estimator is not None

    def transform(self, scores: np.ndarray) -> np.ndarray:
        """Map raw scores to calibrated scores.

        Returns the identity when no estimator was fitted.
        """
        if self._estimator is None:
            return scores.copy()
        scores_2d = scores.reshape(-1, 1)
        if self.spec.method == "platt":
            return self._estimator.predict_proba(scores_2d)[:, 1].astype(np.float32)
        # isotonic
        return self._estimator.transform(scores.astype(float)).astype(np.float32)

    def save(self, path: Path) -> None:
        """Persist the calibrator to ``path.npy`` (numpy pickle)."""
        np.save(str(path), self._estimator, allow_pickle=True)

    @classmethod
    def load(cls, path: Path, aspect: str, spec: CalibrationSpec) -> "AspectCalibrator":
        """Load a previously saved calibrator."""
        estimator = np.load(str(path), allow_pickle=True).item()
        return cls(aspect=aspect, spec=spec, estimator=estimator)


def _fit_one_calibrator(
    scores: np.ndarray,
    labels: np.ndarray,
    spec: CalibrationSpec,
) -> object | None:
    """Fit and return one sklearn estimator, or None when min_samples not met."""
    if len(scores) < spec.min_samples or len(np.unique(labels)) < 2:
        return None
    if spec.method == "isotonic":
        from sklearn.isotonic import IsotonicRegression
        ir = IsotonicRegression(out_of_bounds=spec.out_of_bounds)
        ir.fit(scores.astype(float), labels.astype(float))
        return ir
    if spec.method == "platt":
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(max_iter=1000, C=1.0)
        lr.fit(scores.reshape(-1, 1).astype(float), labels.astype(int))
        return lr
    raise ValueError(f"Unknown calibration method: {spec.method!r}")


def fit_aspect_calibrators(
    scores_per_aspect: dict[str, tuple[np.ndarray, np.ndarray]],
    spec: CalibrationSpec,
) -> dict[str, AspectCalibrator]:
    """Fit one calibrator per aspect from ``(scores, labels)`` pairs.

    Parameters
    ----------
    scores_per_aspect:
        Mapping from aspect name (``"mfo"``, ``"bpo"``, ``"cco"``) to a
        ``(scores, labels)`` tuple where both arrays are 1-D and aligned.
    spec:
        Calibration configuration.

    Returns
    -------
    Mapping from aspect name to a fitted (or identity) :class:`AspectCalibrator`.
    """
    calibrators: dict[str, AspectCalibrator] = {}
    for aspect, (scores, labels) in scores_per_aspect.items():
        estimator = _fit_one_calibrator(scores, labels, spec)
        calibrators[aspect] = AspectCalibrator(aspect=aspect, spec=spec, estimator=estimator)
    return calibrators


def calibration_stats(
    calibrators: dict[str, AspectCalibrator],
    scores_per_aspect: dict[str, tuple[np.ndarray, np.ndarray]],
) -> list[CalibrationStats]:
    """Build per-aspect diagnostics for ``run.json``."""
    stats: list[CalibrationStats] = []
    for aspect, cal in calibrators.items():
        arr = scores_per_aspect.get(aspect)
        if arr is not None:
            s, lbl = arr
            n_train = int(len(s))
            n_pos = int((lbl > 0).sum())
            s_min = float(s.min()) if n_train > 0 else 0.0
            s_max = float(s.max()) if n_train > 0 else 0.0
        else:
            n_train = 0
            n_pos = 0
            s_min = 0.0
            s_max = 0.0
        stats.append(CalibrationStats(
            aspect=aspect,
            method=cal.spec.method,
            n_train=n_train,
            n_positives=n_pos,
            fitted=cal.is_fitted,
            score_min=s_min,
            score_max=s_max,
        ))
    return sorted(stats, key=lambda s: s.aspect)


def calibrate_scores(
    scores: np.ndarray,
    aspects: np.ndarray,
    calibrators: dict[str, AspectCalibrator],
) -> np.ndarray:
    """Apply per-aspect calibration to a flat score array.

    Parameters
    ----------
    scores:
        Flat 1-D array of raw booster scores (one row per candidate pair).
    aspects:
        Flat 1-D array of aspect labels aligned to ``scores``.
    calibrators:
        Per-aspect calibrator mapping from :func:`fit_aspect_calibrators`.

    Returns
    -------
    Calibrated score array of the same shape and dtype (float32).
    """
    out = scores.astype(np.float32).copy()
    for asp, cal in calibrators.items():
        mask = aspects == asp
        if not mask.any() or not cal.is_fitted:
            continue
        out[mask] = cal.transform(scores[mask])
    return out


def save_calibrators(
    calibrators: dict[str, AspectCalibrator],
    out_dir: Path,
    spec: CalibrationSpec,
    stats: list[CalibrationStats],
) -> None:
    """Persist all calibrators + a JSON sidecar to ``out_dir``.

    Creates ``{aspect}_calibrator.npy`` per aspect and
    ``calibration_meta.json`` with spec + stats.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for asp, cal in calibrators.items():
        if cal.is_fitted:
            cal.save(out_dir / f"{asp}_calibrator.npy")
    meta = {
        "spec": asdict(spec),
        "aspects_fitted": [asp for asp, cal in calibrators.items() if cal.is_fitted],
        "stats": [asdict(s) for s in stats],
    }
    (out_dir / "calibration_meta.json").write_text(
        json.dumps(meta, indent=2)
    )


def load_calibrators(
    out_dir: Path,
    spec: CalibrationSpec | None = None,
) -> dict[str, AspectCalibrator]:
    """Load calibrators from ``out_dir``.

    Reads ``calibration_meta.json`` for the spec (if ``spec`` is not
    provided), then loads each ``{aspect}_calibrator.npy`` file.
    Missing files produce identity calibrators.
    """
    out_dir = Path(out_dir)
    meta_path = out_dir / "calibration_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        fitted_aspects = set(meta.get("aspects_fitted", []))
        if spec is None:
            spec_data = meta.get("spec", {})
            spec = CalibrationSpec(**spec_data)
    else:
        fitted_aspects = set()
        if spec is None:
            spec = CalibrationSpec()

    calibrators: dict[str, AspectCalibrator] = {}
    for asp in ASPECT_NAMES:
        npy_path = out_dir / f"{asp}_calibrator.npy"
        if asp in fitted_aspects and npy_path.exists():
            calibrators[asp] = AspectCalibrator.load(npy_path, aspect=asp, spec=spec)
        else:
            calibrators[asp] = AspectCalibrator(aspect=asp, spec=spec, estimator=None)
    return calibrators
