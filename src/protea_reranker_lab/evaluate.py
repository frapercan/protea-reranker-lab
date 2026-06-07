"""Per-cell CAFA Fmax on numpy arrays + IA-weighted f_micro_w evaluator.

Mirrors PROTEA's ``run_cafa_evaluation`` fmax logic but operates on flat
numpy arrays plus a contiguous ``group_sizes`` vector. No pandas, no
DataFrame materialisation. Used after :func:`predict_streaming`.

The new :func:`eval_f_micro_w` function is the canonical selection metric
for all downstream gating: it wraps the cafaeval subprocess driver lifted
from ``scripts/farm_exp_15_knn_226_227.py`` and resolves the IA path via
the band-registry bridge so the IA is never hardcoded.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypedDict

import numpy as np

_DEFAULT_PROTEA_PYTHON = (
    "/home/frapercan/Thesis2/repositories/PROTEA/.venv/bin/python"
)


@dataclass(frozen=True)
class PredictionArrays:
    """Flat row-aligned arrays for one prediction cell.

    All arrays must share the same length (one row per protein-term pair).
    ``proteins`` and ``go_terms`` are string arrays; ``scores`` and
    ``labels`` are float/int arrays aligned to the same rows.
    """

    proteins: np.ndarray
    go_terms: np.ndarray
    scores: np.ndarray
    labels: np.ndarray


@dataclass(frozen=True)
class BandArtifacts:
    """Resolved (OBO path, IA path) pair for one evaluation band.

    Construct directly or via
    :func:`~protea_reranker_lab.band_registry_bridge.resolve_band_artifacts`,
    which returns a ``(obo_path, ia_path)`` tuple that can be unpacked as
    ``BandArtifacts(*resolve_band_artifacts("v227"))``.
    """

    obo_path: Path
    ia_path: Path


@dataclass(frozen=True)
class EvalOptions:
    """Cafaeval subprocess options bundled for :func:`eval_f_micro_w`.

    All fields are optional. ``protea_python`` defaults to the
    ``PROTEA_PYTHON`` environment variable, then the PROTEA venv default.
    ``work_dir`` pins the temporary working directory (no cleanup);
    leave as ``None`` to let the function create a throwaway temp dir.
    """

    protea_python: Path | None = field(default=None)
    timeout: int = field(default=900)
    work_dir: Path | None = field(default=None)

    def resolved_python(self) -> Path:
        """Return the effective Python interpreter path."""
        if self.protea_python is not None:
            return self.protea_python
        env_py = os.environ.get("PROTEA_PYTHON")
        if env_py:
            return Path(env_py)
        return Path(_DEFAULT_PROTEA_PYTHON)


def fmax_per_protein_group(
    scores: np.ndarray,
    labels: np.ndarray,
    group_sizes: np.ndarray,
    *,
    n_thresholds: int = 101,
) -> float:
    """Protein-averaged Fmax.

    Inputs are flat row-aligned arrays (rows sorted by protein, contiguous
    per protein) and ``group_sizes`` gives the number of rows per protein.
    """
    if scores.size == 0 or group_sizes.size == 0:
        return 0.0

    edges = np.empty(group_sizes.size + 1, dtype=np.int64)
    edges[0] = 0
    np.cumsum(group_sizes, out=edges[1:])

    lo, hi = float(scores.min()), float(scores.max())
    thresholds = np.array([lo]) if hi <= lo else np.linspace(lo, hi, n_thresholds)

    starts = edges[:-1]

    pos_mask = labels > 0
    n_pos_per_group = np.add.reduceat(pos_mask.astype(np.int64), starts)
    has_pos = n_pos_per_group > 0

    best = 0.0
    for t in thresholds:
        pred = scores >= t
        n_pred_per_group = np.add.reduceat(pred.astype(np.int64), starts)
        tp_per_group = np.add.reduceat((pred & pos_mask).astype(np.int64), starts)

        with np.errstate(divide="ignore", invalid="ignore"):
            precisions = np.where(
                n_pred_per_group > 0,
                tp_per_group / np.maximum(n_pred_per_group, 1),
                0.0,
            )
            recalls = np.where(
                has_pos,
                tp_per_group / np.maximum(n_pos_per_group, 1),
                0.0,
            )

        m_pred = n_pred_per_group > 0
        if not m_pred.any():
            continue
        p = float(precisions[m_pred].mean()) if m_pred.any() else 0.0
        r = float(recalls[has_pos].mean()) if has_pos.any() else 0.0
        if p + r > 0:
            f = 2 * p * r / (p + r)
            if f > best:
                best = f
    return float(best)


class FMicroWResult(TypedDict, total=False):
    """Per-namespace metrics returned by :func:`eval_f_micro_w`."""

    f_micro_w: float | None
    f_micro: float | None
    f_w: float | None
    fmax: float | None


# cafaeval subprocess driver source (lifted from farm_exp_15 verbatim so
# both the script and this function are byte-identical in their cafaeval
# invocation; prop/norm/no_orphans/weighted_only are the contract).
_CAFAEVAL_DRIVER_SRC = '''
import json, signal
from cafaeval.evaluation import cafa_eval
signal.signal(signal.SIGTERM, signal.SIG_DFL)
signal.signal(signal.SIGINT, signal.SIG_DFL)
df, dfs_best = cafa_eval(
    "{obo}", "{pred_dir}", "{gt}",
    ia="{ia}",
    prop="fill", norm="cafa", no_orphans=True,
    max_terms=500, th_step=0.001, n_cpu=1, weighted_only=False,
)
out = {{}}
for kind, df_best in dfs_best.items():
    out[kind] = df_best.reset_index().to_dict(orient="records")
with open("{out_json}", "w") as f:
    json.dump(out, f, indent=2, default=str)
'''

_ASPECT_TO_NS = {
    "bpo": "biological_process",
    "mfo": "molecular_function",
    "cco": "cellular_component",
}


def _run_cafaeval_subprocess(
    pred_tsv: Path,
    gt_tsv: Path,
    artifacts: BandArtifacts,
    options: EvalOptions,
) -> dict:
    """Stage files, invoke the cafaeval driver subprocess, return raw JSON.

    Creates a temporary directory, copies ``pred_tsv`` into the per-set
    ``pred_dir/`` subdirectory cafaeval requires, writes a driver script,
    and runs it under ``options.resolved_python()``.  Raises
    ``RuntimeError`` on non-zero exit.
    """
    with tempfile.TemporaryDirectory(prefix="lab_eval_") as tmp_str:
        tmp = Path(tmp_str)
        pred_dir = tmp / "pred_dir"
        pred_dir.mkdir()
        # cafaeval expects exactly one TSV per prediction set in pred_dir.
        (pred_dir / "predictions.tsv").write_bytes(pred_tsv.read_bytes())
        out_json = tmp / "out.json"
        driver = tmp / "_driver.py"
        driver.write_text(
            _CAFAEVAL_DRIVER_SRC.format(
                obo=str(artifacts.obo_path),
                pred_dir=str(pred_dir),
                gt=str(gt_tsv),
                ia=str(artifacts.ia_path),
                out_json=str(out_json),
            )
        )
        proc = subprocess.run(
            [str(options.resolved_python()), str(driver)],
            capture_output=True,
            text=True,
            timeout=options.timeout,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"cafaeval subprocess failed (rc={proc.returncode}):\n"
                f"{proc.stderr[-600:]}"
            )
        return json.loads(out_json.read_text())  # type: ignore[no-any-return]


def _pick_metric(
    raw: dict, namespace: str, kind: str, col: str
) -> float | None:
    """Extract a scalar metric for ``namespace`` from cafaeval raw output."""
    for rec in raw.get(kind, []):
        ns = rec.get("ns") or rec.get("namespace") or ""
        if ns == namespace and rec.get(col) is not None:
            try:
                return float(rec[col])
            except (TypeError, ValueError):
                return None
    return None


def eval_f_micro_w(
    pred_tsv: Path,
    gt_tsv: Path,
    artifacts: BandArtifacts,
    namespace: str,
    *,
    options: EvalOptions | None = None,
) -> FMicroWResult:
    """Run cafaeval ia= and return per-namespace IA-weighted metrics.

    This is the canonical selection metric for all downstream gating,
    replacing :func:`fmax_per_protein_group` for band-aware evaluation.

    Parameters
    ----------
    pred_tsv:
        CAFA-format prediction file (protein TAB go TAB score, no header).
    gt_tsv:
        Ground-truth file (protein TAB go, positives only, no header).
    artifacts:
        Resolved ``BandArtifacts`` (obo_path, ia_path) for the evaluation
        band.  Obtain via
        ``BandArtifacts(*resolve_band_artifacts("v227"))``.
    namespace:
        One of ``"molecular_function"``, ``"biological_process"``,
        ``"cellular_component"`` -- the namespace to extract metrics for.
    options:
        ``EvalOptions`` controlling the Python interpreter and timeout.
        Defaults to ``EvalOptions()`` when ``None``.

    Returns
    -------
    FMicroWResult with keys ``f_micro_w``, ``f_micro``, ``f_w``, ``fmax``
    (values may be ``None`` if cafaeval produced no output for the
    requested namespace).
    """
    opts = options if options is not None else EvalOptions()
    raw = _run_cafaeval_subprocess(pred_tsv, gt_tsv, artifacts, opts)
    return FMicroWResult(
        f_micro_w=_pick_metric(raw, namespace, "f_micro_w", "f_micro_w"),
        f_micro=_pick_metric(raw, namespace, "f_micro", "f_micro"),
        f_w=_pick_metric(raw, namespace, "f_w", "f_w"),
        fmax=_pick_metric(raw, namespace, "f", "f"),
    )


def eval_f_micro_w_from_arrays(
    arrays: PredictionArrays,
    artifacts: BandArtifacts,
    aspect: str,
    *,
    options: EvalOptions | None = None,
) -> FMicroWResult:
    """Convenience wrapper: build TSVs from arrays and call :func:`eval_f_micro_w`.

    ``aspect`` is one of ``"mfo"``, ``"bpo"``, ``"cco"``; the namespace is
    looked up via the ``_ASPECT_TO_NS`` table.  Pass an :class:`EvalOptions`
    instance (with ``work_dir`` set) to pin the working directory.
    """
    namespace = _ASPECT_TO_NS.get(aspect)
    if namespace is None:
        raise ValueError(
            f"Unknown aspect {aspect!r}; must be one of {list(_ASPECT_TO_NS)}"
        )

    opts = options if options is not None else EvalOptions()
    ctx = (
        tempfile.TemporaryDirectory(prefix="lab_eval_arrays_")
        if opts.work_dir is None
        else _NullContext(opts.work_dir)
    )
    with ctx as tmp_str:
        tmp = Path(tmp_str)
        tmp.mkdir(parents=True, exist_ok=True)
        pred_tsv = tmp / "pred.tsv"
        gt_tsv = tmp / "gt.tsv"

        pos = arrays.labels > 0
        with pred_tsv.open("w") as fh:
            for p, g, s in zip(arrays.proteins, arrays.go_terms, arrays.scores):
                fh.write(f"{p}\t{g}\t{float(s):.6f}\n")
        with gt_tsv.open("w") as fh:
            for p, g in zip(arrays.proteins[pos], arrays.go_terms[pos]):
                fh.write(f"{p}\t{g}\n")

        return eval_f_micro_w(pred_tsv, gt_tsv, artifacts, namespace, options=opts)


class _NullContext:
    """Trivial context manager that yields a fixed path (no cleanup)."""

    def __init__(self, path: Path) -> None:
        self._path = str(path)

    def __enter__(self) -> str:
        return self._path

    def __exit__(self, *_: object) -> None:
        pass


def fmax_binary(
    y_true: np.ndarray, y_score: np.ndarray, n_thresholds: int = 101,
) -> float:
    if y_true.size == 0:
        return 0.0
    thresholds = np.linspace(0.0, 1.0, n_thresholds)
    pos = y_true > 0
    best = 0.0
    for t in thresholds:
        pred = y_score >= t
        tp = int((pred & pos).sum())
        if tp == 0:
            continue
        fp = int((pred & ~pos).sum())
        fn = int((~pred & pos).sum())
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        if p + r > 0:
            f = 2 * p * r / (p + r)
            if f > best:
                best = f
    return float(best)
