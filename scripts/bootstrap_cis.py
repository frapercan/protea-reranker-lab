#!/usr/bin/env python
"""Group-by-axis aggregation of per-cell bootstrap CIs (FARM-EXP.3).

Reads a directory of bootstrap run records (one JSON per cell-run),
groups them by user-selected axis names from the FARM-EXP.2 canonical
catalog vocabulary, and emits one CSV plus one paired-CI plot per
grouping under ``outputs/bootstrap_cis/``.

Run-record format (forward-compatible with FARM-EXP.5 writers)::

    {
      "run_id": "<unique run identifier>",
      "axis": {
        "plm": "esm2_3b",
        "k": 5,
        "reranker": "lgbm.per_cell_9",
        "features": "v6",
        "eval_set": "bench-v1-K5-filtered",
        "propagation": "tpr_pred",
        "ensemble": "none"
      },
      "shortid": "029587a6fc40",
      "fmax_samples": [0.41, 0.42, ...],
      "fmax_point": 0.4203,
      "cell": "nk-bpo",
      "baseline_fmax_samples": [0.04, ...],
      "baseline_fmax_point": 0.048
    }

``fmax_samples`` is the per-iter bootstrap distribution produced by
:func:`protea_reranker_lab.bootstrap.bootstrap_paired_fmax` (the array
the writer is expected to persist alongside the aggregate
``BootstrapResult``). Records missing the ``axis`` block are grouped
under the ``axis_unknown`` bucket unless ``--strict`` is passed.

CI math: percentile bootstrap. For a group with M runs each carrying
S samples, the script concatenates the M*S pooled samples, then
resamples that pooled vector with replacement ``--n-iter`` times
(default 1000) and takes the 2.5 / 97.5 quantiles of the per-iter
means. This is a real resampling computation; no normal-approx
shortcut. The same ``--seed`` produces identical CI bounds (verified
by the test fixture).

Plot: paired-CI dot-and-whisker. One row per run inside the group,
showing the candidate fmax CI (and baseline fmax CI when present).
This preserves the pairing structure that the per-protein bootstrap
already established.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Canonical axis vocabulary from FARM-EXP.2's transversal catalog.
# Mirrors experiments/_catalog/transversal.yaml stanza keys. Keep in
# sync with scripts/build_study_specs.py if the catalog grows axes.
CANONICAL_AXES: tuple[str, ...] = (
    "plm",
    "k",
    "reranker",
    "features",
    "eval_set",
    "propagation",
    "ensemble",
)

GROUP_ALL_SENTINEL = "all"
AXIS_UNKNOWN_LABEL = "axis_unknown"


@dataclass(frozen=True)
class RunRecord:
    """Forward-compat parse of a bootstrap run-record JSON."""

    run_id: str
    axis: dict[str, Any]
    fmax_samples: np.ndarray
    fmax_point: float
    baseline_fmax_samples: np.ndarray | None
    baseline_fmax_point: float | None
    cell: str | None
    shortid: str | None

    @classmethod
    def from_dict(cls, payload: dict[str, Any], source: str) -> "RunRecord":
        run_id = str(payload.get("run_id") or source)
        axis = dict(payload.get("axis") or {})
        samples = payload.get("fmax_samples")
        if samples is None:
            raise ValueError(f"{source}: missing required field 'fmax_samples'")
        arr = np.asarray(samples, dtype=np.float64)
        if arr.ndim != 1 or arr.size == 0:
            raise ValueError(
                f"{source}: fmax_samples must be a non-empty 1-D array"
            )
        point = payload.get("fmax_point")
        if point is None:
            point = float(arr.mean())
        baseline_samples = payload.get("baseline_fmax_samples")
        bl_arr: np.ndarray | None = None
        if baseline_samples is not None:
            bl_arr = np.asarray(baseline_samples, dtype=np.float64)
            if bl_arr.shape != arr.shape:
                raise ValueError(
                    f"{source}: baseline_fmax_samples shape "
                    f"{bl_arr.shape} != fmax_samples shape {arr.shape}"
                )
        bl_point = payload.get("baseline_fmax_point")
        if bl_point is None and bl_arr is not None:
            bl_point = float(bl_arr.mean())
        return cls(
            run_id=run_id,
            axis=axis,
            fmax_samples=arr,
            fmax_point=float(point),
            baseline_fmax_samples=bl_arr,
            baseline_fmax_point=float(bl_point) if bl_point is not None else None,
            cell=payload.get("cell"),
            shortid=payload.get("shortid"),
        )


# ----------------------------------------------------------- parsing


def parse_group_by(raw: str | None) -> tuple[str, ...]:
    """Parse the --group-by CLI value.

    Returns the empty tuple for the sentinel ``all`` (or empty
    string), meaning "no grouping; emit a single flat aggregation".
    Otherwise returns the validated axis subset in the order the
    user passed.
    """
    if raw is None:
        return ()
    raw = raw.strip()
    if raw == "" or raw == GROUP_ALL_SENTINEL:
        return ()
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        return ()
    unknown = [p for p in parts if p not in CANONICAL_AXES]
    if unknown:
        raise ValueError(
            f"unknown axis name(s): {unknown}. Allowed axes "
            f"(from FARM-EXP.2 catalog): {list(CANONICAL_AXES)}"
        )
    # de-dup while preserving order
    seen: set[str] = set()
    deduped: list[str] = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            deduped.append(p)
    return tuple(deduped)


def load_runs(runs_dir: Path, *, strict: bool = False) -> list[RunRecord]:
    """Load every ``*.json`` under ``runs_dir`` as a RunRecord.

    A record whose top-level shape does not parse cleanly is skipped
    with a warning unless ``strict=True``.
    """
    runs_dir = Path(runs_dir)
    if not runs_dir.is_dir():
        raise FileNotFoundError(f"runs dir does not exist: {runs_dir}")
    records: list[RunRecord] = []
    for path in sorted(runs_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            if strict:
                raise
            print(f"[warn] skipping {path.name}: invalid JSON ({exc})",
                  file=sys.stderr)
            continue
        try:
            records.append(RunRecord.from_dict(payload, source=path.name))
        except ValueError as exc:
            if strict:
                raise
            print(f"[warn] skipping {path.name}: {exc}", file=sys.stderr)
            continue
    return records


# ----------------------------------------------------------- grouping


def group_signature(axes: tuple[str, ...], values: tuple[Any, ...]) -> str:
    """Filename-safe signature for a (axes, values) grouping key."""
    if not axes:
        return f"group={GROUP_ALL_SENTINEL}"
    pieces = [f"{a}={v}" for a, v in zip(axes, values)]
    return "group=" + "__".join(pieces)


def _axis_value(record: RunRecord, axis: str, *, strict: bool) -> Any:
    """Extract an axis value from a record's axis block."""
    val = record.axis.get(axis)
    if val is None:
        if strict:
            raise KeyError(
                f"run {record.run_id} missing axis '{axis}'"
            )
        return AXIS_UNKNOWN_LABEL
    return val


def group_runs(
    records: Iterable[RunRecord],
    axes: tuple[str, ...],
    *,
    strict: bool = False,
) -> dict[tuple[Any, ...], list[RunRecord]]:
    """Bucket records by their axis-value tuple under ``axes``.

    Returns a dict mapping the tuple of axis values (in ``axes``
    order) to the list of records in that bucket. When ``axes`` is
    empty, returns a single bucket keyed by the empty tuple holding
    every record.
    """
    buckets: dict[tuple[Any, ...], list[RunRecord]] = {}
    for r in records:
        if not axes:
            key: tuple[Any, ...] = ()
        else:
            key = tuple(_axis_value(r, a, strict=strict) for a in axes)
        buckets.setdefault(key, []).append(r)
    return buckets


# ----------------------------------------------------------- CI math


@dataclass(frozen=True)
class GroupCI:
    n_runs: int
    n_samples_total: int
    n_iter: int
    seed: int
    fmax_mean: float
    fmax_ci_lo: float
    fmax_ci_hi: float
    paired_diff_mean: float | None
    paired_diff_ci_lo: float | None
    paired_diff_ci_hi: float | None


def bootstrap_group_ci(
    records: list[RunRecord],
    *,
    n_iter: int,
    seed: int,
) -> GroupCI:
    """Bootstrap CI on the group's pooled fmax samples.

    For M runs each with S samples, pool to an M*S vector and
    resample with replacement ``n_iter`` times, taking the mean of
    each draw. The 2.5 / 97.5 quantiles of that distribution are the
    group CI on the mean fmax. The paired difference statistic is
    computed analogously when every record carries baseline_samples.
    """
    if not records:
        raise ValueError("bootstrap_group_ci called with empty group")
    pooled_fmax = np.concatenate([r.fmax_samples for r in records])
    rng = np.random.default_rng(seed)
    n = pooled_fmax.size
    # Draw indices of shape (n_iter, n) and take row means in one shot.
    idx = rng.integers(0, n, size=(n_iter, n))
    boot_means = pooled_fmax[idx].mean(axis=1)
    fmax_lo = float(np.quantile(boot_means, 0.025))
    fmax_hi = float(np.quantile(boot_means, 0.975))
    fmax_mean = float(pooled_fmax.mean())

    paired_diff_mean: float | None = None
    paired_lo: float | None = None
    paired_hi: float | None = None
    if all(r.baseline_fmax_samples is not None for r in records):
        pooled_baseline = np.concatenate(
            [r.baseline_fmax_samples for r in records]
        )
        # Pair element-wise via shared resample indices.
        diff_means = (
            pooled_fmax[idx].mean(axis=1)
            - pooled_baseline[idx].mean(axis=1)
        )
        paired_diff_mean = float(
            (pooled_fmax - pooled_baseline).mean()
        )
        paired_lo = float(np.quantile(diff_means, 0.025))
        paired_hi = float(np.quantile(diff_means, 0.975))

    return GroupCI(
        n_runs=len(records),
        n_samples_total=int(n),
        n_iter=int(n_iter),
        seed=int(seed),
        fmax_mean=fmax_mean,
        fmax_ci_lo=fmax_lo,
        fmax_ci_hi=fmax_hi,
        paired_diff_mean=paired_diff_mean,
        paired_diff_ci_lo=paired_lo,
        paired_diff_ci_hi=paired_hi,
    )


# ----------------------------------------------------------- emit


CSV_COLUMNS_BASE = [
    "n_runs",
    "n_samples_total",
    "n_iter",
    "seed",
    "fmax_mean",
    "fmax_ci_lo",
    "fmax_ci_hi",
    "paired_diff_mean",
    "paired_diff_ci_lo",
    "paired_diff_ci_hi",
    "run_ids",
]


def write_csv(
    out_path: Path,
    axes: tuple[str, ...],
    rows: list[tuple[tuple[Any, ...], GroupCI, list[str]]],
) -> None:
    """Write one CSV row per group to ``out_path``."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(axes) + CSV_COLUMNS_BASE
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for key, ci, run_ids in rows:
            row: dict[str, Any] = {a: v for a, v in zip(axes, key)}
            row.update(
                {
                    "n_runs": ci.n_runs,
                    "n_samples_total": ci.n_samples_total,
                    "n_iter": ci.n_iter,
                    "seed": ci.seed,
                    "fmax_mean": ci.fmax_mean,
                    "fmax_ci_lo": ci.fmax_ci_lo,
                    "fmax_ci_hi": ci.fmax_ci_hi,
                    "paired_diff_mean": (
                        ci.paired_diff_mean
                        if ci.paired_diff_mean is not None
                        else ""
                    ),
                    "paired_diff_ci_lo": (
                        ci.paired_diff_ci_lo
                        if ci.paired_diff_ci_lo is not None
                        else ""
                    ),
                    "paired_diff_ci_hi": (
                        ci.paired_diff_ci_hi
                        if ci.paired_diff_ci_hi is not None
                        else ""
                    ),
                    "run_ids": "|".join(run_ids),
                }
            )
            writer.writerow(row)


def plot_paired_ci(
    out_path: Path,
    title: str,
    records: list[RunRecord],
) -> None:
    """Render a paired-CI dot-and-whisker plot for one group.

    One row per run. Candidate (blue) and baseline (grey) CIs are
    drawn side by side. If no record has baseline samples, only the
    candidate row is shown.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(records)
    fig_h = max(2.0, 0.4 * n + 1.0)
    fig, ax = plt.subplots(figsize=(7.0, fig_h))
    y_positions = np.arange(n)
    has_baseline = any(r.baseline_fmax_samples is not None for r in records)
    offset = 0.15 if has_baseline else 0.0

    for i, r in enumerate(records):
        lo = float(np.quantile(r.fmax_samples, 0.025))
        hi = float(np.quantile(r.fmax_samples, 0.975))
        y = float(y_positions[i]) + offset
        ax.hlines(y, lo, hi, color="C0", linewidth=2.0)
        ax.scatter([r.fmax_point], [y], color="C0", zorder=3, s=24)
        if r.baseline_fmax_samples is not None:
            bl_lo = float(np.quantile(r.baseline_fmax_samples, 0.025))
            bl_hi = float(np.quantile(r.baseline_fmax_samples, 0.975))
            yb = float(y_positions[i]) - offset
            ax.hlines(yb, bl_lo, bl_hi, color="grey", linewidth=2.0)
            ax.scatter(
                [r.baseline_fmax_point], [yb], color="grey", zorder=3, s=24,
            )

    labels = [r.cell or r.run_id for r in records]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("fmax (95% percentile CI)")
    ax.set_title(title, fontsize=10)
    if has_baseline:
        from matplotlib.lines import Line2D
        legend_handles = [
            Line2D([0], [0], color="C0", lw=2, label="candidate"),
            Line2D([0], [0], color="grey", lw=2, label="baseline"),
        ]
        ax.legend(handles=legend_handles, loc="best", fontsize=8)
    ax.grid(True, axis="x", linestyle=":", alpha=0.5)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ----------------------------------------------------------- orchestrator


def run(
    *,
    runs_dir: Path,
    out_dir: Path,
    group_by: tuple[str, ...],
    n_iter: int,
    seed: int,
    strict: bool,
    emit_plot: bool,
) -> list[Path]:
    """Execute the grouping + emit pipeline.

    Returns the list of CSV paths written (one per grouping bucket).
    """
    records = load_runs(runs_dir, strict=strict)
    if not records:
        raise RuntimeError(f"no usable run records under {runs_dir}")

    buckets = group_runs(records, group_by, strict=strict)
    written: list[Path] = []
    for key in sorted(buckets, key=lambda k: tuple(str(v) for v in k)):
        bucket_records = buckets[key]
        ci = bootstrap_group_ci(bucket_records, n_iter=n_iter, seed=seed)
        sig = group_signature(group_by, key)
        csv_path = out_dir / f"{sig}.csv"
        write_csv(
            csv_path,
            group_by,
            [(key, ci, [r.run_id for r in bucket_records])],
        )
        written.append(csv_path)
        if emit_plot:
            plot_paired_ci(
                out_dir / f"{sig}.png",
                title=sig,
                records=bucket_records,
            )
    return written


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Bootstrap CI aggregation grouped by axis "
        "(FARM-EXP.3).",
    )
    p.add_argument(
        "--runs-dir",
        type=Path,
        required=True,
        help="directory of bootstrap run-record JSON files",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/bootstrap_cis"),
    )
    p.add_argument(
        "--group-by",
        default="",
        help=(
            "comma-separated subset of canonical axes: "
            f"{', '.join(CANONICAL_AXES)}. Use 'all' or '' for a "
            "single flat aggregation."
        ),
    )
    p.add_argument("--n-iter", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--strict",
        action="store_true",
        help="error out on records with missing axis or invalid JSON",
    )
    p.add_argument(
        "--no-plot",
        action="store_true",
        help="skip png emission; only write CSVs",
    )
    args = p.parse_args(argv)

    try:
        group_by = parse_group_by(args.group_by)
    except ValueError as exc:
        print(f"[err] {exc}", file=sys.stderr)
        return 2

    written = run(
        runs_dir=args.runs_dir,
        out_dir=args.out_dir,
        group_by=group_by,
        n_iter=args.n_iter,
        seed=args.seed,
        strict=args.strict,
        emit_plot=not args.no_plot,
    )
    print(
        f"[done] wrote {len(written)} grouping CSV(s) under {args.out_dir}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
