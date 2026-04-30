"""Per-protein paired bootstrap of Fmax for the v9 study (Phase 3).

For one cell: take the trainer's ``predictions.parquet`` (reranker scores) +
re-stage the eval set for a single baseline column (default ``vote_count``,
the natural KNN-only score), then resample proteins with replacement to build
a 95% CI on Fmax for each scorer and on the paired difference.

The baseline is computed by :func:`stage_eval_only` rather than re-running
the trainer, so this module never needs the booster nor the train side.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .evaluate import fmax_per_protein_group
from .staging import stage_eval_only


@dataclass
class BootstrapResult:
    cell: str
    n_proteins: int
    n_rows: int
    n_iter: int
    seed: int
    baseline_column: str

    fmax_v9: float
    fmax_v9_ci_lo: float
    fmax_v9_ci_hi: float
    fmax_baseline: float
    fmax_baseline_ci_lo: float
    fmax_baseline_ci_hi: float
    paired_diff_mean: float
    paired_diff_ci_lo: float
    paired_diff_ci_hi: float
    p_value_one_sided: float

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def _read_predictions(predictions_parquet: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    table = pq.read_table(str(predictions_parquet),
                          columns=["protein_accession", "label", "score"])
    proteins = table.column("protein_accession").to_numpy(zero_copy_only=False)
    labels = table.column("label").to_numpy(zero_copy_only=False).astype(np.int8, copy=False)
    scores = table.column("score").to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
    return proteins, labels, scores


def _read_baseline(staged_eval_split, baseline_column: str) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for bp in staged_eval_split.bucket_paths:
        table = pq.read_table(str(bp), columns=[baseline_column])
        chunks.append(table.column(baseline_column).to_numpy(zero_copy_only=False))
    return np.concatenate(chunks).astype(np.float32, copy=False)


def _groups_from_proteins(proteins: np.ndarray) -> np.ndarray:
    edges = np.flatnonzero(np.concatenate(([True], proteins[1:] != proteins[:-1])))
    sizes = np.diff(np.concatenate((edges, [len(proteins)]))).astype(np.int32)
    return sizes


def bootstrap_paired_fmax(
    *,
    cell: str,
    predictions_parquet: Path,
    source_eval_parquet: Path,
    eval_snapshot_pair: str | None,
    baseline_column: str = "vote_count",
    n_iter: int = 1000,
    seed: int = 42,
    workdir: Path,
    higher_is_better: bool = True,
) -> BootstrapResult:
    """Run per-protein paired bootstrap on a winner run + KNN-only baseline.

    The baseline is read from a freshly-staged eval (deterministic row order
    matching predictions.parquet given identical staging parameters). For
    distance-style baselines, set ``higher_is_better=False`` and the column
    will be negated before scoring.
    """
    cat, asp = cell.lower().split("-", 1)
    proteins, labels, scores_v9 = _read_predictions(predictions_parquet)

    workdir.mkdir(parents=True, exist_ok=True)
    eval_split = stage_eval_only(
        source_eval_parquet=source_eval_parquet,
        cell=(cat, asp),
        feature_cols=[baseline_column],
        categorical_cols=[],
        out_dir=workdir / "eval_baseline_stage",
        eval_snapshot_pair=eval_snapshot_pair,
    )

    baseline_raw = _read_baseline(eval_split, baseline_column)
    if baseline_raw.shape[0] != scores_v9.shape[0]:
        raise RuntimeError(
            f"row mismatch: predictions has {scores_v9.shape[0]} rows but "
            f"re-staged eval has {baseline_raw.shape[0]} — bucket-sort order drifted"
        )
    scores_baseline = baseline_raw if higher_is_better else -baseline_raw

    groups = _groups_from_proteins(proteins)
    n_groups = groups.size

    fmax_v9_point = fmax_per_protein_group(scores_v9, labels, groups)
    fmax_baseline_point = fmax_per_protein_group(scores_baseline, labels, groups)

    edges = np.empty(n_groups + 1, dtype=np.int64)
    edges[0] = 0
    np.cumsum(groups, out=edges[1:])
    rng = np.random.default_rng(seed)

    fmax_v9_iters = np.empty(n_iter, dtype=np.float32)
    fmax_baseline_iters = np.empty(n_iter, dtype=np.float32)
    diff_iters = np.empty(n_iter, dtype=np.float32)

    for k in range(n_iter):
        idx = rng.integers(0, n_groups, size=n_groups)
        sampled_starts = edges[idx]
        sampled_sizes = groups[idx]
        n_sampled = int(sampled_sizes.sum())
        if n_sampled == 0:
            fmax_v9_iters[k] = 0.0
            fmax_baseline_iters[k] = 0.0
            diff_iters[k] = 0.0
            continue
        sample_offsets = np.empty(n_sampled, dtype=np.int64)
        cur = 0
        for s, sz in zip(sampled_starts, sampled_sizes):
            sample_offsets[cur:cur + sz] = np.arange(s, s + sz)
            cur += sz
        s_v9 = scores_v9[sample_offsets]
        s_bl = scores_baseline[sample_offsets]
        s_lab = labels[sample_offsets]
        fmax_v9_iters[k] = fmax_per_protein_group(s_v9, s_lab, sampled_sizes)
        fmax_baseline_iters[k] = fmax_per_protein_group(s_bl, s_lab, sampled_sizes)
        diff_iters[k] = fmax_v9_iters[k] - fmax_baseline_iters[k]

    def _ci(arr: np.ndarray) -> tuple[float, float]:
        return float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))

    v9_lo, v9_hi = _ci(fmax_v9_iters)
    bl_lo, bl_hi = _ci(fmax_baseline_iters)
    d_lo, d_hi = _ci(diff_iters)
    p_value_one_sided = float((diff_iters <= 0).mean())

    return BootstrapResult(
        cell=cell, n_proteins=int(n_groups), n_rows=int(scores_v9.size),
        n_iter=n_iter, seed=seed, baseline_column=baseline_column,
        fmax_v9=fmax_v9_point, fmax_v9_ci_lo=v9_lo, fmax_v9_ci_hi=v9_hi,
        fmax_baseline=fmax_baseline_point,
        fmax_baseline_ci_lo=bl_lo, fmax_baseline_ci_hi=bl_hi,
        paired_diff_mean=float(diff_iters.mean()),
        paired_diff_ci_lo=d_lo, paired_diff_ci_hi=d_hi,
        p_value_one_sided=p_value_one_sided,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cell", required=True)
    p.add_argument("--predictions", required=True, type=Path)
    p.add_argument("--source-eval", required=True, type=Path)
    p.add_argument("--eval-snapshot-pair", default=None)
    p.add_argument("--baseline-column", default="vote_count")
    p.add_argument("--baseline-direction",
                   choices=["higher_is_better", "lower_is_better"],
                   default="higher_is_better")
    p.add_argument("--n-iter", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workdir", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path,
                   help="path to write the bootstrap result JSON")
    args = p.parse_args(argv)

    result = bootstrap_paired_fmax(
        cell=args.cell,
        predictions_parquet=args.predictions,
        source_eval_parquet=args.source_eval,
        eval_snapshot_pair=args.eval_snapshot_pair,
        baseline_column=args.baseline_column,
        higher_is_better=args.baseline_direction == "higher_is_better",
        n_iter=args.n_iter,
        seed=args.seed,
        workdir=args.workdir,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(result.to_json())
    print(result.to_json())
    return 0


if __name__ == "__main__":
    sys.exit(main())
