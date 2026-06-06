"""Turn an :class:`ExperimentSpec` into results — with full traceability.

Streaming end-to-end: source parquet → :mod:`staging` (filter + cell + split
+ cat-encode + bucket-sort) → :class:`ParquetFeatureSequence` → LightGBM
(``free_raw_data=True``) → batched ``predict_streaming`` → numpy fmax.

Artefacts written to ``output_dir`` on every launch (``run.json`` and
``spec.yaml`` are written *at start* so a crash still leaves a trace):

- ``spec.yaml``          — the ExperimentSpec that produced this run
- ``run.json``           — run_id, status, timings, git sha, resolved
                           hparams, dataset lineage (spec/schema shas),
                           features used, metrics, feature_importance
- ``model.txt``          — LightGBM booster (only on success)
- ``predictions.parquet``— eval rows + score column (only on success)
- ``staging/…``          — per-cell sorted bucket parquets + labels/groups npy
"""

from __future__ import annotations

import dataclasses
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .builder import build_dataset
from protea_contracts import (
    CATEGORICAL_FEATURES,
    FEATURE_FAMILIES,
    NUMERIC_FEATURES,
)
from .evaluate import fmax_per_protein_group
from .experiment import ExperimentSpec, ModelSpec, TrainingSpec
from .ia_weighting import ia_weights, load_ia_table
from .reranker import (
    SplitArrays,
    TrainConfig,
    fit,
    predict_streaming,
)
from .schemas import ManifestV1
from .sequences import ParquetFeatureSequence
from .staging import StagePlan, StageResult, stage_for_training


_TRAINING_FIELDS: frozenset[str] = frozenset({
    "cell", "val_strategy", "val_fraction",
    "val_holdout_snapshot", "neg_pos_ratio", "seed",
})


def resolve_dataset(spec: ExperimentSpec, *, datasets_root: str | Path = "datasets") -> Path:
    if spec.dataset.manifest is not None:
        return Path(spec.dataset.manifest)

    ds_spec = spec.dataset.spec
    assert ds_spec is not None
    out_dir = Path(datasets_root) / ds_spec.name
    manifest_path = out_dir / "manifest.json"

    if manifest_path.exists():
        existing = ManifestV1.load(manifest_path)
        if existing.spec_hash == ds_spec.hash():
            return manifest_path

    build_dataset(ds_spec, out_dir)
    return manifest_path


def run_experiment(
    spec: ExperimentSpec,
    *,
    datasets_root: str | Path = "datasets",
    hparam_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if spec.sweep.backend == "local_grid":
        raise NotImplementedError("sweep backend 'local_grid' not yet wired to the runner")

    wandb_run = None
    overrides = dict(hparam_overrides or {})
    if spec.sweep.backend == "wandb":
        import wandb
        wandb_run = wandb.init(
            project=spec.sweep.project,
            name=spec.name,
            tags=spec.tags or None,
            config={"spec_name": spec.name, "spec_hash": spec.hash(), **overrides},
        )
        for k, v in dict(wandb_run.config).items():
            if k in ("spec_name", "spec_hash"):
                continue
            overrides[k] = v

    if overrides:
        spec = _apply_overrides(spec, overrides)

    run_id = _make_run_id(spec)
    out_dir = Path(spec.output_dir) if spec.output_dir else Path("runs") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    staging_root = out_dir / "staging"

    report: dict[str, Any] = {
        "run_id": run_id,
        "status": "started",
        "started_at": _iso_now(),
        "spec_name": spec.name,
        "spec_hash": spec.hash(),
        "spec_tags": spec.tags,
        "output_dir": str(out_dir),
        "hparam_overrides": overrides,
        "environment": _environment_info(),
        "wandb": _wandb_info(wandb_run),
    }
    spec.to_yaml(out_dir / "spec.yaml")
    _dump_report(out_dir, report)

    t0 = time.monotonic()
    try:
        manifest_path = resolve_dataset(spec, datasets_root=datasets_root)
        report["dataset"] = _dataset_lineage(spec, manifest_path)
        _dump_report(out_dir, report)

        cfg, feature_cols, categorical_cols, stage, ds_dir = _prepare_stage(
            spec, manifest_path, staging_root, out_dir, report,
        )

        if wandb_run is not None:
            wandb_run.config.update(
                {"resolved_hparams": report["resolved_hparams"], **report["split"]},
                allow_val_change=True,
            )

        _train_predict_report(
            cfg=cfg,
            stage=stage,
            ds_dir=ds_dir,
            ctx=_RunContext(
                out_dir=out_dir,
                run_id=run_id,
                spec=spec,
                feature_cols=feature_cols,
                categorical_cols=categorical_cols,
                report=report,
                wandb_run=wandb_run,
            ),
        )

        if not spec.keep_staging:
            shutil.rmtree(staging_root, ignore_errors=True)
    except Exception as e:
        report["status"] = "failed"
        report["error"] = repr(e)
        raise
    finally:
        report["finished_at"] = _iso_now()
        report["duration_s"] = round(time.monotonic() - t0, 2)
        _dump_report(out_dir, report)
        if wandb_run is not None:
            wandb_run.finish(exit_code=0 if report["status"] == "ok" else 1)

    return report


def _prepare_stage(
    spec: ExperimentSpec,
    manifest_path: Path,
    staging_root: Path,
    out_dir: Path,
    report: dict[str, Any],
) -> tuple[TrainConfig, list[str], list[str], StageResult, Path]:
    """Resolve config + feature columns, then stage train/val/eval splits.

    Mutates ``report`` with resolved hparams, features and split info and
    flushes it to disk. Returns ``(cfg, feature_cols, categorical_cols,
    stage, ds_dir)``.
    """
    cat, asp = _split_cell(spec.training.cell)
    ds_dir = manifest_path.parent
    cfg = _build_train_config(spec)
    report["resolved_hparams"] = dataclasses.asdict(cfg)
    report["features"] = _features_info(spec, cfg)
    _dump_report(out_dir, report)

    numeric_cols = [c for c in cfg.selected_features() if c in set(NUMERIC_FEATURES)]
    categorical_cols = [c for c in cfg.selected_features() if c in set(CATEGORICAL_FEATURES)]
    feature_cols = numeric_cols + categorical_cols

    parent_map_path = None
    if spec.training.propagate_labels:
        parent_map_path = ds_dir / "parent_map.json"
        if not parent_map_path.exists():
            raise FileNotFoundError(
                f"propagate_labels=True but {parent_map_path} not found. "
                f"Run scripts/export_parent_map.py with the PROTEA venv first."
            )
    stage = stage_for_training(
        source_train_parquet=ds_dir / "train.parquet",
        source_eval_parquet=ds_dir / "eval.parquet",
        cell=(cat, asp),
        feature_cols=feature_cols,
        categorical_cols=categorical_cols,
        out_dir=staging_root / f"{cat}-{asp}",
        plan=StagePlan(
            val_strategy=spec.training.val_strategy,
            val_fraction=cfg.val_fraction,
            val_holdout_snapshot=spec.training.val_holdout_snapshot,
            neg_pos_ratio=cfg.neg_pos_ratio,
            seed=cfg.seed,
            parent_map_path=parent_map_path,
            carry_go_terms=cfg.ia_weighting != "none",
        ),
    )

    report["split"] = _split_info(spec, stage)
    _dump_report(out_dir, report)
    return cfg, feature_cols, categorical_cols, stage, ds_dir


@dataclasses.dataclass
class _RunContext:
    """Per-run identifiers + sinks shared by the train/predict helpers."""

    out_dir: Path
    run_id: str
    spec: ExperimentSpec
    feature_cols: list[str]
    categorical_cols: list[str]
    report: dict[str, Any]
    wandb_run: Any


def _build_splits(
    cfg: TrainConfig,
    stage: StageResult,
    ds_dir: Path,
    feature_cols: list[str],
    report: dict[str, Any],
) -> tuple[SplitArrays, SplitArrays | None]:
    """Load train/val sequences + arrays and attach IA weights."""
    train_labels = np.load(stage.train.labels_path)

    val_seq = val_labels = val_groups = None
    if stage.val is not None and stage.val.n_rows > 0:
        val_seq = ParquetFeatureSequence(
            [str(p) for p in stage.val.bucket_paths], feature_cols,
        )
        val_labels = np.load(stage.val.labels_path)
        val_groups = np.load(stage.val.groups_path)

    train_weights, val_weights = _build_ia_weights(
        cfg, ds_dir, stage, train_labels, val_labels, report,
    )

    train_split = SplitArrays(
        seq=ParquetFeatureSequence(
            [str(p) for p in stage.train.bucket_paths], feature_cols,
        ),
        labels=train_labels,
        groups=np.load(stage.train.groups_path),
        weights=train_weights,
    )
    val_split = None
    if val_seq is not None:
        val_split = SplitArrays(
            seq=val_seq, labels=val_labels, groups=val_groups, weights=val_weights,
        )
    return train_split, val_split


def _evaluate_and_record(
    booster: Any,
    train_metrics: dict[str, Any],
    stage: StageResult,
    ctx: _RunContext,
) -> None:
    """Score eval split, write predictions, and fill ``ctx.report``."""
    eval_seq = ParquetFeatureSequence(
        [str(p) for p in stage.eval.bucket_paths], ctx.feature_cols,
    )
    eval_labels = np.load(stage.eval.labels_path)
    eval_groups = np.load(stage.eval.groups_path)
    eval_proteins = np.load(stage.eval.proteins_path, allow_pickle=True)

    scores = predict_streaming(booster, eval_seq)
    fmax = fmax_per_protein_group(scores, eval_labels, eval_groups)
    _write_predictions(
        ctx.out_dir / "predictions.parquet",
        scores=scores, labels=eval_labels,
        groups=eval_groups, proteins_per_group=eval_proteins,
    )

    best_iter = int(train_metrics["best_iteration"])
    ctx.report["metrics"] = {"test_fmax": float(fmax), "best_iteration": best_iter}
    ctx.report["feature_importance"] = {k: float(v) for k, v in sorted(
        train_metrics["feature_importance"].items(), key=lambda kv: -kv[1]
    )}
    ctx.report["status"] = "ok"

    # LM.1 champion candidate hook (appends to champions.candidates.md).
    _maybe_append_champion_candidate(
        ctx.spec.training.cell, float(fmax), ctx.run_id, ctx.out_dir
    )

    if ctx.wandb_run is not None:
        ctx.wandb_run.log({"test_fmax": float(fmax), "best_iteration": best_iter})
        ctx.wandb_run.summary["test_fmax"] = float(fmax)
        ctx.wandb_run.summary["best_iteration"] = best_iter
        ctx.wandb_run.save(str(ctx.out_dir / "run.json"))


def _train_predict_report(
    *,
    cfg: TrainConfig,
    stage: StageResult,
    ds_dir: Path,
    ctx: _RunContext,
) -> None:
    """Load splits, fit the booster, predict on eval, and fill the report."""
    train_split, val_split = _build_splits(
        cfg, stage, ds_dir, ctx.feature_cols, ctx.report,
    )
    booster, train_metrics = fit(
        train_split, val_split, cfg,
        feature_names=ctx.feature_cols,
        categorical_features=ctx.categorical_cols,
    )
    booster.save_model(str(ctx.out_dir / "model.txt"))
    _evaluate_and_record(booster, train_metrics, stage, ctx)


def _make_run_id(spec: ExperimentSpec) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{ts}_{spec.name}_{spec.hash()[:6]}"


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _environment_info() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": platform.node(),
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
        "cwd": os.getcwd(),
    }


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _git_dirty() -> bool | None:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True,
        )
        return bool(out.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _dataset_lineage(spec: ExperimentSpec, manifest_path: Path) -> dict[str, Any]:
    m = ManifestV1.load(manifest_path)
    info: dict[str, Any] = {
        "manifest_path": str(manifest_path),
        "name": m.name,
        "k": m.k,
        "schema_sha": m.schema_sha,
        "parent_schema_sha": m.parent_schema_sha,
        "spec_hash": m.spec_hash,
        "train_snapshot_pairs": m.train_snapshot_pairs,
        "eval_snapshot_pair": m.eval_snapshot_pair,
        "embedding_config_id": m.embedding_config_id,
        "ontology_snapshot_id": m.ontology_snapshot_id,
        "feature_families": m.feature_families,
        "n_train_rows": m.n_train_rows,
        "n_eval_rows": m.n_eval_rows,
    }
    if spec.dataset.spec is not None:
        info["built_from_spec"] = spec.dataset.spec.model_dump(mode="json")
    return info


def _features_info(spec: ExperimentSpec, cfg: TrainConfig) -> dict[str, Any]:
    if spec.dataset.spec is not None:
        families = spec.dataset.spec.enabled_feature_families
        spec_drop = list(spec.dataset.spec.drop_features)
    else:
        families = cfg.enabled_feature_families
        spec_drop = []
    drop = sorted(set(spec_drop) | set(cfg.drop_features))
    selected = cfg.selected_features()
    return {
        "families_enabled": families,
        "families_available": sorted(FEATURE_FAMILIES),
        "drop_features": drop,
        "feature_count": len(selected),
        "feature_columns": selected,
        "selected_numeric_count": len([c for c in selected if c in set(NUMERIC_FEATURES)]),
        "selected_categorical_count": len([c for c in selected if c in set(CATEGORICAL_FEATURES)]),
    }


def _split_info(spec: ExperimentSpec, stage: StageResult) -> dict[str, Any]:
    info = {
        "strategy": spec.training.val_strategy,
        "val_holdout_snapshot": spec.training.val_holdout_snapshot,
        "n_train": int(stage.train.n_rows),
        "n_val": int(stage.val.n_rows) if stage.val else 0,
        "n_eval": int(stage.eval.n_rows),
        "n_train_groups": int(stage.train.n_groups),
        "n_val_groups": int(stage.val.n_groups) if stage.val else 0,
        "n_eval_groups": int(stage.eval.n_groups),
    }
    train_labels = np.load(stage.train.labels_path)
    info["positive_rate_train"] = (
        float(train_labels.mean()) if train_labels.size else 0.0
    )
    return info


def _split_cell(cell: str) -> tuple[str, str]:
    cat, asp = cell.lower().split("-", 1)
    return cat, asp


def _build_train_config(spec: ExperimentSpec) -> TrainConfig:
    defaults = dict(spec.model.defaults)
    defaults["seed"] = spec.training.seed
    defaults["val_fraction"] = spec.training.val_fraction
    if spec.training.neg_pos_ratio is not None:
        defaults["neg_pos_ratio"] = spec.training.neg_pos_ratio
    allowed = set(TrainConfig.__dataclass_fields__)
    return TrainConfig(**{k: v for k, v in defaults.items() if k in allowed})


def _resolve_ia_path(cfg: TrainConfig, ds_dir: Path) -> Path:
    """Resolve the IA table path, defaulting to the tracked v227 table."""
    if cfg.ia_path:
        return Path(cfg.ia_path)
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "datasets" / "ia" / "IA-swissprot-exp-v227.txt"


def _build_ia_weights(
    cfg: TrainConfig,
    ds_dir: Path,
    stage: StageResult,
    train_labels: np.ndarray,
    val_labels: np.ndarray | None,
    report: dict[str, Any],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Palanca 1: per-row IA sample weights for the train/val splits.

    Returns ``(None, None)`` when ``ia_weighting`` is ``none`` so the
    historical uniform-weight path is byte-for-byte unchanged.
    """
    if cfg.ia_weighting == "none":
        return None, None

    if stage.train.go_terms_path is None:
        raise RuntimeError(
            "ia_weighting requested but staging did not emit go_terms.npy; "
            "the source dump is missing the go_term_id column."
        )

    ia_path = _resolve_ia_path(cfg, ds_dir)
    ia_table = load_ia_table(ia_path)
    train_go = np.load(stage.train.go_terms_path, allow_pickle=True)
    train_w = ia_weights(
        train_go, train_labels, ia_table,
        mode=cfg.ia_weighting, scale=cfg.ia_scale,
    )

    val_w = None
    if val_labels is not None and stage.val is not None and stage.val.go_terms_path:
        val_go = np.load(stage.val.go_terms_path, allow_pickle=True)
        val_w = ia_weights(
            val_go, val_labels, ia_table,
            mode=cfg.ia_weighting, scale=cfg.ia_scale,
        )

    report["ia_weighting"] = {
        "mode": cfg.ia_weighting,
        "ia_path": str(ia_path),
        "ia_scale": cfg.ia_scale,
        "ia_terms_loaded": len(ia_table),
        "train_weight_mean": float(train_w.mean()) if train_w is not None else None,
        "train_weight_max": float(train_w.max()) if train_w is not None else None,
        "train_rows_weighted_gt1": (
            int((train_w > 1.0).sum()) if train_w is not None else None
        ),
    }
    return train_w, val_w


def _write_predictions(
    path: Path,
    *,
    scores: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    proteins_per_group: np.ndarray,
) -> None:
    proteins_per_row = np.repeat(proteins_per_group, groups)
    table = pa.table({
        "protein_accession": pa.array(proteins_per_row.astype(object)),
        "label": pa.array(labels, type=pa.int8()),
        "score": pa.array(scores, type=pa.float32()),
    })
    pq.write_table(table, str(path), compression="zstd")


def _dump_report(out_dir: Path, report: dict[str, Any]) -> None:
    (out_dir / "run.json").write_text(json.dumps(report, indent=2, default=float))


_CANDIDATES_HEADER = (
    "# Champion candidates\n\n"
    "Rows appended automatically by the training hook in "
    "`src/protea_reranker_lab/runner.py` when a run's lab fmax "
    "exceeds the prior champion's CI lower bound for that cell.\n\n"
    "To promote a candidate: verify with the full cafaeval pipeline "
    "(prop=fill, norm=cafa), then run:\n\n"
    "    python scripts/update_champions.py --bootstrap --apply\n\n"
    "champions.md is NEVER auto-mutated by training.\n\n"
    "| cell | run_id | test_fmax | ci_lower_bound | date_detected |\n"
    "| --- | --- | --- | --- | --- |\n"
)


def _lb3_cell_ci(lb3_path: Path, cell: str) -> float | None:
    """Return paired_diff_ci_lo for ``cell`` from the LB.3 CSV, or None."""
    import csv as _csv

    with lb3_path.open(newline="") as fh:
        for row in _csv.DictReader(fh):
            if row["cell"] == cell:
                return float(row["paired_diff_ci_lo"])
    return None


def _append_candidate_row(
    candidates_path: Path,
    cell: str,
    run_id: str,
    test_fmax: float,
    ci_lower: float,
) -> None:
    """Append one row to ``champions.candidates.md`` (creating it if absent)."""
    import datetime as _dt

    now_iso = _dt.date.today().isoformat()
    row_line = (
        f"| {cell} | {run_id} | {test_fmax:.4f} "
        f"| {ci_lower:.4f} | {now_iso} |\n"
    )
    if candidates_path.exists():
        candidates_path.write_text(candidates_path.read_text() + row_line)
    else:
        candidates_path.write_text(_CANDIDATES_HEADER + row_line)
    print(
        f"[champion-hook] NEW CANDIDATE for cell {cell!r}: "
        f"test_fmax={test_fmax:.4f} > ci_lo={ci_lower:.4f}. "
        f"Run ID: {run_id}. Row appended to {candidates_path}. "
        "Verify with cafaeval, then run: "
        "python scripts/update_champions.py --bootstrap --apply",
        file=sys.stderr,
    )


def _maybe_append_champion_candidate(
    cell: str,
    test_fmax: float,
    run_id: str,
    out_dir: Path,
) -> None:
    """Append a candidate row when test_fmax clears the champion CI lower bound.

    Reads ``experiments/lb3/per_cell_paired_ci.csv`` for the threshold.
    champions.md is NEVER mutated here; operator promotes via
    ``python scripts/update_champions.py --bootstrap --apply``.
    All errors are caught and logged; a hook failure never aborts training.
    """
    repo_root = Path(__file__).resolve().parents[3]
    lb3_path = repo_root / "experiments" / "lb3" / "per_cell_paired_ci.csv"
    candidates_path = repo_root / "champions.candidates.md"

    try:
        if not lb3_path.exists():
            print(
                f"[champion-hook] {lb3_path} not found; skipping.",
                file=sys.stderr,
            )
            return
        ci_lower = _lb3_cell_ci(lb3_path, cell)
        if ci_lower is None:
            print(
                f"[champion-hook] cell {cell!r} not in {lb3_path}; skipping.",
                file=sys.stderr,
            )
            return
        if test_fmax <= ci_lower:
            return
        _append_candidate_row(candidates_path, cell, run_id, test_fmax, ci_lower)
    except Exception as exc:  # noqa: BLE001
        print(
            f"[champion-hook] warning: candidate check failed for "
            f"cell {cell!r} (run {run_id}): {exc}",
            file=sys.stderr,
        )


def _apply_overrides(spec: ExperimentSpec, overrides: dict[str, Any]) -> ExperimentSpec:
    training_over = {k: v for k, v in overrides.items() if k in _TRAINING_FIELDS}
    model_over = {k: v for k, v in overrides.items() if k not in _TRAINING_FIELDS}
    new_training = TrainingSpec(**{**spec.training.model_dump(), **training_over})
    new_model = ModelSpec(
        kind=spec.model.kind,
        defaults={**spec.model.defaults, **model_over},
    )
    return spec.model_copy(update={"model": new_model, "training": new_training})


def _wandb_info(run: Any) -> dict[str, Any] | None:
    if run is None:
        return None
    return {
        "backend": "wandb",
        "run_id": getattr(run, "id", None),
        "run_name": getattr(run, "name", None),
        "project": getattr(run, "project", None),
        "entity": getattr(run, "entity", None),
        "url": getattr(run, "url", None),
    }
