"""Turn an :class:`ExperimentSpec` into results — with full traceability.

Artefacts written to ``output_dir`` on every launch (``run.json`` and
``spec.yaml`` are written *at start* so a crash still leaves a trace):

- ``spec.yaml``          — the ExperimentSpec that produced this run
- ``run.json``           — run_id, status, timings, git sha, resolved
                           hparams, dataset lineage (spec/schema shas),
                           features used, metrics, feature_importance
- ``model.txt``          — LightGBM booster (only on success)
- ``predictions.parquet``— eval rows + score column (only on success)

W&B sweep backends are deferred: they keep using ``train.py`` under
``wandb agent`` until we have a reason to unify.
"""

from __future__ import annotations

import dataclasses
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .builder import build_dataset
from .data import load_partition, split_train_val, split_train_val_temporal
from .evaluate import fmax_per_protein_group
from .experiment import ExperimentSpec
from .reranker import FEATURE_FAMILIES, TrainConfig, fit, predict
from .schemas import ManifestV1, required_columns


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


def run_experiment(spec: ExperimentSpec, *, datasets_root: str | Path = "datasets") -> dict[str, Any]:
    if spec.sweep.backend != "none":
        raise NotImplementedError(
            f"sweep backend '{spec.sweep.backend}' not yet wired to the runner; "
            "launch via wandb agent + scripts/train.py for now"
        )

    run_id = _make_run_id(spec)
    out_dir = Path(spec.output_dir) if spec.output_dir else Path("runs") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "run_id": run_id,
        "status": "started",
        "started_at": _iso_now(),
        "spec_name": spec.name,
        "spec_hash": spec.hash(),
        "spec_tags": spec.tags,
        "output_dir": str(out_dir),
        "environment": _environment_info(),
    }
    spec.to_yaml(out_dir / "spec.yaml")
    _dump_report(out_dir, report)

    t0 = time.monotonic()
    try:
        manifest_path = resolve_dataset(spec, datasets_root=datasets_root)
        report["dataset"] = _dataset_lineage(spec, manifest_path)
        _dump_report(out_dir, report)

        cat, asp = _split_cell(spec.training.cell)
        ds_dir = manifest_path.parent
        df_tr_all = load_partition(ds_dir / "train.parquet", category=cat, aspect=asp)
        df_eval = load_partition(ds_dir / "eval.parquet", category=cat, aspect=asp)

        cfg = _build_train_config(spec)
        df_tr_all = _maybe_downsample_negatives(df_tr_all, cfg.neg_pos_ratio, cfg.seed)
        df_tr, df_val = _split(df_tr_all, spec, cfg)
        _check_nonempty(df_tr, spec, df_tr_all)

        report["resolved_hparams"] = dataclasses.asdict(cfg)
        report["features"] = _features_info(spec)
        report["split"] = {
            "strategy": spec.training.val_strategy,
            "val_holdout_snapshot": spec.training.val_holdout_snapshot,
            "n_train": int(len(df_tr)),
            "n_val": int(len(df_val)),
            "n_eval": int(len(df_eval)),
            "positive_rate_train": float(df_tr["label"].mean()) if len(df_tr) else 0.0,
        }
        _dump_report(out_dir, report)

        booster, train_metrics = fit(df_tr, df_val if len(df_val) else None, cfg)
        booster.save_model(str(out_dir / "model.txt"))

        scores = predict(booster, df_eval, cfg)
        df_eval = df_eval.assign(score=scores)
        fmax = fmax_per_protein_group(df_eval, score_col="score") if len(df_eval) else 0.0
        df_eval[["protein_accession", "go_term_id", "label", "score"]].to_parquet(
            out_dir / "predictions.parquet", index=False
        )

        report["metrics"] = {
            "test_fmax": float(fmax),
            "best_iteration": int(train_metrics["best_iteration"]),
        }
        report["feature_importance"] = {k: float(v) for k, v in sorted(
            train_metrics["feature_importance"].items(), key=lambda kv: -kv[1]
        )}
        report["status"] = "ok"
    except Exception as e:
        report["status"] = "failed"
        report["error"] = repr(e)
        raise
    finally:
        report["finished_at"] = _iso_now()
        report["duration_s"] = round(time.monotonic() - t0, 2)
        _dump_report(out_dir, report)

    return report


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
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _git_dirty() -> bool | None:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True
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


def _features_info(spec: ExperimentSpec) -> dict[str, Any]:
    # Use the dataset-level families (what the parquet actually contains)
    # rather than re-deriving from reranker.py.
    if spec.dataset.spec is not None:
        families = spec.dataset.spec.enabled_feature_families
        drop = list(spec.dataset.spec.drop_features)
    else:
        families = None
        drop = []
    cols = required_columns(families, drop)
    feature_cols = [c for c in cols if c not in (
        "protein_accession", "go_term_id", "label",
        "category", "aspect", "snapshot_pair",
    )]
    return {
        "families_enabled": families,
        "families_available": sorted(FEATURE_FAMILIES),
        "drop_features": drop,
        "feature_count": len(feature_cols),
        "feature_columns": feature_cols,
    }


def _split_cell(cell: str) -> tuple[str, str]:
    cat, asp = cell.lower().split("-", 1)
    return cat, asp


def _build_train_config(spec: ExperimentSpec) -> TrainConfig:
    defaults = dict(spec.model.defaults)
    # training-level fields always override model defaults (more specific).
    defaults["seed"] = spec.training.seed
    defaults["val_fraction"] = spec.training.val_fraction
    if spec.training.neg_pos_ratio is not None:
        defaults["neg_pos_ratio"] = spec.training.neg_pos_ratio
    allowed = set(TrainConfig.__dataclass_fields__)
    return TrainConfig(**{k: v for k, v in defaults.items() if k in allowed})


def _maybe_downsample_negatives(df, ratio, seed):
    if ratio is None or ratio <= 0:
        return df
    pos = df[df["label"] == 1]
    neg = df[df["label"] == 0]
    target = int(len(pos) * ratio)
    if target >= len(neg):
        return df
    import pandas as pd
    neg_ds = neg.sample(n=target, random_state=seed)
    return pd.concat([pos, neg_ds]).sort_values("protein_accession", kind="stable").reset_index(drop=True)


def _split(df_tr_all, spec: ExperimentSpec, cfg: TrainConfig):
    strat = spec.training.val_strategy
    if strat == "none":
        return df_tr_all, df_tr_all.iloc[0:0]
    if strat == "temporal":
        assert spec.training.val_holdout_snapshot
        return split_train_val_temporal(df_tr_all, spec.training.val_holdout_snapshot)
    return split_train_val(df_tr_all, cfg.val_fraction, seed=cfg.seed)


def _check_nonempty(df_tr, spec: ExperimentSpec, df_tr_all) -> None:
    if len(df_tr):
        return
    pairs = sorted(df_tr_all["snapshot_pair"].unique()) if "snapshot_pair" in df_tr_all.columns else []
    raise ValueError(
        f"training set empty after split "
        f"(strategy={spec.training.val_strategy}, holdout={spec.training.val_holdout_snapshot}); "
        f"train.parquet contains snapshot_pairs={pairs} — "
        "either pick a holdout outside this set, use val_strategy=protein_group, "
        "or use a dataset with multiple train snapshot_pairs."
    )


def _dump_report(out_dir: Path, report: dict[str, Any]) -> None:
    (out_dir / "run.json").write_text(json.dumps(report, indent=2, default=float))
