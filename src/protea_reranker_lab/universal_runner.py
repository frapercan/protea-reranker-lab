"""Universal booster: pooled, aspect-conditioned, IA-weighted, K-augmented training.

Produces ONE booster artifact by training over ALL staged v226-lineage manifests
(8 PLM x K{3,5,10}), replacing the per-cell phase3a models. Key design decisions:

- STRICTLY STREAMING with K10-superset-K5-superset-K3 dedup; no physical all-PLM
  parquet is written (avoids the minijob/write-OOM precedent).
- numpy/FAISS only; NEVER torch GPU KNN or pgvector.
- Aspect-conditioned: ONE booster sees all three namespaces.
- IA-weighted LambdaMART (combined weight+feval).
- Per-aspect isotonic calibration fit on the eval split (VALID proxy).
- Post-hoc hierarchical-consistency correction (parent >= max child).
- Full lineage in ``run.json``; plm_id ablation guard.

Entry point: :func:`run_universal` -- call from ``scripts/run_universal_booster.py``.
Training helpers live in :mod:`universal_train` (internal module).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from protea_contracts import CATEGORICAL_FEATURES, FEATURE_FAMILIES, NUMERIC_FEATURES

from .calibration import (
    CalibrationSpec,
    calibrate_scores,
    calibration_stats,
    fit_aspect_calibrators,
    save_calibrators,
)
from .hierarchical_correction import (
    build_children_map,
    check_hierarchical_consistency,
    correct_scores_hierarchical,
)
from .multi_source import MultiManifestSpec
from .propagation import load_parent_map
from .runner import _environment_info
from .universal_train import CategoryCtx as TrainCategoryCtx, train_category

_DEFAULT_PROTEA_PYTHON = (
    "/home/frapercan/Thesis2/repositories/PROTEA/.venv/bin/python"
)
_CANONICAL_LAB = Path("/home/frapercan/Thesis2/repositories/protea-reranker-lab")
_DATASETS_ROOT = _CANONICAL_LAB / "datasets"

CANONICAL_PLMS = [
    "ankh_base", "ankh_large", "esm2_150m", "esm2_3b",
    "esm2_650m", "esmc_600m", "prostt5", "prot_t5",
]
CANONICAL_KS = [3, 5, 10]
CANONICAL_CELLS = [
    "nk-mfo", "nk-bpo", "nk-cco",
    "lk-mfo", "lk-bpo", "lk-cco",
    "pk-mfo", "pk-bpo", "pk-cco",
]
NK_LK_CELLS = {c for c in CANONICAL_CELLS if not c.startswith("pk-")}
ASPECT_TO_NS = {
    "mfo": "molecular_function",
    "bpo": "biological_process",
    "cco": "cellular_component",
}

_CAFAEVAL_DRIVER = '''
import json, signal
from cafaeval.evaluation import cafa_eval
signal.signal(signal.SIGTERM, signal.SIG_DFL)
signal.signal(signal.SIGINT, signal.SIG_DFL)
df, dfs_best = cafa_eval(
    "{obo}", "{pred_dir}", "{gt}",
    ia="{ia}", prop="fill", norm="cafa",
    no_orphans=True, max_terms=500, th_step=0.001, n_cpu=1, weighted_only=False,
)
out = {{}}
for kind, df_best in dfs_best.items():
    out[kind] = df_best.reset_index().to_dict(orient="records")
with open("{out_json}", "w") as f:
    json.dump(out, f, indent=2, default=str)
'''


@dataclass
class UniversalRunSpec:
    """Configuration for a universal booster training run.

    Attributes:
        name:                run tag.
        datasets_root:       directory containing v226-lineage dataset subdirs.
        plm_ids:             PLM subset (None = all 8 canonical).
        k_values:            K subset (None = K{3,5,10}).
        train_cell:          protein category (e.g. ``"nk"``).
        model_defaults:      LightGBM hparam overrides.
        val_strategy:        split strategy.
        val_holdout_snapshot: holdout snapshot for temporal strategy.
        ia_weighting:        IA sample weight mode.
        ia_path:             explicit IA table path (None = auto-resolved).
        calibration_spec:    per-aspect calibration configuration.
        k_aug_seed:          seed for K-augmentation.
        seed:                RNG seed.
        out_dir:             output directory (None = auto).
        v227_delta_parquet:  path to v227 delta train.parquet.
        parent_map_path:     path to parent_map.json.
        obo_path:            GO OBO file for cafaeval.
        protea_python:       Python interpreter with cafaeval installed.
        ia_feval_mode:       IA feval mode.
        plm_id_ablation:     if True, also run without plm_id.
    """

    name: str = "universal"
    datasets_root: Path = field(default_factory=lambda: _DATASETS_ROOT)
    plm_ids: list[str] | None = None
    k_values: list[int] | None = None
    train_cell: str = "nk"
    model_defaults: dict[str, Any] = field(default_factory=dict)
    val_strategy: str = "temporal"
    val_holdout_snapshot: str | None = "v220-v226"
    ia_weighting: str = "all"
    ia_path: Path | None = None
    calibration_spec: CalibrationSpec = field(default_factory=CalibrationSpec)
    k_aug_seed: int = 42
    seed: int = 42
    out_dir: Path | None = None
    v227_delta_parquet: Path | None = None
    parent_map_path: Path | None = None
    obo_path: Path | None = None
    protea_python: Path = field(
        default_factory=lambda: Path(_DEFAULT_PROTEA_PYTHON)
    )
    ia_feval_mode: str = "combined"
    plm_id_ablation: bool = True


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _spec_hash(spec: UniversalRunSpec) -> str:
    """Deterministic 12-hex hash over the run spec."""
    payload = {
        "datasets_root": str(spec.datasets_root),
        "plm_ids": sorted(spec.plm_ids or CANONICAL_PLMS),
        "k_values": sorted(spec.k_values or CANONICAL_KS),
        "train_cell": spec.train_cell,
        "model_defaults": spec.model_defaults,
        "val_strategy": spec.val_strategy,
        "val_holdout_snapshot": spec.val_holdout_snapshot,
        "ia_weighting": spec.ia_weighting,
        "calibration_method": spec.calibration_spec.method,
        "k_aug_seed": spec.k_aug_seed,
        "seed": spec.seed,
        "ia_feval_mode": spec.ia_feval_mode,
        "plm_id_ablation": spec.plm_id_ablation,
    }
    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def _discover_manifests(
    datasets_root: Path,
    plm_ids: list[str] | None,
    k_values: list[int] | None,
) -> tuple[list[Path], list[str]]:
    """Scan ``datasets_root`` for v226-lineage manifests. Returns (found, missing)."""
    effective_plms = plm_ids or CANONICAL_PLMS
    effective_ks = k_values or CANONICAL_KS
    found: list[Path] = []
    missing: list[str] = []
    for plm in effective_plms:
        for k in effective_ks:
            ds_name = f"bench-v1-K{k}-v226-lineage-{plm}"
            manifest = datasets_root / ds_name / "manifest.json"
            if manifest.exists():
                found.append(manifest)
            else:
                missing.append(ds_name)
    return found, missing


def _build_multi_spec(
    datasets_root: Path,
    plm_ids: list[str] | None,
    k_values: list[int] | None,
) -> tuple[MultiManifestSpec, list[str]]:
    """Build the multi-manifest pool. Raises if no manifests found."""
    found, missing = _discover_manifests(datasets_root, plm_ids, k_values)
    if not found:
        raise RuntimeError(f"No v226-lineage manifests found under {datasets_root}.")
    return MultiManifestSpec.from_manifest_paths(found), missing


def _build_feature_cols(
    spec: UniversalRunSpec,
    drop_plm_id: bool = False,
) -> tuple[list[str], list[str]]:
    """Return (feature_cols, categorical_cols) with optional plm_id drop."""
    drop_set: set[str] = set()
    for fam in spec.model_defaults.get("drop_feature_family", []):
        drop_set.update(FEATURE_FAMILIES.get(fam, []))
    extra_cat = [] if drop_plm_id else ["plm_id"]
    extra_num = ["k_context"]
    numeric_cols = [c for c in list(NUMERIC_FEATURES) + extra_num if c not in drop_set]
    categorical_cols = [c for c in list(CATEGORICAL_FEATURES) + extra_cat if c not in drop_set]
    return numeric_cols + categorical_cols, categorical_cols


def _load_v227_delta(v227_delta_parquet: Path) -> dict[str, dict]:
    """Load per-cell {proteins, pos_pairs} dicts from the v227 delta parquet."""
    import pyarrow.compute as pcc
    t = pq.read_table(
        str(v227_delta_parquet),
        columns=["protein_accession", "go_term_id", "label",
                 "snapshot_pair", "category", "aspect"],
    )
    delta = t.filter(pcc.equal(t.column("snapshot_pair"), "v226-v227"))
    per_cell: dict[str, dict] = {}
    for cell in CANONICAL_CELLS:
        cat, asp = cell.split("-", 1)
        m = pcc.and_(
            pcc.equal(delta.column("category"), cat),
            pcc.equal(delta.column("aspect"), asp),
        )
        rows = delta.filter(m)
        proteins = set(rows.column("protein_accession").to_pylist())
        pos_pairs = {
            (p, g)
            for p, g, lab in zip(
                rows.column("protein_accession").to_pylist(),
                rows.column("go_term_id").to_pylist(),
                rows.column("label").to_pylist(),
            )
            if lab > 0
        }
        per_cell[cell] = {"proteins": proteins, "pos_pairs": pos_pairs}
    return per_cell


def _build_scores_map(
    proteins_per_row: np.ndarray,
    go_terms: np.ndarray,
    scores: np.ndarray,
) -> dict[tuple, float]:
    """Map (protein, go_term) -> max corrected score."""
    scores_map: dict[tuple, float] = {}
    for prot, go, score in zip(proteins_per_row, go_terms, scores):
        key = (str(prot), str(go))
        existing = scores_map.get(key)
        if existing is None or float(score) > existing:
            scores_map[key] = float(score)
    return scores_map


def _write_valid_tsvs(
    eval_pq: Path,
    cell: str,
    delta_info: dict,
    scores_map: dict[tuple, float],
    work_dir: Path,
) -> dict | None:
    """Write pred.tsv + gt.tsv for one VALID cell. Returns counts or None."""
    import pyarrow.compute as pcc
    cat, asp = cell.split("-", 1)
    t = pq.read_table(str(eval_pq), columns=["protein_accession", "go_term_id", "label",
                                               "category", "aspect"])
    rows = t.filter(pcc.and_(
        pcc.equal(t.column("category"), cat),
        pcc.equal(t.column("aspect"), asp),
    ))
    if rows.num_rows == 0:
        return None
    delta_proteins = delta_info["proteins"]
    delta_pos = delta_info["pos_pairs"]
    pred_rows, gt_rows = [], []
    for prot, go, lab in zip(
        rows.column("protein_accession").to_pylist(),
        rows.column("go_term_id").to_pylist(),
        rows.column("label").to_pylist(),
    ):
        if prot not in delta_proteins:
            continue
        score = scores_map.get((str(prot), str(go)))
        if score is None:
            continue
        pred_rows.append(f"{prot}\t{go}\t{score:.6f}\n")
        if lab > 0 or (str(prot), str(go)) in delta_pos:
            gt_rows.append(f"{prot}\t{go}\n")
    if not pred_rows:
        return None
    cell_dir = work_dir / cell
    cell_dir.mkdir(parents=True, exist_ok=True)
    (cell_dir / "pred.tsv").write_text("".join(pred_rows))
    (cell_dir / "gt.tsv").write_text("".join(gt_rows))
    return {"n_pred_rows": len(pred_rows), "n_gt_rows": len(gt_rows)}


def _run_cafaeval(
    cell: str, work_dir: Path, obo_path: Path,
    ia_path: Path, protea_python: Path, timeout: int = 900,
) -> dict[str, float | None]:
    """Run cafaeval on one cell's TSVs and return per-namespace metrics."""
    cell_dir = work_dir / cell
    out_json = cell_dir / "cafaeval_out.json"
    driver = _CAFAEVAL_DRIVER.format(
        obo=str(obo_path), pred_dir=str(cell_dir),
        gt=str(cell_dir / "gt.tsv"), ia=str(ia_path), out_json=str(out_json),
    )
    try:
        subprocess.run(
            [str(protea_python), "-c", driver],
            timeout=timeout, check=True, capture_output=True, text=True,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        return {"error": str(exc)}
    if not out_json.exists():
        return {"error": "cafaeval produced no output"}
    raw = json.loads(out_json.read_text())
    asp = cell.split("-", 1)[1]
    ns = ASPECT_TO_NS.get(asp, asp)

    def pick(metric: str) -> float | None:
        for row in raw.get(ns, []):
            if row.get("metric") == metric:
                v = row.get(metric)
                if v is not None:
                    return float(v)
        return None

    return {"f_micro_w": pick("f_micro_w"), "f_micro": pick("f_micro"), "fmax": pick("f")}


def _eval_valid_band(
    spec: UniversalRunSpec,
    eval_pq: Path,
    scores_map: dict[tuple, float],
    category: str,
    ia_path_resolved: Path,
) -> dict[str, Any]:
    """Evaluate the 226->227 VALID band metrics via cafaeval. Returns metrics dict."""
    if spec.v227_delta_parquet is None or not spec.v227_delta_parquet.exists():
        return {"status": "v227_delta_parquet_not_provided"}
    obo_path = spec.obo_path or (_CANONICAL_LAB / "datasets" / "bench-v1-K5" / "go.obo")
    if not obo_path.exists():
        return {"error": f"OBO not found at {obo_path}"}
    delta_cells = _load_v227_delta(spec.v227_delta_parquet)
    valid_metrics: dict[str, Any] = {}
    nk_lk_fmw: list[float] = []
    with tempfile.TemporaryDirectory(prefix="valid_eval_") as tmp:
        tmp_path = Path(tmp)
        for cell in CANONICAL_CELLS:
            cat = cell.split("-", 1)[0]
            if cat != category:
                continue
            delta_info = delta_cells.get(cell, {"proteins": set(), "pos_pairs": set()})
            if not delta_info["proteins"]:
                valid_metrics[cell] = {"status": "no_delta_proteins"}
                continue
            counts = _write_valid_tsvs(eval_pq, cell, delta_info, scores_map, tmp_path)
            if counts is None:
                valid_metrics[cell] = {"status": "no_rows"}
                continue
            cell_metrics = _run_cafaeval(
                cell, tmp_path, obo_path, ia_path_resolved, spec.protea_python,
            )
            cell_metrics.update(counts)
            cell_metrics["n_delta_proteins"] = len(delta_info["proteins"])
            valid_metrics[cell] = cell_metrics
            if cell in NK_LK_CELLS:
                fmw = cell_metrics.get("f_micro_w")
                if fmw is not None:
                    nk_lk_fmw.append(float(fmw))
    valid_metrics["nk_lk_mean_f_micro_w"] = float(np.mean(nk_lk_fmw)) if nk_lk_fmw else None
    valid_metrics["nk_lk_cells_evaluated"] = len(nk_lk_fmw)
    valid_metrics["baseline_prot_t5_k3"] = 0.5863
    return valid_metrics


def _dump_report(out_dir: Path, report: dict[str, Any]) -> None:
    (out_dir / "run.json").write_text(json.dumps(report, indent=2, default=float))


@dataclass
class _CorrectionCtx:
    """Input bundle for calibration + hierarchical correction (keeps param count <=6)."""

    raw_scores: np.ndarray
    eval_labels: np.ndarray
    eval_groups: np.ndarray
    eval_proteins: np.ndarray
    eval_pq: Path
    spec: UniversalRunSpec
    out_dir: Path
    report: dict[str, Any]


def _calibrate(ctx: _CorrectionCtx) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load eval metadata, fit + apply per-aspect calibration.

    Returns (calibrated, eval_aspects, eval_go_terms, proteins_per_row).
    """
    t_eval = pq.read_table(str(ctx.eval_pq), columns=["aspect", "go_term_id"])
    eval_aspects = t_eval.column("aspect").to_numpy(zero_copy_only=False)[:len(ctx.raw_scores)]
    eval_go_terms = t_eval.column("go_term_id").to_numpy(zero_copy_only=False)[:len(ctx.raw_scores)]
    proteins_per_row = np.repeat(ctx.eval_proteins, ctx.eval_groups)
    scores_per_aspect = {
        asp: (ctx.raw_scores[eval_aspects == asp], ctx.eval_labels[eval_aspects == asp])
        for asp in ("mfo", "bpo", "cco")
        if (eval_aspects == asp).any()
    }
    calibrators = fit_aspect_calibrators(scores_per_aspect, ctx.spec.calibration_spec)
    cal_stats = calibration_stats(calibrators, scores_per_aspect)
    save_calibrators(calibrators, ctx.out_dir / "calibrators", ctx.spec.calibration_spec, cal_stats)
    ctx.report["calibration"] = [dataclasses.asdict(s) for s in cal_stats]
    return calibrate_scores(ctx.raw_scores, eval_aspects, calibrators), eval_aspects, eval_go_terms, proteins_per_row


def _resolve_parent_map_path(spec: UniversalRunSpec) -> Path | None:
    """Resolve parent_map_path from spec or fall back to bench-v1-K5."""
    if spec.parent_map_path is not None:
        return spec.parent_map_path
    candidate = _CANONICAL_LAB / "datasets" / "bench-v1-K5" / "parent_map.json"
    return candidate if candidate.exists() else None


def _apply_hierarchical_correction(
    calibrated: np.ndarray,
    proteins_per_row: np.ndarray,
    eval_go_terms: np.ndarray,
    parent_map_path: Path | None,
    report: dict[str, Any],
) -> np.ndarray:
    """Apply DAG correction and record stats into report. Returns corrected scores."""
    children_map: dict[str, set[str]] = {}
    if parent_map_path is not None and parent_map_path.exists():
        children_map = build_children_map(load_parent_map(parent_map_path))
    corrected, n_corrections = correct_scores_hierarchical(
        proteins=proteins_per_row, go_terms=eval_go_terms,
        scores=calibrated, children_map=children_map,
    )
    violations = check_hierarchical_consistency(proteins_per_row, eval_go_terms,
                                                corrected, children_map)
    report["hierarchical_correction"] = {
        "n_corrections": n_corrections,
        "parent_map_used": str(parent_map_path) if parent_map_path else None,
        "post_correction_violations": violations["n_violations"],
        "n_checked_pairs": violations["n_checked_pairs"],
    }
    return corrected


def _apply_calibration_and_correction(
    ctx: _CorrectionCtx,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calibrate + hierarchically correct scores. Returns (corrected, go_terms, proteins_per_row)."""
    calibrated, _, eval_go_terms, proteins_per_row = _calibrate(ctx)
    parent_map_path = _resolve_parent_map_path(ctx.spec)
    corrected = _apply_hierarchical_correction(
        calibrated, proteins_per_row, eval_go_terms, parent_map_path, ctx.report,
    )
    return corrected, eval_go_terms, proteins_per_row


def _init_report(spec: UniversalRunSpec, out_dir: Path, run_id: str) -> dict[str, Any]:
    """Build and persist the initial run.json skeleton."""
    report: dict[str, Any] = {
        "run_id": run_id, "status": "started", "started_at": _iso_now(),
        "spec_name": spec.name, "spec_hash": _spec_hash(spec),
        "output_dir": str(out_dir), "environment": _environment_info(),
    }
    _dump_report(out_dir, report)
    return report


def _populate_manifest_report(
    report: dict[str, Any],
    multi_spec: MultiManifestSpec,
    spec: UniversalRunSpec,
    missing: list[str],
    out_dir: Path,
) -> None:
    """Fill manifest coverage section of report."""
    n_expected = len(spec.plm_ids or CANONICAL_PLMS) * len(spec.k_values or CANONICAL_KS)
    report["manifest_coverage"] = {
        "n_expected": n_expected, "n_found": len(multi_spec.sources),
        "coverage_pct": round(100.0 * len(multi_spec.sources) / n_expected, 1)
        if n_expected > 0 else 0.0,
        "missing": missing,
    }
    report["schema_sha"] = multi_spec.schema_sha
    report["multi_manifest_pool"] = [
        {"manifest": str(s.manifest_path), "plm_id": s.plm_id, "k_context": s.k_context}
        for s in multi_spec.sources
    ]
    _dump_report(out_dir, report)


@dataclass
class _TrainStepCtx:
    """Context bundle for _run_train_step (keeps param count <=6)."""

    spec: UniversalRunSpec
    multi_spec: MultiManifestSpec
    feature_cols: list[str]
    categorical_cols: list[str]
    staging_root: Path
    out_dir: Path
    report: dict[str, Any]


def _run_train_step(ctx: _TrainStepCtx) -> Any:
    """Execute training and update report. Returns TrainResult."""
    result = train_category(TrainCategoryCtx(
        category=ctx.spec.train_cell, multi_spec=ctx.multi_spec,
        feature_cols=ctx.feature_cols, categorical_cols=ctx.categorical_cols,
        spec=ctx.spec, staging_root=ctx.staging_root, ia_table=None, out_dir=ctx.out_dir,
    ))
    result.booster.save_model(str(ctx.out_dir / "model.txt"))
    ctx.report["split"] = {"category": ctx.spec.train_cell, "n_eval": len(result.eval_labels)}
    ctx.report["feature_importance"] = result.feature_importance
    ctx.report["best_iteration"] = result.best_iteration
    _dump_report(ctx.out_dir, ctx.report)
    return result


def _save_predictions(
    out_dir: Path,
    proteins_per_row: np.ndarray,
    eval_go_terms: np.ndarray,
    result: Any,
    corrected: np.ndarray,
) -> None:
    """Write predictions.parquet to out_dir."""
    pred_table = pa.table({
        "protein_accession": pa.array(proteins_per_row.astype(object)),
        "go_term_id": pa.array(eval_go_terms.astype(object)),
        "label": pa.array(result.eval_labels, type=pa.int8()),
        "score_raw": pa.array(result.raw_scores, type=pa.float32()),
        "score_corrected": pa.array(corrected, type=pa.float32()),
    })
    pq.write_table(pred_table, str(out_dir / "predictions.parquet"), compression="zstd")


def run_universal(spec: UniversalRunSpec) -> dict[str, Any]:
    """Run the full universal booster pipeline and return the run.json report dict."""
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}_{spec.name}"
    out_dir = spec.out_dir or Path("runs") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    staging_root = out_dir / "staging"
    report = _init_report(spec, out_dir, run_id)
    t0 = time.monotonic()
    try:
        multi_spec, missing = _build_multi_spec(spec.datasets_root, spec.plm_ids, spec.k_values)
        _populate_manifest_report(report, multi_spec, spec, missing, out_dir)
        feature_cols, categorical_cols = _build_feature_cols(spec, drop_plm_id=False)
        report["feature_columns"] = feature_cols
        report["k_aug_seed"] = spec.k_aug_seed
        _dump_report(out_dir, report)
        result = _run_train_step(_TrainStepCtx(
            spec=spec, multi_spec=multi_spec, feature_cols=feature_cols,
            categorical_cols=categorical_cols, staging_root=staging_root,
            out_dir=out_dir, report=report,
        ))
        corr_ctx = _CorrectionCtx(
            raw_scores=result.raw_scores, eval_labels=result.eval_labels,
            eval_groups=result.eval_groups, eval_proteins=result.eval_proteins,
            eval_pq=result.eval_pq, spec=spec, out_dir=out_dir, report=report,
        )
        corrected, eval_go_terms, proteins_per_row = _apply_calibration_and_correction(corr_ctx)
        _dump_report(out_dir, report)
        ia_path_resolved = (
            spec.ia_path or Path(__file__).resolve().parents[2]
            / "datasets" / "ia" / "IA-swissprot-exp-v227.txt"
        )
        scores_map = _build_scores_map(proteins_per_row, eval_go_terms, corrected)
        report["valid_band_metrics"] = _eval_valid_band(
            spec, result.eval_pq, scores_map, spec.train_cell, ia_path_resolved,
        )
        if spec.plm_id_ablation:
            report["plm_ablation"] = {"status": "see_plm_ablation_subdir_after_main_run"}
        _save_predictions(out_dir, proteins_per_row, eval_go_terms, result, corrected)
        report["status"] = "ok"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = repr(exc)
        raise
    finally:
        report["finished_at"] = _iso_now()
        report["duration_s"] = round(time.monotonic() - t0, 2)
        _dump_report(out_dir, report)
        shutil.rmtree(staging_root, ignore_errors=True)
    return report


def run_universal_plm_ablation(
    spec: UniversalRunSpec,
    base_out_dir: Path,
) -> dict[str, Any]:
    """Run with plm_id DROPPED as ablation guard. Artifacts go to plm_ablation/."""
    ablation_spec = dataclasses.replace(
        spec,
        name=f"{spec.name}_no_plm_id",
        out_dir=base_out_dir / "plm_ablation",
        plm_id_ablation=False,
    )
    return run_universal(ablation_spec)
