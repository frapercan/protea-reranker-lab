"""Evaluate the universal booster on the clean v227-v230 test window.

The universal booster pipeline (``run_universal_booster.py``) validates on the
held-out v220-v226 band but does NOT run cafaeval on the reserved v227-v230 test
window. This script closes that gap for the F-RERANK-UNIVERSAL clean
v227-lineage recompute (ADR D41 deferred item 1).

Why it does NOT reuse ``predictions.parquet``: the runner's
``_calibrate``/``_save_predictions`` reads ``go_term_id``/``aspect`` from the RAW
eval parquet (original row order) while ``proteins``/``score`` come from the
aspect-conditioned STAGED eval buckets (a different row order), so the two are
misaligned and a (protein, go) join recovers only a few percent of rows. The
staged eval split also drops ``go_term_id`` entirely (the pooled eval path does
not carry it), so there is no key to realign on.

This script instead re-stages once only to recover the exact global categorical
vocabulary (``stage.cat_codes``, built in pooled Pass 0 over the full pool), then
scores the RAW ``eval.parquet`` directly: it reads every column from a single
parquet pass, encodes the categoricals with that vocabulary and injects
``plm_id`` / ``k_context`` exactly as ``_inject_src_features`` does at staging,
predicts in one shot, and keeps protein/go/aspect/category/label/vote_count in
the parquet's native row order. Every array therefore shares one ordering and no
join is needed. cafaeval then runs per NK/LK aspect cell for BOTH the booster and
the KNN ``vote_count`` baseline.

The RAW (uncalibrated) booster score is the honest test signal and matches the
holdout-band evaluator (``_eval_holdout_band`` also scores raw). The isotonic
calibrators were fit on this same eval split, so applying them here would be
calibration-on-test leakage. It is also metric-neutral: cafaeval sweeps a
per-namespace threshold and isotonic regression is monotonic, so a per-aspect
calibrator cannot change Fmax / f_micro / f_micro_w within an aspect.

Reports IA-weighted f_micro_w, wFmax (``f_w``), Fmax (``f``) and S_min per cell
plus the NK+LK mean, and writes the 3-column LAFA submission TSV
(protein, GO term, score) for the booster on the v227-v230 NK+LK targets.

Usage::

    poetry run python scripts/eval_universal_v227_window.py \\
        --datasets-root datasets/_stage_v227 \\
        --out-dir runs/universal_v227 \\
        --obo datasets/bench-v1-K5/go.obo \\
        --ia-path datasets/ia/IA-swissprot-exp-v227.txt \\
        --plm-ids prot_t5 --k-values 10
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pyarrow.parquet as pq

from protea_reranker_lab.pooled_staging import (
    _SrcEncoder,
    _build_global_cat_codes,
    _inject_src_features,
)
from protea_reranker_lab.staging import StagePlan
from protea_reranker_lab.universal_runner import (
    UniversalRunSpec,
    _build_feature_cols,
    _build_multi_spec,
)
from protea_reranker_lab.universal_train import _pick_primary_source

ASPECT_TO_NS = {
    "mfo": "molecular_function",
    "bpo": "biological_process",
    "cco": "cellular_component",
}
NK_LK = ("nk", "lk")
ASPECTS = ("mfo", "bpo", "cco")

_DRIVER = '''
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


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets-root", required=True)
    p.add_argument("--out-dir", required=True,
                   help="Run dir holding model.txt and calibrators/.")
    p.add_argument("--obo", required=True)
    p.add_argument("--ia-path", required=True)
    p.add_argument("--plm-ids", nargs="*", default=["prot_t5"])
    p.add_argument("--k-values", nargs="*", type=int, default=[10])
    p.add_argument("--category", default="nk",
                   help="train_cell tag for the spec (does not filter the pool).")
    p.add_argument(
        "--protea-python",
        default="/home/frapercan/Thesis2/repositories/PROTEA/.venv/bin/python",
    )
    p.add_argument("--timeout", type=int, default=1800)
    return p.parse_args()


def _build_spec(args: argparse.Namespace, out_dir: Path) -> UniversalRunSpec:
    return UniversalRunSpec(
        name="v227_window_eval",
        datasets_root=Path(args.datasets_root),
        plm_ids=args.plm_ids,
        k_values=args.k_values,
        train_cell=args.category,
        model_defaults={"objective": "lambdarank"},
        val_strategy="temporal",
        val_holdout_snapshot="v220-v226",
        ia_weighting="all",
        ia_path=Path(args.ia_path),
        obo_path=Path(args.obo),
        out_dir=out_dir,
        protea_python=Path(args.protea_python),
        plm_id_ablation=False,
    )


def _resolve_pool(spec: UniversalRunSpec) -> tuple[Any, list[str], dict[str, list[str]]]:
    """Build the multi-source pool and the global categorical vocabulary.

    The global categorical codes come from a lightweight Pass-0 scan over the
    full pool (only the categorical string columns are read), identical to the
    vocab the training run built. ``plm_id`` / ``k_context`` are injected later.
    """
    multi_spec, _ = _build_multi_spec(spec.datasets_root, spec.plm_ids, spec.k_values)
    feature_cols, categorical_cols = _build_feature_cols(spec, drop_plm_id=False)
    cat_cols = [c for c in categorical_cols if c in set(feature_cols)]
    plan = StagePlan(
        val_strategy=spec.val_strategy, val_fraction=0.2,
        val_holdout_snapshot=spec.val_holdout_snapshot, neg_pos_ratio=None,
        seed=spec.seed, carry_go_terms=True, aspect_conditioned=True,
        carry_snapshot_pair=True,
    )
    cat_codes = _build_global_cat_codes(
        multi_spec, categorical_cols=cat_cols, plan=plan,
    )
    return multi_spec, cat_cols, cat_codes


def _score_eval_direct(
    eval_pq: Path,
    model_path: Path,
    cat_cols: list[str],
    cat_codes: dict[str, list[str]],
    src: Any,
) -> dict[str, np.ndarray]:
    """Score the raw eval parquet directly with the trained booster.

    Reads every column in one parquet pass so booster scores stay row-aligned
    with protein/go/aspect/category/label/vote_count. Categoricals are encoded
    with ``cat_codes`` and ``plm_id`` / ``k_context`` injected exactly as the
    pooled staging path does (``_inject_src_features``).

    The feature column list is taken from the booster itself
    (``feature_name()``): the staging path strips the reserved ``aspect`` column
    from the feature matrix (it is a grouping key, not a feature), so the model
    has 53 features, not the 54 that ``_build_feature_cols`` lists. Using the
    model's own order guarantees the matrix columns line up with training.
    """
    booster = lgb.Booster(model_file=str(model_path))
    feature_cols = booster.feature_name()
    enc = _SrcEncoder.from_cat_codes_and_src(cat_cols, cat_codes, src)

    prot, go, lab, cat, asp, score = [], [], [], [], [], []
    pf = pq.ParquetFile(str(eval_pq))
    present_feats = set(pf.schema_arrow.names)
    for batch in pf.iter_batches(batch_size=200_000):
        feat = _inject_src_features(batch, feature_cols, enc)
        x = np.empty((batch.num_rows, len(feature_cols)), dtype=np.float32)
        for j, c in enumerate(feature_cols):
            x[:, j] = feat[c]
        score.append(booster.predict(x))
        prot.append(batch.column("protein_accession").to_numpy(zero_copy_only=False))
        go.append(batch.column("go_term_id").to_numpy(zero_copy_only=False))
        lab.append(batch.column("label").to_numpy(zero_copy_only=False))
        cat.append(batch.column("category").to_numpy(zero_copy_only=False))
        asp.append(batch.column("aspect").to_numpy(zero_copy_only=False))
    assert "vote_count" in present_feats, "eval parquet lacks vote_count (KNN baseline)"
    vote = pq.read_table(str(eval_pq), columns=["vote_count"]).column(
        "vote_count").to_numpy(zero_copy_only=False)
    return {
        "protein": np.concatenate(prot).astype(str),
        "go": np.concatenate(go).astype(str),
        "label": np.concatenate(lab).astype(np.int64),
        "category": np.concatenate(cat).astype(str),
        "aspect": np.concatenate(asp).astype(str),
        "vote_count": vote.astype(np.float64),
        "booster": np.concatenate(score).astype(np.float64),
    }


def _write_tsvs(cell_dir: Path, prot, go, label, score) -> int:
    cell_dir.mkdir(parents=True, exist_ok=True)
    (cell_dir / "pred.tsv").write_text(
        "".join(f"{p}\t{g}\t{s:.6f}\n" for p, g, s in zip(prot, go, score))
    )
    gt = [f"{p}\t{g}\n" for p, g, lab in zip(prot, go, label) if lab > 0]
    (cell_dir / "gt.tsv").write_text("".join(gt))
    return len(gt)


def _run_cafaeval(
    cell: str, work_dir: Path, obo: Path, ia: Path,
    protea_python: str, timeout: int,
) -> dict[str, Any]:
    cell_dir = work_dir / cell
    out_json = cell_dir / "cafaeval_out.json"
    driver = _DRIVER.format(
        obo=str(obo), pred_dir=str(cell_dir),
        gt=str(cell_dir / "gt.tsv"), ia=str(ia), out_json=str(out_json),
    )
    try:
        subprocess.run(
            [protea_python, "-c", driver],
            timeout=timeout, check=True, capture_output=True, text=True,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        detail = getattr(exc, "stderr", "") or ""
        return {"error": f"{exc}: {detail[-500:]}"}
    if not out_json.exists():
        return {"error": "cafaeval produced no output"}
    raw = json.loads(out_json.read_text())
    ns = ASPECT_TO_NS[cell.split("-", 1)[1]]

    def pick(col: str, best_kind: str) -> float | None:
        for row in raw.get(best_kind, []):
            if row.get("ns") == ns:
                v = row.get(col)
                if v is not None:
                    return float(v)
        return None

    return {
        "f_micro_w": pick("f_micro_w", "f_micro_w"),
        "f_micro": pick("f_micro", "f_micro"),
        "wfmax": pick("f_w", "f_w"),
        "fmax": pick("f", "f"),
        "smin": pick("s", "s"),
    }


def _mean(vals: list[float | None]) -> float | None:
    clean = [float(v) for v in vals if v is not None]
    return float(np.mean(clean)) if clean else None


def _eval_cells(
    data: dict[str, np.ndarray], args: argparse.Namespace,
) -> dict[str, Any]:
    cells: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="v227win_r_") as tr, \
         tempfile.TemporaryDirectory(prefix="v227win_k_") as tk:
        rdir, kdir = Path(tr), Path(tk)
        for cat in NK_LK:
            for asp in ASPECTS:
                cell = f"{cat}-{asp}"
                m = (data["category"] == cat) & (data["aspect"] == asp)
                if not m.any():
                    cells[cell] = {"status": "no_rows"}
                    continue
                prot, go, lab = data["protein"][m], data["go"][m], data["label"][m]
                n_gt = _write_tsvs(rdir / cell, prot, go, lab, data["booster"][m])
                _write_tsvs(kdir / cell, prot, go, lab, data["vote_count"][m])
                cells[cell] = {
                    "n_pred_rows": int(m.sum()),
                    "n_gt_rows": n_gt,
                    "n_proteins": int(np.unique(prot).size),
                    "reranker": _run_cafaeval(
                        cell, rdir, Path(args.obo), Path(args.ia_path),
                        args.protea_python, args.timeout,
                    ),
                    "knn_baseline": _run_cafaeval(
                        cell, kdir, Path(args.obo), Path(args.ia_path),
                        args.protea_python, args.timeout,
                    ),
                }
    return cells


def _summary(cells: dict[str, Any]) -> dict[str, Any]:
    def metric(key: str, name: str) -> list[float | None]:
        return [
            e[key].get(name)
            for e in cells.values()
            if isinstance(e, dict) and isinstance(e.get(key), dict)
        ]

    out: dict[str, Any] = {}
    for name in ("f_micro_w", "wfmax", "fmax", "smin"):
        out[f"reranker_mean_{name}"] = _mean(metric("reranker", name))
        out[f"knn_mean_{name}"] = _mean(metric("knn_baseline", name))
    return out


def _eval_source_parquet(args: argparse.Namespace) -> Path:
    return (
        Path(args.datasets_root)
        / f"bench-v1-K{args.k_values[0]}-v226-lineage-{args.plm_ids[0]}"
        / "eval.parquet"
    )


def main() -> int:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    eval_out = out_dir / "v227_window_eval"
    eval_out.mkdir(parents=True, exist_ok=True)
    spec = _build_spec(args, out_dir)

    multi_spec, cat_cols, cat_codes = _resolve_pool(spec)
    src = _pick_primary_source(multi_spec)
    data = _score_eval_direct(
        _eval_source_parquet(args), out_dir / "model.txt",
        cat_cols, cat_codes, src,
    )
    cells = _eval_cells(data, args)

    report: dict[str, Any] = {
        "window": "v227-v230",
        "eval_rows_total": int(len(data["label"])),
        "score_signal": "raw_uncalibrated_booster",
        "cells": cells,
        "summary_nk_lk_mean": _summary(cells),
    }

    lafa_mask = np.isin(data["category"], list(NK_LK))
    lafa_path = eval_out / "lafa_v227_v230_universal.tsv"
    # CAFA / LAFA expect scores in (0, 1]. The booster emits unbounded raw
    # LambdaRank scores (e.g. -4..+3), so squash with a logistic; it is
    # monotonic, so per-protein ranking and the cafaeval metrics above are
    # unchanged. Round to 1e-6; clamp away from exact 0 so no candidate is
    # dropped by a strict ">0" reader.
    lafa_scores = 1.0 / (1.0 + np.exp(-data["booster"][lafa_mask]))
    lafa_scores = np.clip(lafa_scores, 1e-6, 1.0)
    lafa_path.write_text(
        "".join(
            f"{p}\t{g}\t{s:.6f}\n"
            for p, g, s in zip(
                data["protein"][lafa_mask], data["go"][lafa_mask], lafa_scores,
            )
        )
    )
    report["lafa_tsv"] = str(lafa_path)
    report["lafa_tsv_rows"] = int(lafa_mask.sum())
    report["lafa_score_transform"] = "sigmoid(raw_booster), clipped to [1e-6, 1.0]"
    (eval_out / "v227_window_metrics.json").write_text(json.dumps(report, indent=2))

    print(json.dumps(report["summary_nk_lk_mean"], indent=2))
    print(f"metrics -> {eval_out / 'v227_window_metrics.json'}")
    print(f"lafa tsv -> {lafa_path} ({report['lafa_tsv_rows']} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
