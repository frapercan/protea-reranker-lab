"""CLI entry-point for the F-RERANK-UNIVERSAL.5 universal booster run.

Trains a single universal reranker over ALL staged v226-lineage manifests
(8 PLM x K{3,5,10}), fits per-aspect isotonic calibration on the eval split,
applies post-hoc hierarchical-consistency correction (parent >= max child),
and reports IA-weighted f_micro_w on the 226->227 VALID band.

Usage::

    poetry run python scripts/run_universal_booster.py \\
        --datasets-root /path/to/lab/datasets \\
        --out-dir runs/universal_booster \\
        --v227-delta-parquet /path/to/v227/train.parquet \\
        --category nk \\
        --ia-weighting all \\
        --seed 42 \\
        --plm-id-ablation

Key outputs under ``--out-dir``:
    model.txt                  -- LightGBM universal booster
    calibrators/               -- isotonic calibrators per aspect
    predictions.parquet        -- raw + calibrated + corrected scores
    run.json                   -- full lineage artifact

Acceptance criteria (F-RERANK-UNIVERSAL.5):
- One universal booster + calibrator artifact exists.
- Predictions on VALID are hierarchically consistent (0 violations reported).
- NK+LK mean f_micro_w on 226->227 VALID band is reported vs baseline 0.5863.
- Candidate recall reported alongside.
- plm_id ablation run completed as guard.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the F-RERANK-UNIVERSAL.5 universal booster pipeline."
    )
    p.add_argument(
        "--datasets-root", default=None,
        help="Directory containing bench-v1-K*-v226-lineage-* subdirs. "
             "Defaults to ~/Thesis2/repositories/protea-reranker-lab/datasets.",
    )
    p.add_argument(
        "--out-dir", default="runs/universal_booster",
        help="Output directory for the universal booster artifacts.",
    )
    p.add_argument(
        "--v227-delta-parquet", default=None,
        help="Path to the v227 delta train.parquet for VALID evaluation. "
             "Defaults to the known agent-farm results path.",
    )
    p.add_argument(
        "--parent-map", default=None,
        help="Path to parent_map.json for hierarchical correction. "
             "Defaults to bench-v1-K5/parent_map.json.",
    )
    p.add_argument(
        "--obo", default=None,
        help="Path to go.obo for cafaeval. Defaults to bench-v1-K5/go.obo.",
    )
    p.add_argument(
        "--ia-path", default=None,
        help="Path to IA TSV table. Defaults to datasets/ia/IA-swissprot-exp-v227.txt.",
    )
    p.add_argument(
        "--category", default="nk",
        help="Protein category to train on (default: nk).",
    )
    p.add_argument(
        "--ia-weighting", default="all",
        choices=["none", "positives", "all"],
        help="IA sample weighting mode (default: all).",
    )
    p.add_argument(
        "--ia-feval-mode", default="combined",
        choices=["none", "feval", "combined"],
        help="IA feval mode (default: combined).",
    )
    p.add_argument(
        "--calibration-method", default="isotonic",
        choices=["isotonic", "platt"],
        help="Score calibration method (default: isotonic).",
    )
    p.add_argument(
        "--num-boost-round", type=int, default=3000,
        help="LightGBM num_boost_round (default: 3000).",
    )
    p.add_argument(
        "--early-stopping-rounds", type=int, default=50,
        help="LightGBM early stopping rounds (default: 50).",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed (default: 42).",
    )
    p.add_argument(
        "--k-aug-seed", type=int, default=42,
        help="K-augmentation RNG seed (default: 42).",
    )
    p.add_argument(
        "--plm-ids", nargs="*", default=None,
        help="PLM subset (default: all 8 canonical PLMs).",
    )
    p.add_argument(
        "--k-values", nargs="*", type=int, default=None,
        help="K subset (default: 3 5 10).",
    )
    p.add_argument(
        "--plm-id-ablation", action="store_true",
        help="Also run without plm_id feature (ablation guard).",
    )
    p.add_argument(
        "--run-name", default="universal",
        help="Run name tag (default: universal).",
    )
    p.add_argument(
        "--protea-python", default=None,
        help="Python interpreter with cafaeval installed. "
             "Defaults to PROTEA venv.",
    )
    args, _ = p.parse_known_args(argv)
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    # Resolve paths
    _CANONICAL_LAB = Path("/home/frapercan/Thesis2/repositories/protea-reranker-lab")
    _AGENT_FARM_RESULTS = Path("/home/frapercan/Thesis2/agent-farm/results")

    datasets_root = (
        Path(args.datasets_root)
        if args.datasets_root
        else _CANONICAL_LAB / "datasets"
    )
    v227_delta = (
        Path(args.v227_delta_parquet)
        if args.v227_delta_parquet
        else _AGENT_FARM_RESULTS / "executor-1780829216-57db" / "v227data" / "train.parquet"
    )
    # OBO + parent_map resolution.
    #
    # The v226-lineage dataset dirs (bench-v1-K{3,5,10}-v226-lineage-<plm>) ship
    # only train/eval parquet + manifest.json (they do NOT carry a go.obo or
    # parent_map.json). All of them share the SAME ontology_snapshot_id
    # (35c3ad67, == data-version releases/2026-01-23) as the canonical
    # bench-v1-K5 / bench-v1-K5-filtered datasets, whose go.obo + parent_map.json
    # are byte-identical. So the canonical copies are the correct GO snapshot.
    #
    # A naive obo arg pointing at a v226-lineage dir resolves to a non-existent
    # file, which previously left valid_band_metrics={error: OBO not found} and
    # produced no real f_micro_w. We therefore validate the supplied/derived
    # path and fall back to the canonical shared-snapshot copy.
    _CANONICAL_GO_DIRS = ("bench-v1-K5-filtered", "bench-v1-K5")

    def _resolve_shared_artifact(supplied: str | None, filename: str) -> Path:
        candidates: list[Path] = []
        if supplied:
            candidates.append(Path(supplied))
        for d in _CANONICAL_GO_DIRS:
            candidates.append(_CANONICAL_LAB / "datasets" / d / filename)
        for cand in candidates:
            if cand.exists():
                if supplied and cand != Path(supplied):
                    print(
                        f"[universal] WARNING: supplied {filename} "
                        f"'{supplied}' missing; falling back to shared-snapshot "
                        f"copy '{cand}'."
                    )
                return cand
        # Nothing exists; return the first canonical candidate so the
        # downstream error message names the canonical location.
        return _CANONICAL_LAB / "datasets" / _CANONICAL_GO_DIRS[0] / filename

    parent_map = _resolve_shared_artifact(args.parent_map, "parent_map.json")
    obo_path = _resolve_shared_artifact(args.obo, "go.obo")
    print(f"[universal] resolved obo_path     = {obo_path} (exists={obo_path.exists()})")
    print(f"[universal] resolved parent_map   = {parent_map} (exists={parent_map.exists()})")
    ia_path = Path(args.ia_path) if args.ia_path else None
    protea_python = (
        Path(args.protea_python)
        if args.protea_python
        else Path("/home/frapercan/Thesis2/repositories/PROTEA/.venv/bin/python")
    )

    # Import here to keep arg parsing fast
    from protea_reranker_lab.calibration import CalibrationSpec
    from protea_reranker_lab.universal_runner import UniversalRunSpec, run_universal, run_universal_plm_ablation

    spec = UniversalRunSpec(
        name=args.run_name,
        datasets_root=datasets_root,
        plm_ids=args.plm_ids,
        k_values=args.k_values,
        train_cell=args.category,
        model_defaults={
            "objective": "lambdarank",
            "num_boost_round": args.num_boost_round,
            "early_stopping_rounds": args.early_stopping_rounds,
        },
        val_strategy="temporal",
        val_holdout_snapshot="v220-v226",
        ia_weighting=args.ia_weighting,
        ia_path=ia_path,
        calibration_spec=CalibrationSpec(method=args.calibration_method),
        k_aug_seed=args.k_aug_seed,
        seed=args.seed,
        out_dir=Path(args.out_dir),
        v227_delta_parquet=v227_delta,
        parent_map_path=parent_map,
        obo_path=obo_path,
        protea_python=protea_python,
        ia_feval_mode=args.ia_feval_mode,
        plm_id_ablation=args.plm_id_ablation,
    )

    print(f"[universal] Starting run '{spec.name}' -> {spec.out_dir}")
    print(f"[universal] datasets_root = {spec.datasets_root}")
    print(f"[universal] v227_delta_parquet = {spec.v227_delta_parquet}")
    print(f"[universal] parent_map_path = {spec.parent_map_path}")

    report = run_universal(spec)

    # Print summary
    print("\n=== UNIVERSAL BOOSTER RUN SUMMARY ===")
    print(f"status:     {report['status']}")
    print(f"run_id:     {report['run_id']}")
    print(f"spec_hash:  {report['spec_hash']}")
    coverage = report.get("manifest_coverage", {})
    print(f"manifests:  {coverage.get('n_found', '?')}/{coverage.get('n_expected', '?')} "
          f"({coverage.get('coverage_pct', '?')}%)")
    print(f"schema_sha: {report.get('schema_sha', '?')}")
    cal = report.get("calibration", [])
    fitted = [c["aspect"] for c in cal if c.get("fitted")]
    print(f"calibrated: {fitted}")
    hc = report.get("hierarchical_correction", {})
    print(f"hc_corrections: {hc.get('n_corrections', '?')} "
          f"violations_after: {hc.get('post_correction_violations', '?')}")
    vb = report.get("valid_band_metrics", {})
    nk_lk_fmw = vb.get("nk_lk_mean_f_micro_w")
    baseline = vb.get("baseline_prot_t5_k3", 0.5863)
    print(f"\nVALID band NK+LK mean f_micro_w: {nk_lk_fmw}")
    print(f"Baseline (prot_t5 K3):           {baseline}")
    if nk_lk_fmw is not None:
        delta = float(nk_lk_fmw) - float(baseline)
        sign = "+" if delta >= 0 else ""
        print(f"Delta vs baseline:               {sign}{delta:.4f}")
        if delta < 0:
            print("NOTE: universal < per-cell baseline -- publishable finding, reported honestly.")

    print(f"\nOutput directory: {spec.out_dir}")
    print(f"run.json:         {spec.out_dir}/run.json")
    print(f"model.txt:        {spec.out_dir}/model.txt")
    print(f"calibrators/:     {spec.out_dir}/calibrators/")

    # plm_id ablation
    if args.plm_id_ablation:
        print("\n[universal] Running plm_id ablation (drop plm_id feature)...")
        ablation_report = run_universal_plm_ablation(spec, Path(args.out_dir))
        ablation_vb = ablation_report.get("valid_band_metrics", {})
        ablation_fmw = ablation_vb.get("nk_lk_mean_f_micro_w")
        print(f"[ablation] NK+LK mean f_micro_w (no plm_id): {ablation_fmw}")
        if nk_lk_fmw is not None and ablation_fmw is not None:
            delta_abl = float(nk_lk_fmw) - float(ablation_fmw)
            print(f"[ablation] plm_id contribution to f_micro_w: +{delta_abl:.4f}")
        (Path(args.out_dir) / "plm_ablation_summary.json").write_text(
            json.dumps({
                "full_model_nk_lk_fmw": nk_lk_fmw,
                "no_plm_id_nk_lk_fmw": ablation_fmw,
                "delta": (float(nk_lk_fmw) - float(ablation_fmw))
                if nk_lk_fmw is not None and ablation_fmw is not None else None,
                "interpretation": (
                    "positive=plm_id adds value; negative=plm_id hurts (collapse risk)"
                    if ablation_fmw is not None else "ablation not evaluated"
                ),
            }, indent=2)
        )

    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
