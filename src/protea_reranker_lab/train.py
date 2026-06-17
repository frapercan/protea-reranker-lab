"""W&B-instrumented CLI entrypoint — thin wrapper around :func:`run_experiment`.

Builds an :class:`ExperimentSpec` from CLI flags (or wandb-injected hparams)
and routes the actual training through the streaming runner. All heavy data
movement happens in :mod:`staging` and :mod:`runner`; this file only assembles
the spec.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from protea_contracts import FEATURE_FAMILIES
from .combiner import DEFAULT_COMBINER_COLUMNS, resolve_combiner_columns
from .experiment import DatasetRef, ExperimentSpec, ModelSpec, SweepRef, TrainingSpec
from .runner import run_experiment

#: Monolith default for ``--num-leaves`` (mirrors the parser default below). Used
#: to detect whether the operator explicitly tuned the leaf count so combiner
#: mode only shrinks it when left at the default.
_MONOLITH_NUM_LEAVES_DEFAULT = 63
#: Shallow leaf count for the combiner (a handful of score-vector inputs do not
#: need 63 leaves; a small tree keeps the meta-learner low-variance / calibrated).
_COMBINER_NUM_LEAVES_DEFAULT = 15


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="dataset directory (containing manifest.json)")
    p.add_argument("--cell", required=True, help="e.g. pk-bpo / nk-mfo / lk-cco")
    p.add_argument("--objective", default="lambdarank", choices=["binary", "lambdarank"])
    p.add_argument("--num-boost-round", "--num_boost_round", dest="num_boost_round", type=int, default=5000)
    p.add_argument("--early-stopping-rounds", "--early_stopping_rounds",
                   dest="early_stopping_rounds", type=int, default=50)
    p.add_argument("--learning-rate", "--learning_rate", dest="learning_rate", type=float, default=0.05)
    p.add_argument("--num-leaves", "--num_leaves", dest="num_leaves", type=int, default=63)
    p.add_argument("--min-data-in-leaf", "--min_data_in_leaf",
                   dest="min_data_in_leaf", type=int, default=100)
    p.add_argument("--feature-fraction", "--feature_fraction",
                   dest="feature_fraction", type=float, default=0.9)
    p.add_argument("--bagging-fraction", "--bagging_fraction",
                   dest="bagging_fraction", type=float, default=0.9)
    p.add_argument("--neg-pos-ratio", "--neg_pos_ratio",
                   dest="neg_pos_ratio", type=_nullable_float, default=None)
    p.add_argument("--val-fraction", "--val_fraction", dest="val_fraction", type=float, default=0.2)
    p.add_argument("--val-strategy", "--val_strategy", dest="val_strategy", default="protein_group",
                   choices=["protein_group", "temporal", "none"])
    p.add_argument("--temporal-holdout", "--temporal_holdout", dest="temporal_holdout", default=None)
    p.add_argument("--drop-feature-family", "--drop_feature_family",
                   dest="drop_feature_family", action="append", default=[])
    # MR-2 combiner mode: train the SHALLOW per-category combiner over the small
    # score vector instead of the 73-feature monolith. Additive + opt-in; when
    # absent the monolith path is byte-for-byte unchanged.
    p.add_argument("--combiner", dest="combiner", action="store_true",
                   help="Train the shallow per-category combiner over the score "
                        "vector (MR-2) instead of the 73-feature monolith.")
    p.add_argument("--combiner-columns", "--combiner_columns",
                   dest="combiner_columns", default=None,
                   help="Comma-separated explicit score-vector columns for the "
                        "combiner (overrides the per-category default "
                        f"{','.join(DEFAULT_COMBINER_COLUMNS)}). Implies --combiner.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb-project", "--wandb_project", dest="wandb_project", default="protea-reranker")
    p.add_argument("--wandb-mode", "--wandb_mode", dest="wandb_mode", default="online",
                   choices=["online", "offline", "disabled"])
    p.add_argument("--run-name", "--run_name", dest="run_name", default=None)
    p.add_argument("--output-dir", "--output_dir", dest="output_dir", default=None)
    p.add_argument("--save-model", "--save_model", dest="save_model", default=None)
    p.add_argument("--keep-staging", "--keep_staging", dest="keep_staging", action="store_true")
    p.add_argument("--ia-weighting", "--ia_weighting", dest="ia_weighting",
                   default="none", choices=["none", "positives", "all"],
                   help="Palanca 1: IA sample weighting mode (default none).")
    p.add_argument("--ia-path", "--ia_path", dest="ia_path", default=None,
                   help="IA table path (default datasets/ia/IA-swissprot-exp-v227.txt).")
    p.add_argument("--ia-scale", "--ia_scale", dest="ia_scale", type=float, default=1.0,
                   help="IA weight scale: weight = 1 + ia_scale * IA(go).")
    args, _ = p.parse_known_args(argv)
    return args


def _nullable_float(v: str) -> float | None:
    if v is None or v.lower() in {"none", "null", ""}:
        return None
    return float(v)


def _build_spec(args: argparse.Namespace) -> ExperimentSpec:
    drop_features: list[str] = []
    for fam in args.drop_feature_family or []:
        drop_features.extend(FEATURE_FAMILIES.get(fam, []))

    defaults = {
        "objective": args.objective,
        "num_boost_round": args.num_boost_round,
        "early_stopping_rounds": args.early_stopping_rounds,
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "min_data_in_leaf": args.min_data_in_leaf,
        "feature_fraction": args.feature_fraction,
        "bagging_fraction": args.bagging_fraction,
        "neg_pos_ratio": args.neg_pos_ratio,
        "val_fraction": args.val_fraction,
        "drop_features": drop_features,
        "ia_weighting": args.ia_weighting,
        "ia_path": args.ia_path,
        "ia_scale": args.ia_scale,
    }

    # MR-2 combiner mode: restrict the booster to the explicit score vector.
    # --combiner-columns implies --combiner. The combiner is shallow by
    # construction (a handful of inputs); cap num_leaves to a small value
    # UNLESS the operator explicitly tuned it, so the default combiner does not
    # overfit a 2-7 column input with the monolith's 63 leaves.
    combiner_on = args.combiner or args.combiner_columns is not None
    if combiner_on:
        explicit_cols = (
            [c.strip() for c in args.combiner_columns.split(",") if c.strip()]
            if args.combiner_columns else None
        )
        defaults["feature_override"] = resolve_combiner_columns(args.cell, explicit_cols)
        if args.num_leaves == _MONOLITH_NUM_LEAVES_DEFAULT:
            defaults["num_leaves"] = _COMBINER_NUM_LEAVES_DEFAULT

    return ExperimentSpec(
        name=args.run_name or f"{args.cell}_{args.objective}",
        dataset=DatasetRef(manifest=Path(args.dataset) / "manifest.json"),
        model=ModelSpec(defaults=defaults),
        training=TrainingSpec(
            cell=args.cell,
            val_strategy=args.val_strategy,
            val_fraction=args.val_fraction,
            val_holdout_snapshot=args.temporal_holdout,
            neg_pos_ratio=args.neg_pos_ratio,
            seed=args.seed,
        ),
        sweep=SweepRef(backend="none"),
        output_dir=Path(args.output_dir) if args.output_dir else None,
        keep_staging=args.keep_staging,
    )


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    spec = _build_spec(args)

    if args.wandb_mode != "disabled":
        import os
        os.environ.setdefault("WANDB_MODE", args.wandb_mode)
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
        spec = spec.model_copy(update={
            "sweep": SweepRef(backend="wandb", project=args.wandb_project),
        })

    report = run_experiment(spec)
    fmax = report.get("metrics", {}).get("test_fmax")
    if fmax is not None:
        print(f"[eval] test_fmax={fmax:.4f}")
    if args.save_model:
        booster_src = Path(report["output_dir"]) / "model.txt"
        if booster_src.exists():
            Path(args.save_model).write_bytes(booster_src.read_bytes())


if __name__ == "__main__":
    main()
    sys.exit(0)
