"""Training entrypoint — wandb-instrumented.

Designed to be the single script invoked by both manual runs and ``wandb agent``.
Reads config from CLI or wandb.config; loads a parquet partition; fits LightGBM;
evaluates per-cell Fmax on eval.parquet; logs everything to W&B.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import DatasetManifest, load_partition, split_train_val, split_train_val_temporal
from .evaluate import fmax_per_protein_group
from .reranker import TrainConfig, fit, predict


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Accept both dashed (manual) and underscored (wandb-agent) flag forms.

    Uses ``parse_known_args`` so wandb-injected hparams that don't match a
    registered flag fall through to ``wandb.config`` in ``main``.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="dataset directory (containing train/eval.parquet)")
    p.add_argument("--cell", required=True, help="e.g. pk-bpo / nk-mfo / lk-cco")
    p.add_argument("--objective", default="lambdarank", choices=["binary", "lambdarank"])
    p.add_argument("--num-boost-round", "--num_boost_round", dest="num_boost_round", type=int, default=5000)
    p.add_argument("--early-stopping-rounds", "--early_stopping_rounds", dest="early_stopping_rounds", type=int, default=50)
    p.add_argument("--learning-rate", "--learning_rate", dest="learning_rate", type=float, default=0.05)
    p.add_argument("--num-leaves", "--num_leaves", dest="num_leaves", type=int, default=63)
    p.add_argument("--min-data-in-leaf", "--min_data_in_leaf", dest="min_data_in_leaf", type=int, default=100)
    p.add_argument("--feature-fraction", "--feature_fraction", dest="feature_fraction", type=float, default=0.9)
    p.add_argument("--bagging-fraction", "--bagging_fraction", dest="bagging_fraction", type=float, default=0.9)
    p.add_argument("--neg-pos-ratio", "--neg_pos_ratio", dest="neg_pos_ratio", type=_nullable_float, default=None)
    p.add_argument("--val-fraction", "--val_fraction", dest="val_fraction", type=float, default=0.2)
    p.add_argument("--val-strategy", "--val_strategy", dest="val_strategy", default="protein_group",
                   choices=["protein_group", "temporal", "none"])
    p.add_argument("--temporal-holdout", "--temporal_holdout", dest="temporal_holdout", default=None,
                   help="snapshot_pair used as val when --val-strategy=temporal")
    p.add_argument("--drop-feature-family", "--drop_feature_family", dest="drop_feature_family",
                   action="append", default=[], help="feature family to exclude (repeatable)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb-project", "--wandb_project", dest="wandb_project", default="protea-reranker")
    p.add_argument("--wandb-mode", "--wandb_mode", dest="wandb_mode", default="online",
                   choices=["online", "offline", "disabled"])
    p.add_argument("--run-name", "--run_name", dest="run_name", default=None)
    p.add_argument("--save-model", "--save_model", dest="save_model", default=None,
                   help="path to dump booster .txt")
    args, _unknown = p.parse_known_args(argv)
    return args


def _nullable_float(v: str) -> float | None:
    """argparse type: accept the literal string ``None``/``null`` → Python None."""
    if v is None or v.lower() in {"none", "null", ""}:
        return None
    return float(v)


def _cell_split(cell: str) -> tuple[str, str]:
    cat, asp = cell.lower().split("-", 1)
    return cat, asp


def _maybe_downsample_negatives(df: pd.DataFrame, ratio: float | None, seed: int) -> pd.DataFrame:
    if ratio is None or ratio <= 0:
        return df
    pos = df[df["label"] == 1]
    neg = df[df["label"] == 0]
    target_neg = int(len(pos) * ratio)
    if target_neg >= len(neg):
        return df
    neg_ds = neg.sample(n=target_neg, random_state=seed)
    out = pd.concat([pos, neg_ds], axis=0).sort_values("protein_accession", kind="stable")
    return out.reset_index(drop=True)


def main(argv: list[str] | None = None) -> None:
    import wandb

    args = _parse_args(argv)
    cat, asp = _cell_split(args.cell)

    run = wandb.init(
        project=args.wandb_project,
        mode=args.wandb_mode,
        name=args.run_name or f"{args.cell}_{args.objective}",
        config=vars(args),
    )
    # Re-read from wandb.config so sweeps can override
    cfg_dict = dict(run.config)
    cfg = TrainConfig(
        objective=cfg_dict.get("objective", args.objective),
        num_boost_round=cfg_dict.get("num_boost_round", args.num_boost_round),
        early_stopping_rounds=cfg_dict.get("early_stopping_rounds", args.early_stopping_rounds),
        learning_rate=cfg_dict.get("learning_rate", args.learning_rate),
        num_leaves=cfg_dict.get("num_leaves", args.num_leaves),
        min_data_in_leaf=cfg_dict.get("min_data_in_leaf", args.min_data_in_leaf),
        feature_fraction=cfg_dict.get("feature_fraction", args.feature_fraction),
        bagging_fraction=cfg_dict.get("bagging_fraction", args.bagging_fraction),
        neg_pos_ratio=cfg_dict.get("neg_pos_ratio", args.neg_pos_ratio),
        val_fraction=cfg_dict.get("val_fraction", args.val_fraction),
        seed=cfg_dict.get("seed", args.seed),
        drop_features=[],
    )
    drop_families = cfg_dict.get("drop_feature_family", args.drop_feature_family) or []
    from .reranker import FEATURE_FAMILIES
    cfg.drop_features = [f for fam in drop_families for f in FEATURE_FAMILIES.get(fam, [])]

    ds_dir = Path(args.dataset)
    manifest = DatasetManifest.load(ds_dir / "manifest.json")
    wandb.config.update({"dataset_name": manifest.name, "dataset_k": manifest.k}, allow_val_change=True)

    print(f"[load] train {cat}-{asp} from {ds_dir/'train.parquet'}")
    df_tr_all = load_partition(ds_dir / "train.parquet", category=cat, aspect=asp)
    print(f"[load] eval {cat}-{asp} from {ds_dir/'eval.parquet'}")
    df_eval = load_partition(ds_dir / "eval.parquet", category=cat, aspect=asp)
    print(f"[load] train={len(df_tr_all):,}  eval={len(df_eval):,}")

    df_tr_all = _maybe_downsample_negatives(df_tr_all, cfg.neg_pos_ratio, cfg.seed)
    if args.val_strategy == "temporal":
        if not args.temporal_holdout:
            raise SystemExit("--val-strategy=temporal requires --temporal-holdout")
        df_tr, df_val = split_train_val_temporal(df_tr_all, args.temporal_holdout)
    elif args.val_strategy == "none":
        df_tr, df_val = df_tr_all, df_tr_all.iloc[0:0]
    else:
        df_tr, df_val = split_train_val(df_tr_all, cfg.val_fraction, seed=cfg.seed)

    wandb.log({
        "n_train": len(df_tr),
        "n_val": len(df_val),
        "n_eval": len(df_eval),
        "positive_rate_train": float(df_tr["label"].mean()) if len(df_tr) else 0.0,
    })

    booster, train_metrics = fit(df_tr, df_val if len(df_val) else None, cfg)
    wandb.log({
        "best_iteration": train_metrics["best_iteration"],
    })

    fi = train_metrics["feature_importance"]
    wandb.log({
        "feature_importance": wandb.Table(
            data=[[k, v] for k, v in sorted(fi.items(), key=lambda x: -x[1])],
            columns=["feature", "importance_gain"],
        ),
    })

    if len(df_eval):
        scores = predict(booster, df_eval, cfg)
        df_eval = df_eval.assign(score=scores)
        fmax = fmax_per_protein_group(df_eval, score_col="score")
        wandb.log({"test_fmax": fmax})
        print(f"[eval] test_fmax={fmax:.4f}")

    if args.save_model:
        booster.save_model(args.save_model)
        wandb.save(args.save_model)

    wandb.finish()


if __name__ == "__main__":
    main()
