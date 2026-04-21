"""Offline grid/random search over LightGBM hparams — no W&B required.

Loads the dataset once, iterates configs, fits + evaluates per trial, and
writes a JSON report to ``experiments/<name>.json``. Complements
``sweeps/*.yaml`` (those need W&B); this one runs fully local.

Grid format (JSON)::

    {
      "learning_rate": [0.05, 0.1],
      "num_leaves": [31, 63],
      "min_data_in_leaf": [50, 100]
    }

Fixed (non-swept) hparams go in ``--base``. Example::

    python scripts/local_grid.py \\
        --dataset datasets/smoke-K5 \\
        --cell pk-bpo \\
        --grid sweeps/local_lr.json \\
        --out experiments/smoke_lr.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import time
from pathlib import Path
from typing import Any

from protea_reranker_lab.data import (
    DatasetManifest,
    load_partition,
    split_train_val,
    split_train_val_temporal,
)
from protea_reranker_lab.evaluate import fmax_per_protein_group
from protea_reranker_lab.reranker import FEATURE_FAMILIES, TrainConfig, fit, predict


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--cell", required=True, help="e.g. pk-bpo")
    p.add_argument("--grid", required=True, help="JSON file: dict[str, list]")
    p.add_argument("--base", default=None, help="JSON file: fixed hparams merged under grid")
    p.add_argument("--out", required=True, help="output report JSON path")
    p.add_argument("--n-samples", type=int, default=None,
                   help="random subset size; default = full grid")
    p.add_argument("--val-strategy", default="protein_group",
                   choices=["protein_group", "temporal", "none"])
    p.add_argument("--temporal-holdout", default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _expand(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(grid.keys())
    return [dict(zip(keys, vals, strict=False)) for vals in itertools.product(*[grid[k] for k in keys])]


def _cfg_from(params: dict[str, Any], base: dict[str, Any]) -> TrainConfig:
    merged = {**base, **params}
    drop_fams = merged.pop("drop_feature_family", []) or []
    drop_feats = [f for fam in drop_fams for f in FEATURE_FAMILIES.get(fam, [])]
    return TrainConfig(
        objective=merged.get("objective", "lambdarank"),
        num_boost_round=merged.get("num_boost_round", 3000),
        early_stopping_rounds=merged.get("early_stopping_rounds", 50),
        learning_rate=merged.get("learning_rate", 0.05),
        num_leaves=merged.get("num_leaves", 63),
        min_data_in_leaf=merged.get("min_data_in_leaf", 100),
        feature_fraction=merged.get("feature_fraction", 0.9),
        bagging_fraction=merged.get("bagging_fraction", 0.9),
        bagging_freq=merged.get("bagging_freq", 5),
        neg_pos_ratio=merged.get("neg_pos_ratio", None),
        val_fraction=merged.get("val_fraction", 0.2),
        seed=merged.get("seed", 42),
        drop_features=drop_feats,
    )


def main() -> None:
    args = _parse_args()
    cat, asp = args.cell.lower().split("-", 1)

    ds_dir = Path(args.dataset)
    manifest = DatasetManifest.load(ds_dir / "manifest.json")

    print(f"[load] {cat}-{asp} from {ds_dir}")
    df_tr_all = load_partition(ds_dir / "train.parquet", category=cat, aspect=asp)
    df_eval = load_partition(ds_dir / "eval.parquet", category=cat, aspect=asp)
    print(f"[load] train={len(df_tr_all):,}  eval={len(df_eval):,}")

    with open(args.grid) as f:
        grid = json.load(f)
    base = {}
    if args.base:
        with open(args.base) as f:
            base = json.load(f)

    configs = _expand(grid)
    if args.n_samples and args.n_samples < len(configs):
        rng = random.Random(args.seed)
        configs = rng.sample(configs, args.n_samples)
    print(f"[grid] {len(configs)} configs to evaluate")

    trials: list[dict[str, Any]] = []
    for i, params in enumerate(configs, 1):
        cfg = _cfg_from(params, base)

        if args.val_strategy == "temporal":
            if not args.temporal_holdout:
                raise SystemExit("--val-strategy=temporal requires --temporal-holdout")
            df_tr, df_val = split_train_val_temporal(df_tr_all, args.temporal_holdout)
        elif args.val_strategy == "none":
            df_tr, df_val = df_tr_all, df_tr_all.iloc[0:0]
        else:
            df_tr, df_val = split_train_val(df_tr_all, cfg.val_fraction, seed=cfg.seed)

        t0 = time.time()
        booster, train_metrics = fit(df_tr, df_val if len(df_val) else None, cfg)
        scores = predict(booster, df_eval, cfg)
        fmax = fmax_per_protein_group(df_eval.assign(score=scores), score_col="score")
        runtime = time.time() - t0

        trial = {
            "idx": i,
            "params": params,
            "best_iteration": train_metrics["best_iteration"],
            "test_fmax": fmax,
            "runtime_s": round(runtime, 2),
            "n_train": len(df_tr),
            "n_val": len(df_val),
        }
        trials.append(trial)
        print(f"[{i:>3}/{len(configs)}] {params} → fmax={fmax:.4f} "
              f"best_iter={train_metrics['best_iteration']} t={runtime:.1f}s")

    trials.sort(key=lambda t: t["test_fmax"], reverse=True)
    report = {
        "dataset": manifest.name,
        "cell": args.cell,
        "val_strategy": args.val_strategy,
        "temporal_holdout": args.temporal_holdout,
        "base": base,
        "grid": grid,
        "n_trials": len(trials),
        "best": trials[0] if trials else None,
        "trials": trials,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[done] wrote {out_path}")
    if trials:
        b = trials[0]
        print(f"[best] fmax={b['test_fmax']:.4f}  params={b['params']}")


if __name__ == "__main__":
    main()
