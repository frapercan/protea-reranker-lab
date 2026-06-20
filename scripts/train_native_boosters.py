#!/usr/bin/env python
"""Train the 3 per-category (NK/LK/PK) native-pipeline LightGBM boosters.

This is the lab-native CLI for what used to live as the ad-hoc
``storage/fullgo_models/train_native_boosters.py``. It trains the boosters
that run inside PROTEA's live ``apply_reranker`` predict path, on a parity
export dataset (the full numeric feature schema incl. the
classifier / self_prior / association columns).

Behaviour preserved from the ad-hoc script:
  * memory-safe row-group streaming (never loads the whole parquet to RAM);
  * one binary-objective booster per category, AUC metric, early stopping;
  * EVAL-ONLY ``valid_sets`` (no train-AUC; identical booster, much faster PK);
  * ``feature_schema_sha`` matching the live predict guard;
  * optional MLflow logging, gated on ``MLFLOW_TRACKING_URI`` (parent run +
    per-category nested runs + per-iteration eval AUC + booster/summary
    artifacts).

Dataset resolution:
  The train/eval parquets are pulled to a local cache via the lab's
  ``scripts/pull_dataset.py`` HTTP path (no ``minio`` SDK dependency). A direct
  MinIO-SDK fallback (``--minio-fallback``) is available when the bucket layout
  differs from the standard ``<prefix>/<dataset>/{train,eval}.parquet``.

Usage::

    # Standard (HTTP pull from MinIO/S3, no MLflow):
    python scripts/train_native_boosters.py \\
        --dataset fullgo-native-parity-SELECT-220-227 \\
        --out-dir runs/native_boosters

    # With MLflow tracking (self-hosted PROTEA-infra server):
    MLFLOW_TRACKING_URI=http://localhost:5000 \\
        python scripts/train_native_boosters.py \\
        --dataset fullgo-native-parity-SELECT-220-227 \\
        --out-dir runs/native_boosters

Environment (shared with scripts/pull_dataset.py):
    PROTEA_S3_ENDPOINT     default http://localhost:9000
    PROTEA_S3_BUCKET       default protea  (parity exports live here)
    PROTEA_S3_PREFIX       default datasets
    PROTEA_S3_ACCESS_KEY / PROTEA_S3_SECRET_KEY
    MLFLOW_TRACKING_URI / MLFLOW_EXPERIMENT / MLFLOW_RUN_NAME
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [native-boosters] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Make src/ + scripts/ importable when run directly (mirrors other lab scripts).
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--dataset", required=True,
                   help="dataset name under the bucket prefix (e.g. "
                        "fullgo-native-parity-SELECT-220-227).")
    p.add_argument("--out-dir", type=Path, default=Path("runs/native_boosters"),
                   help="output dir for ensemble_gbm_{NK,LK,PK}.txt + summary.json.")
    p.add_argument("--cache-dir", type=Path, default=Path("datasets"),
                   help="local cache root for pulled parquets (default: datasets/).")
    p.add_argument("--features", default=None,
                   help="comma-separated numeric feature subset (MR-2 flat "
                        "combiner over the component-score vector). Default: all "
                        "numeric ALL_FEATURES present in the parquet.")
    p.add_argument("--num-boost-round", type=int, default=5000)
    p.add_argument("--early-stopping-rounds", type=int, default=50)
    p.add_argument("--batch-rows", type=int, default=500_000,
                   help="row-group streaming batch size (memory knob).")
    # MinIO / S3 connection (shared semantics with pull_dataset.py)
    p.add_argument("--endpoint", default=os.environ.get("PROTEA_S3_ENDPOINT", "http://localhost:9000"))
    p.add_argument("--bucket", default=os.environ.get("PROTEA_S3_BUCKET", "protea"))
    p.add_argument("--prefix", default=os.environ.get("PROTEA_S3_PREFIX", "datasets"))
    p.add_argument("--access-key", default=os.environ.get("PROTEA_S3_ACCESS_KEY", "minioadmin"))
    p.add_argument("--secret-key", default=os.environ.get("PROTEA_S3_SECRET_KEY", "minioadmin"))
    p.add_argument("--force", action="store_true", help="re-download even if cached.")
    p.add_argument("--minio-fallback", action="store_true",
                   help="fetch parquets via the minio SDK instead of HTTP pull "
                        "(use when the bucket layout is non-standard).")
    return p


def _resolve_dataset_http(args: argparse.Namespace) -> tuple[Path, Path]:
    """Pull train/eval parquets to the local cache via the lab HTTP puller."""
    import pull_dataset

    dest = pull_dataset.pull_dataset(
        args.dataset,
        endpoint=args.endpoint,
        bucket=args.bucket,
        prefix=args.prefix,
        dest_root=args.cache_dir,
        access_key=args.access_key,
        secret_key=args.secret_key,
        force=args.force,
    )
    return dest / "train.parquet", dest / "eval.parquet"


def _resolve_dataset_minio(args: argparse.Namespace) -> tuple[Path, Path]:
    """Direct-MinIO-SDK fallback: stream the two parquets to the local cache."""
    from minio import Minio

    host = args.endpoint.split("://", 1)[-1]
    secure = args.endpoint.startswith("https")
    client = Minio(host, access_key=args.access_key, secret_key=args.secret_key, secure=secure)
    base = f"{args.prefix}/{args.dataset}" if args.prefix else args.dataset
    dest = args.cache_dir / args.dataset
    dest.mkdir(parents=True, exist_ok=True)
    out = []
    for fname in ("train.parquet", "eval.parquet"):
        local = dest / fname
        if not local.exists() or args.force:
            log.info("minio: fetching %s/%s -> %s", args.bucket, f"{base}/{fname}", local)
            client.fget_object(args.bucket, f"{base}/{fname}", str(local))
        out.append(local)
    return out[0], out[1]


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    # Import after arg-parse so --help stays fast and free of heavy imports.
    from protea_reranker_lab.native_boosters import (
        NativeBoosterConfig,
        feature_schema_sha,
        train_native_boosters,
    )
    from protea_reranker_lab.native_boosters_mlflow import MlflowLogger

    if args.minio_fallback:
        train_pq, eval_pq = _resolve_dataset_minio(args)
    else:
        train_pq, eval_pq = _resolve_dataset_http(args)

    cfg = NativeBoosterConfig(
        train_parquet=train_pq,
        eval_parquet=eval_pq,
        out_dir=args.out_dir,
        dataset_name=args.dataset,
        features=[f for f in args.features.split(",") if f.strip()] if args.features else None,
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=args.early_stopping_rounds,
        batch_rows=args.batch_rows,
    )

    logger = MlflowLogger.maybe_create(
        run_name=os.environ.get("MLFLOW_RUN_NAME", "native-boosters"),
        s3_endpoint=os.environ.get("MLFLOW_S3_ENDPOINT_URL", args.endpoint),
        s3_access_key=args.access_key,
        s3_secret_key=args.secret_key,
    )
    with (logger or contextlib.nullcontext()):
        if logger:
            logger.log_run_params(
                {
                    "dataset": args.dataset,
                    "out_dir": str(args.out_dir),
                    "objective": cfg.params["objective"],
                    "metric": cfg.params["metric"],
                    "learning_rate": cfg.params["learning_rate"],
                    "num_leaves": cfg.params["num_leaves"],
                    "min_data_in_leaf": cfg.params["min_data_in_leaf"],
                    "num_boost_round": cfg.num_boost_round,
                    "early_stopping_rounds": cfg.early_stopping_rounds,
                    "valid_sets": "eval-only (no train-AUC)",
                    "feature_schema_sha": feature_schema_sha(),
                },
                tags={"pipeline": "native-reranker", "categories": "nk,lk,pk"},
            )
        summary = train_native_boosters(cfg, mlflow_logger=logger)

    print(f"[done] {args.out_dir}  schema_sha={summary['feature_schema_sha']}")
    for cat, info in summary["boosters"].items():
        print(f"  {cat}: rows={info['train_rows']} best_iter={info['best_iter']} "
              f"eval_auc={info['eval_auc']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
