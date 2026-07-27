"""Log the native-reranker validation-frame f_micro_w to the self-hosted MLflow.

Reads result_no_toi.json (primary) and any result_*.json sensitivity runs, logs the
4 f_micro_w numbers (NK/LK/PK/MEAN) as metrics plus tags marking the frame and the
OPTIMISTIC nature (boosters early-stopped on this very eval split).
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import mlflow

HERE = "/home/frapercan/Thesis2/storage/fullgo_models/native_boosters_v5/validation_score"

os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "minioadmin")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "minioadmin")
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("native-reranker-validation")

primary = json.load(open(f"{HERE}/result_no_toi.json"))

with mlflow.start_run(run_name="native-boosters-v5-validation"):
    mlflow.set_tags({
        "frame": "validation 220->227",
        "optimistic": "true",
        "optimistic_reason": "boosters early-stopped on this eval split (mild upward bias)",
        "booster_set": "native_boosters_v5",
        "pipeline": "native-reranker",
        "cafaeval_recipe": "ia,prop=fill,norm=cafa,no_orphans,max_terms=None,th_step=0.01",
        "toi": "none (validation-frame TOI unavailable)",
        "pk_exclude": "none (validation-frame PK-known unavailable; PK is pessimistic)",
        "prediction_collapse": "max score per (protein,term) over union candidate streams",
        "gt_derivation": "label==1 leaf rows per category; cafaeval propagates",
    })
    fmw = primary["f_micro_w"]
    mlflow.log_metrics({
        "f_micro_w_NK": fmw["nk"],
        "f_micro_w_LK": fmw["lk"],
        "f_micro_w_PK": fmw["pk"],
        "f_micro_w_MEAN": primary["mean"],
    })
    # per-namespace detail
    for cat, d in primary["f_micro_w_per_ns"].items():
        for ns, v in d.items():
            mlflow.log_metric(f"f_micro_w_{cat}_{ns}", v)
    # log all result_*.json + apply_stats + results md as artifacts
    for f in glob.glob(f"{HERE}/result_*.json") + [f"{HERE}/apply_stats.json", f"{HERE}/RESULTS.md"]:
        if Path(f).exists():
            mlflow.log_artifact(f)
    run = mlflow.active_run()
    print("MLflow run_id:", run.info.run_id)
    print("experiment:", run.info.experiment_id)
