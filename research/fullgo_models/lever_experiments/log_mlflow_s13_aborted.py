"""Log the S13 DART lever as an ABORTED/inconclusive run to MLflow.

DART did not converge in a tractable budget on this box (see s13/train_meta.json).
We still record the attempt so the experiment is complete side-by-side with S2.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import mlflow

os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "minioadmin")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "minioadmin")
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("native-reranker-levers")

HERE = Path("/home/frapercan/Thesis2/storage/fullgo_models/lever_experiments/s13")
meta = json.load(open(HERE / "train_meta.json"))

with mlflow.start_run(run_name="pk-lever-s13"):
    mlflow.set_tags({
        "lever": "s13",
        "lever_desc": "DART: boosting_type=dart, drop_rate=0.1, fixed 769 rounds",
        "category": "PK",
        "frame": "validation 220->227",
        "status": "ABORTED_INTRACTABLE",
        "abort_reason": meta["reason"],
        "baseline_ref": "native_boosters_v5 PK 0.4142 (same recipe)",
        "pipeline": "native-reranker",
    })
    mlflow.log_params({
        "param_boosting_type": "dart",
        "param_drop_rate": 0.1,
        "param_objective": "binary",
        "param_learning_rate": 0.05,
        "param_num_leaves": 63,
        "num_boost_round": 769,
        "train_rows": meta["train_rows"],
        "pos": meta["pos"],
        "neg": meta["neg"],
        "status": "ABORTED_INTRACTABLE",
    })
    # No pk_f_micro_w metric: the booster never finished, so no score exists.
    mlflow.log_artifact(str(HERE / "train_meta.json"))
    run = mlflow.active_run()
    print(f"lever=s13 status=ABORTED run_id={run.info.run_id}")
