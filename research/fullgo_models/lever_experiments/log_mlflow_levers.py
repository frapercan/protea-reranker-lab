"""Log a single PK SCORE-LEVER run to the self-hosted MLflow.

Experiment: native-reranker-levers. One run per lever (s2 / s13). Logs the lever
config as params, pk_f_micro_w_validation as the headline metric, the delta vs the
0.4142 baseline, per-namespace detail, and tags (lever, frame, optimistic).

Usage:
  python log_mlflow_levers.py --lever s2  --out s2
  python log_mlflow_levers.py --lever s13 --out s13
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import mlflow

BASELINE = 0.4142

os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "minioadmin")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "minioadmin")
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("native-reranker-levers")

LEVER_DESC = {
    "s2": "PK imbalance: scale_pos_weight=(#neg/#pos)",
    "s2_unbal": "PK imbalance: is_unbalance=true",
    "s13": "DART: boosting_type=dart, drop_rate=0.1, fixed budget",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lever", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    meta = json.load(open(out / "train_meta.json"))
    result = json.load(open(out / f"result_{args.lever}.json"))
    pk = result["pk_f_micro_w"]
    delta = round(pk - BASELINE, 6)

    with mlflow.start_run(run_name=f"pk-lever-{args.lever}"):
        mlflow.set_tags({
            "lever": args.lever,
            "lever_desc": LEVER_DESC.get(args.lever, args.lever),
            "category": "PK",
            "frame": "validation 220->227",
            "optimistic": "true",
            "optimistic_reason": "early-stopped/eval-tuned on this eval split; no TOI; no PK-known exclude",
            "cafaeval_recipe": "ia,prop=fill,norm=cafa,no_orphans,max_terms=None,th_step=0.01",
            "baseline_ref": "native_boosters_v5 PK 0.4142 (same recipe)",
            "pipeline": "native-reranker",
        })
        params = {f"param_{k}": v for k, v in meta.get("params", {}).items()}
        params.update({
            "num_boost_round": meta.get("num_boost_round"),
            "best_iteration": meta.get("best_iteration"),
            "scale_pos_weight_value": meta.get("scale_pos_weight_value"),
            "train_rows": meta.get("train_rows"),
            "pos": meta.get("pos"),
            "neg": meta.get("neg"),
        })
        mlflow.log_params(params)
        mlflow.log_metric("pk_f_micro_w_validation", pk)
        mlflow.log_metric("pk_f_micro_w_delta_vs_baseline", delta)
        mlflow.log_metric("baseline_pk_f_micro_w", BASELINE)
        if meta.get("eval_auc") is not None:
            mlflow.log_metric("eval_auc", float(meta["eval_auc"]))
        for ns, v in result.get("per_ns", {}).items():
            mlflow.log_metric(f"pk_f_micro_w_{ns}", v)
        for f in glob.glob(str(out / "*.json")):
            mlflow.log_artifact(f)
        run = mlflow.active_run()
        print(f"lever={args.lever} pk={pk} delta={delta:+.6f} run_id={run.info.run_id}")


if __name__ == "__main__":
    main()
