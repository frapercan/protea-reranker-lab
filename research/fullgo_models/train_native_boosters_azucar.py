"""H-azucar: FINAL TEST-ready native reranker boosters (refit on train+eval).

The validation boosters (native_boosters_v5/) were trained train-on-train with
early-stopping on the held-out 227 eval split, yielding per-category best_iter
NK=542 / LK=170 / PK=769. This is the standard "refit on all data with the
chosen config" step: we CONCATENATE train.parquet + eval.parquet per category
(everything is <=227) and train one LightGBM per category at a FIXED
num_boost_round (no early-stop, since no held-out remains). The result is the
model that will be sealed and applied to the LAFA TEST frame (227->230).

Leakage-clean: all rows are <=227; TEST is 227->230, so folding the 220->227
validation window into the final fit introduces NO test leakage.

num_boost_round choice: we use the validated best_iter AS-IS (NK 542 / LK 170 /
PK 769). Reasoning: more training data nominally shifts the early-stop optimum
up a little, but the relationship is noisy and adding rounds with NO held-out to
catch over/under-shoot is the riskier move. The validated best_iter was selected
on a real held-out eval split and is the defensible config; keeping it as-is is
the conservative, reproducible "refit on all data with chosen config". A +10-15%
bump is left as a deliberate non-action (documented here and in summary.json).

Memory-safe: reuses the proven streaming row-group accumulator from
train_native_boosters.py (peak well under the box RAM). PK (~50M rows) dominates.
Read-only on MinIO; does not touch the live stack.
"""

from __future__ import annotations

import gc
import json
import os
import sys

import lightgbm as lgb
import numpy as np

# Reuse the proven streaming/accumulation logic + helpers from the validated
# training script (same directory). We override BASE/OUT via env BEFORE import so
# its module-level constants pick up the azucar dataset/output, then call its
# _accumulate directly. This avoids duplicating the (delicate, memory-bounded)
# row-group streaming code.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

# Azucar refit: train on train+eval CONCATENATED, fixed rounds, no early-stop.
DATASET = "datasets/fullgo-union-SELECT-160-220-227-v5"
OUT_DIR = os.path.join(_HERE, "native_boosters_azucar")
os.environ["BOOSTER_DATASET"] = DATASET
os.environ["BOOSTER_OUT"] = OUT_DIR

# MLflow env (set BEFORE importing the base module so its MLFLOW_ON gate + the
# S3 creds setdefault see them). The base module's MLflow plumbing is reused only
# for its import; we drive logging explicitly here (no per-category eval cb).
os.environ.setdefault("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "minioadmin")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "minioadmin")

import train_native_boosters as base  # noqa: E402

# Validated per-category early-stop best_iter from native_boosters_v5/summary.json.
FIXED_ROUNDS = {"nk": 542, "lk": 170, "pk": 769}
ROUND_BUMP_PCT = 0  # deliberate non-action: keep validated best_iter as-is.

PARAMS = base.PARAMS  # same hyperparameters as the validation fit.
CATS = base.CATS


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)

    # Feature columns: numeric ALL_FEATURES present in the parquet (drop the 4
    # string categoricals) -- identical selection to the validated run.
    import io

    import pyarrow.parquet as pq

    schema_names = pq.read_schema(
        io.BytesIO(base._CLIENT.get_object(base.BUCKET, f"{DATASET}/train.parquet").read())
    ).names
    cat_set = set(base.CATEGORICAL_FEATURES)
    feat = [f for f in base.ALL_FEATURES if f in schema_names and f not in cat_set]

    sha = base.compute_feature_schema_sha(
        base.infer_active_feature_families(
            compute_alignments=True, compute_taxonomy=True, compute_v6_features=False
        )
    )
    assert sha == "7fcecf26aa0a", f"feature_schema_sha drift: {sha} != 7fcecf26aa0a"
    print(
        f"numeric_features={len(feat)} (dropped {len(cat_set)} categoricals) "
        f"schema_sha={sha}",
        flush=True,
    )

    # --- MLflow run (explicit; no nested per-category eval callback) ----------
    mlflow = None
    run_id = None
    if base.MLFLOW_ON:
        import mlflow as mlflow  # noqa: F811

        mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT", "native-reranker-boosters"))

    def _round_for(cat: str) -> int:
        n = FIXED_ROUNDS[cat]
        return int(round(n * (1 + ROUND_BUMP_PCT / 100.0)))

    # --- stream + concatenate train AND eval per category --------------------
    print("streaming TRAIN...", flush=True)
    train = base._accumulate(f"{DATASET}/train.parquet", feat)
    print("streaming EVAL...", flush=True)
    ev = base._accumulate(f"{DATASET}/eval.parquet", feat)

    summary = {
        "feature_schema_sha": sha,
        "frame": "train<=227 (refit train+eval CONCATENATED)",
        "combined": True,
        "early_stopping": False,
        "round_bump_pct": ROUND_BUMP_PCT,
        "fixed_rounds_source": "native_boosters_v5/summary.json best_iter (validated 220->227 early-stop)",
        "features": feat,
        "dropped_categoricals": sorted(cat_set),
        "boosters": {},
    }

    cm = (
        mlflow.start_run(run_name="native-boosters-azucar-le227")
        if base.MLFLOW_ON
        else _NullCtx()
    )
    with cm as run:
        if base.MLFLOW_ON:
            run_id = run.info.run_id
            mlflow.set_tags(
                {
                    "pipeline": "native-reranker",
                    "frame": "train<=227 (refit train+eval)",
                    "combined": "true",
                    "categories": ",".join(CATS),
                }
            )
            mlflow.log_params(
                {
                    "booster_dataset": DATASET,
                    "out_dir": OUT_DIR,
                    "objective": PARAMS["objective"],
                    "metric": PARAMS["metric"],
                    "learning_rate": PARAMS["learning_rate"],
                    "num_leaves": PARAMS["num_leaves"],
                    "min_data_in_leaf": PARAMS["min_data_in_leaf"],
                    "feature_fraction": PARAMS["feature_fraction"],
                    "bagging_fraction": PARAMS["bagging_fraction"],
                    "bagging_freq": PARAMS["bagging_freq"],
                    "early_stopping": False,
                    "round_bump_pct": ROUND_BUMP_PCT,
                    "num_boost_round_nk": _round_for("nk"),
                    "num_boost_round_lk": _round_for("lk"),
                    "num_boost_round_pk": _round_for("pk"),
                    "n_features": len(feat),
                    "n_dropped_categoricals": len(cat_set),
                    "feature_schema_sha": sha,
                }
            )

        for cat in CATS:
            if cat not in train or train[cat] is None:
                print(f"!! {cat}: 0 train rows", flush=True)
                continue
            Xtr, ytr = train[cat]
            if cat in ev and ev[cat] is not None:
                Xev, yev = ev[cat]
                Xcat = np.vstack([Xtr, Xev])
                ycat = np.concatenate([ytr, yev])
                del Xev, yev
                ev[cat] = None
            else:
                Xcat, ycat = Xtr, ytr
            train[cat] = None
            del Xtr, ytr
            gc.collect()

            nrounds = _round_for(cat)
            print(
                f"[{cat}] combined rows={len(ycat)} pos={int(ycat.sum())} "
                f"num_boost_round={nrounds} (no early-stop)",
                flush=True,
            )
            dtr = lgb.Dataset(Xcat, label=ycat, feature_name=feat, free_raw_data=False)
            booster = lgb.train(
                PARAMS,
                dtr,
                num_boost_round=nrounds,
                callbacks=[lgb.log_evaluation(period=50)],
            )
            path = f"{OUT_DIR}/ensemble_gbm_{cat.upper()}.txt"
            booster.save_model(path)
            rec = {
                "path": path,
                "train_rows": int(len(ycat)),
                "pos": int(ycat.sum()),
                "num_boost_round": nrounds,
                "validated_best_iter": FIXED_ROUNDS[cat],
            }
            summary["boosters"][cat] = rec
            print(f"  {cat}: {rec} -> {path}", flush=True)
            if base.MLFLOW_ON:
                mlflow.log_metrics(
                    {f"{cat}_train_rows": float(len(ycat)), f"{cat}_pos": float(ycat.sum())}
                )
                mlflow.log_artifact(path, artifact_path="boosters")
            del Xcat, ycat, dtr, booster
            gc.collect()

        with open(f"{OUT_DIR}/summary.json", "w") as w:
            json.dump(summary, w, indent=2)
        if base.MLFLOW_ON:
            mlflow.log_artifact(f"{OUT_DIR}/summary.json")
            print(f"mlflow run_id={run_id}", flush=True)
            with open(f"{OUT_DIR}/mlflow_run.json", "w") as w:
                json.dump(
                    {
                        "run_id": run_id,
                        "tracking_uri": os.environ["MLFLOW_TRACKING_URI"],
                        "experiment": os.environ.get(
                            "MLFLOW_EXPERIMENT", "native-reranker-boosters"
                        ),
                        "run_name": "native-boosters-azucar-le227",
                        "url": f"{os.environ['MLFLOW_TRACKING_URI']}/#/experiments/"
                        f"_/runs/{run_id}",
                    },
                    w,
                    indent=2,
                )

    print("DONE ->", OUT_DIR, flush=True)
    return 0


class _NullCtx:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


if __name__ == "__main__":
    sys.exit(main())
