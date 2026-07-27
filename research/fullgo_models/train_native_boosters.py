"""Train the 3 per-category (NK/LK/PK) LightGBM boosters for the NATIVE
pipeline, on the parity export parquet (73-feature schema incl. the new
classifier/self_prior/association columns). These REPLACE the offline
16-feature composite boosters (which the SchemaShaMismatchError guard rejects).

Memory-safe: the train parquet is ~54M rows x 78 cols, so a naive
``to_pandas()`` blows past RAM. Here we stream row-group batches, keep only the
needed columns as float32, and split into per-category accumulators (peak well
under the box RAM). Binary objective (the export's reranker_objective default),
early-stopping on the 227 eval split. Saves boosters + the feature_schema_sha
the live predict guard expects.
"""

from __future__ import annotations

import contextlib
import gc
import io
import json
import os
import sys

import lightgbm as lgb
import numpy as np
import pyarrow.parquet as pq
from minio import Minio

sys.path.insert(0, "/home/frapercan/Thesis2/repositories/PROTEA")
from protea_contracts import compute_feature_schema_sha  # noqa: E402
from protea_method.reranker import (  # noqa: E402
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    infer_active_feature_families,
)

import os as _os
BUCKET = "protea"
BASE = _os.environ.get("BOOSTER_DATASET", "datasets/fullgo-native-parity-SELECT-220-227")
OUT_DIR = _os.environ.get(
    "BOOSTER_OUT", "/home/frapercan/Thesis2/storage/fullgo_models/native_boosters"
)
CATS = ("nk", "lk", "pk")
PARAMS = {
    "objective": "binary", "metric": "auc", "learning_rate": 0.05,
    "num_leaves": 63, "min_data_in_leaf": 100, "feature_fraction": 0.9,
    "bagging_fraction": 0.9, "bagging_freq": 1, "verbosity": -1,
}
NUM_BOOST_ROUND = 5000
EARLY_STOP = 50
_CLIENT = Minio("localhost:9000", access_key="minioadmin", secret_key="minioadmin", secure=False)

# --- MLflow (best-effort, gated on MLFLOW_TRACKING_URI) -----------------------
# Logs to the self-hosted PROTEA-infra tracking server (Postgres `mlflow` DB +
# MinIO `mlflow` bucket). Entirely optional: if MLFLOW_TRACKING_URI is unset or
# mlflow import/connect fails, training proceeds unchanged. See
# ~/Thesis2/storage/mlflow/README.md.
MLFLOW_ON = bool(os.environ.get("MLFLOW_TRACKING_URI"))
_mlflow = None
if MLFLOW_ON:
    try:
        import mlflow as _mlflow  # noqa: F401

        # MinIO/S3 creds for artifact upload; honour any caller-provided env.
        os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
        os.environ.setdefault("AWS_ACCESS_KEY_ID", "minioadmin")
        os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "minioadmin")
        _mlflow.set_experiment(
            os.environ.get("MLFLOW_EXPERIMENT", "native-reranker-boosters")
        )
        print(f"mlflow: logging to {os.environ['MLFLOW_TRACKING_URI']}", flush=True)
    except Exception as exc:  # pragma: no cover - best-effort
        print(f"mlflow: disabled ({exc})", flush=True)
        _mlflow = None
        MLFLOW_ON = False


def _mlflow_lgb_metric_cb(cat: str):
    """LightGBM callback: stream per-iteration eval metrics to the active run.

    Best-effort; swallows any logging error so training never breaks.
    """

    def _cb(env) -> None:
        if not MLFLOW_ON:
            return
        try:
            for ds_name, metric, value, _ in env.evaluation_result_list:
                _mlflow.log_metric(f"{cat}_{ds_name}_{metric}", value, step=env.iteration)
        except Exception:
            pass

    return _cb


@contextlib.contextmanager
def _mlflow_run(run_name: str):
    """Start a nested-friendly MLflow run if enabled, else a no-op context."""
    if not MLFLOW_ON:
        yield None
        return
    try:
        # nested=True so the per-category run lives under the active parent run.
        with _mlflow.start_run(run_name=run_name, nested=True) as run:
            yield run
    except Exception as exc:  # pragma: no cover - best-effort
        print(f"mlflow: run '{run_name}' failed ({exc})", flush=True)
        yield None


def _mlflow_log(fn):
    """Run a best-effort mlflow logging call, swallowing errors."""
    if not MLFLOW_ON:
        return
    try:
        fn()
    except Exception:
        pass


def _to_float(col) -> np.ndarray:
    """Numeric column -> float32, coercing ''/None to NaN."""
    a = np.asarray(col, dtype=object)
    out = np.empty(len(a), dtype=np.float32)
    for i, v in enumerate(a):
        try:
            out[i] = float(v)
        except (TypeError, ValueError):
            out[i] = np.nan
    return out


def _accumulate(key: str, feat: list[str]):
    """Stream row groups; return {cat: (X float32, y int8)} for the parquet.

    feat is NUMERIC-only (the 4 string categoricals are dropped: the generic
    apply_reranker predict path coerces string categoricals to NaN, so training
    on them would create a train/predict mismatch).
    """
    raw = _CLIENT.get_object(BUCKET, key).read()
    pf = pq.ParquetFile(io.BytesIO(raw))
    cols = feat + ["label", "category"]
    parts: dict[str, list] = {c: [] for c in CATS}
    ys: dict[str, list] = {c: [] for c in CATS}
    n = 0
    for batch in pf.iter_batches(batch_size=500_000, columns=cols):
        d = batch.to_pydict()
        cat = np.asarray([str(x) for x in d["category"]])
        lab = np.asarray(d["label"], dtype=np.int8)
        xb = np.empty((len(cat), len(feat)), dtype=np.float32)
        for j, f in enumerate(feat):
            col = d[f]
            try:
                xb[:, j] = np.asarray(col, dtype=np.float32)
            except (TypeError, ValueError):
                xb[:, j] = _to_float(col)
        for c in CATS:
            m = cat == c
            if m.any():
                parts[c].append(xb[m])
                ys[c].append(lab[m])
        n += len(cat)
        del d, xb, cat, lab
        gc.collect()
        print(f"    ...{n} rows streamed", flush=True)
    out = {}
    for c in CATS:
        if parts[c]:
            out[c] = (np.vstack(parts[c]), np.concatenate(ys[c]))
        parts[c] = []
        gc.collect()
    del raw, pf
    gc.collect()
    return out


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    # feature columns present in the parquet, in ALL_FEATURES order.
    schema_names = pq.read_schema(
        io.BytesIO(_CLIENT.get_object(BUCKET, f"{BASE}/train.parquet").read())
    ).names
    cat_set = set(CATEGORICAL_FEATURES)
    _subset = _os.environ.get("FEATURES")
    if _subset:
        # MR-2: flat combiner over the component-score VECTOR (a handful), not 69 features.
        feat = [f for f in _subset.split(",") if f.strip() and f in schema_names]
    else:
        feat = [f for f in ALL_FEATURES if f in schema_names and f not in cat_set]
    sha = compute_feature_schema_sha(
        infer_active_feature_families(
            compute_alignments=True, compute_taxonomy=True, compute_v6_features=False
        )
    )
    print(f"numeric_features={len(feat)} (dropped {len(cat_set)} categoricals) schema_sha={sha}", flush=True)

    # Parent run: shared params for the whole 3-booster training job.
    parent_cm = (
        _mlflow.start_run(run_name=os.environ.get("MLFLOW_RUN_NAME", "native-boosters"))
        if MLFLOW_ON
        else contextlib.nullcontext()
    )
    with parent_cm:
        _mlflow_log(lambda: _mlflow.log_params({
            "booster_dataset": BASE,
            "out_dir": OUT_DIR,
            "frame": "validation 220->227 (SELECT held-out eval)",
            "objective": PARAMS["objective"],
            "metric": PARAMS["metric"],
            "learning_rate": PARAMS["learning_rate"],
            "num_leaves": PARAMS["num_leaves"],
            "min_data_in_leaf": PARAMS["min_data_in_leaf"],
            "feature_fraction": PARAMS["feature_fraction"],
            "bagging_fraction": PARAMS["bagging_fraction"],
            "bagging_freq": PARAMS["bagging_freq"],
            "num_boost_round": NUM_BOOST_ROUND,
            "early_stopping_rounds": EARLY_STOP,
            "valid_sets": "eval-only (no train-AUC)",
            "n_features": len(feat),
            "n_dropped_categoricals": len(cat_set),
            "feature_schema_sha": sha,
        }))
        _mlflow_log(lambda: _mlflow.set_tags({
            "pipeline": "native-reranker",
            "categories": ",".join(CATS),
        }))
        return _train_all(feat, sha, cat_set)


def _train_all(feat, sha, cat_set) -> int:
    print("streaming TRAIN...", flush=True)
    train = _accumulate(f"{BASE}/train.parquet", feat)
    print("streaming EVAL...", flush=True)
    ev = _accumulate(f"{BASE}/eval.parquet", feat)

    summary = {"feature_schema_sha": sha, "features": feat, "dropped_categoricals": sorted(cat_set), "boosters": {}}
    for cat in CATS:
        if cat not in train:
            print(f"!! {cat}: 0 train rows", flush=True)
            continue
        Xtr, ytr = train[cat]
        dtr = lgb.Dataset(Xtr, label=ytr, feature_name=feat, free_raw_data=False)
        # eval-only: do NOT put the (up to 50M-row) train set in valid_sets;
        # computing train-AUC every iteration dominates PK runtime. Early-stop
        # only needs the eval set. (Does not change the trained booster.)
        valid, names, cb = [], [], [lgb.log_evaluation(period=50)]
        # per-iteration eval metric streamed to MLflow (no-op if disabled).
        cb.append(_mlflow_lgb_metric_cb(cat))
        if cat in ev:
            Xev, yev = ev[cat]
            dev = lgb.Dataset(Xev, label=yev, feature_name=feat, reference=dtr, free_raw_data=False)
            valid.append(dev)
            names.append("eval")
            cb.append(lgb.early_stopping(EARLY_STOP, verbose=True))
        # one nested MLflow run per category (NK/LK/PK); no-op context if disabled.
        with _mlflow_run(f"booster-{cat}"):
            _mlflow_log(lambda c=cat: _mlflow.log_params({
                "category": c, "train_rows": int(len(ytr)), "pos": int(ytr.sum()),
            }))
            booster = lgb.train(PARAMS, dtr, num_boost_round=NUM_BOOST_ROUND, valid_sets=valid, valid_names=names, callbacks=cb)
            path = f"{OUT_DIR}/ensemble_gbm_{cat.upper()}.txt"
            booster.save_model(path)
            best = booster.best_score.get("eval", {}).get("auc") if cat in ev else None
            print(f"  {cat}: train={len(ytr)} pos={int(ytr.sum())} best_iter={booster.best_iteration} eval_auc={best} -> {path}", flush=True)
            summary["boosters"][cat] = {"path": path, "train_rows": int(len(ytr)), "pos": int(ytr.sum()), "best_iter": int(booster.best_iteration), "eval_auc": best}
            _mlflow_log(lambda b=best, it=booster.best_iteration, c=cat: _mlflow.log_metrics({
                f"{c}_best_eval_auc": float(b) if b is not None else float("nan"),
                f"{c}_best_iter": int(it),
            }))
            # log the trained booster .txt as an artifact (-> MinIO mlflow bucket).
            _mlflow_log(lambda p=path: _mlflow.log_artifact(p, artifact_path="boosters"))
        del Xtr, ytr, dtr
        train[cat] = None
        gc.collect()

    with open(f"{OUT_DIR}/summary.json", "w") as w:
        json.dump(summary, w, indent=2)
    # log the run summary as a parent-run artifact.
    _mlflow_log(lambda: _mlflow.log_artifact(f"{OUT_DIR}/summary.json"))
    print("DONE ->", OUT_DIR, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
