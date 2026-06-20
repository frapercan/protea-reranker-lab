"""Optional MLflow tracking for the native-booster training job.

Entirely best-effort and gated: the trainer in :mod:`native_boosters` works
with ``mlflow_logger=None``. Construct :class:`MlflowLogger` only when an
``MLFLOW_TRACKING_URI`` is configured. Every logging call swallows its own
errors so a flaky tracking server never breaks training.

The self-hosted PROTEA-infra tracking server uses a Postgres ``mlflow`` DB +
MinIO ``mlflow`` artifact bucket; see ``storage/mlflow/README.md`` in the
Thesis2 root for the deployment.
"""

from __future__ import annotations

import contextlib
import logging
import os
from pathlib import Path
from typing import Iterator

log = logging.getLogger(__name__)


def mlflow_enabled() -> bool:
    """True iff an MLflow tracking URI is configured."""
    return bool(os.environ.get("MLFLOW_TRACKING_URI"))


class MlflowLogger:
    """Thin best-effort wrapper over an active MLflow parent run.

    Use :meth:`maybe_create` to get an instance (or ``None`` when disabled),
    then call it as a context manager so the parent run opens/closes cleanly::

        logger = MlflowLogger.maybe_create(run_name="native-boosters", params={...})
        with (logger or contextlib.nullcontext()):
            train_native_boosters(cfg, mlflow_logger=logger)
    """

    def __init__(self, mlflow_module, run_name: str) -> None:
        self._mlflow = mlflow_module
        self._run_name = run_name

    # -- construction --------------------------------------------------------
    @classmethod
    def maybe_create(
        cls,
        *,
        run_name: str = "native-boosters",
        experiment: str = "native-reranker-boosters",
        s3_endpoint: str = "http://localhost:9000",
        s3_access_key: str = "minioadmin",
        s3_secret_key: str = "minioadmin",
    ) -> "MlflowLogger | None":
        """Return a logger if MLflow is configured and importable, else None."""
        if not mlflow_enabled():
            return None
        try:
            import mlflow

            os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", s3_endpoint)
            os.environ.setdefault("AWS_ACCESS_KEY_ID", s3_access_key)
            os.environ.setdefault("AWS_SECRET_ACCESS_KEY", s3_secret_key)
            mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT", experiment))
            log.info("mlflow: logging to %s", os.environ["MLFLOW_TRACKING_URI"])
            return cls(mlflow, run_name)
        except Exception as exc:  # pragma: no cover - best-effort
            log.warning("mlflow: disabled (%s)", exc)
            return None

    # -- parent run context --------------------------------------------------
    def __enter__(self) -> "MlflowLogger":
        self._parent = self._mlflow.start_run(run_name=self._run_name)
        self._parent.__enter__()
        return self

    def __exit__(self, *exc) -> None:
        with contextlib.suppress(Exception):
            self._parent.__exit__(*exc)

    # -- logging helpers (all best-effort) -----------------------------------
    def _safe(self, fn) -> None:
        with contextlib.suppress(Exception):
            fn()

    def log_run_params(self, params: dict, tags: dict | None = None) -> None:
        self._safe(lambda: self._mlflow.log_params(params))
        if tags:
            self._safe(lambda: self._mlflow.set_tags(tags))

    def log_category_params(self, cat: str, train_rows: int, pos: int) -> None:
        self._safe(lambda: self._mlflow.log_params(
            {"category": cat, "train_rows": train_rows, "pos": pos}
        ))

    def log_category_result(self, cat: str, info: dict, path: Path) -> None:
        auc = info.get("eval_auc")
        self._safe(lambda: self._mlflow.log_metrics({
            f"{cat}_best_eval_auc": float(auc) if auc is not None else float("nan"),
            f"{cat}_best_iter": int(info["best_iter"]),
        }))
        self._safe(lambda: self._mlflow.log_artifact(str(path), artifact_path="boosters"))

    def log_summary_artifact(self, summary_path: Path) -> None:
        self._safe(lambda: self._mlflow.log_artifact(str(summary_path)))

    @contextlib.contextmanager
    def category_run(self, cat: str) -> Iterator[None]:
        """Nested run for one category (NK/LK/PK); no-op on failure."""
        try:
            with self._mlflow.start_run(run_name=f"booster-{cat}", nested=True):
                yield
        except Exception as exc:  # pragma: no cover - best-effort
            log.warning("mlflow: nested run 'booster-%s' failed (%s)", cat, exc)
            yield

    def metric_callback(self, cat: str):
        """LightGBM callback streaming per-iteration eval AUC to the run."""
        mlflow = self._mlflow

        def _cb(env) -> None:
            with contextlib.suppress(Exception):
                for ds_name, metric, value, _ in env.evaluation_result_list:
                    mlflow.log_metric(f"{cat}_{ds_name}_{metric}", value, step=env.iteration)

        return _cb
