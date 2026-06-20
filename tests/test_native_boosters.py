"""Tests for src/protea_reranker_lab/native_boosters.py.

No live MinIO / DB / large parquet: a tiny synthetic parquet is written to a
tmp dir and streamed through the same memory-safe code path the production job
uses. Training runs on a handful of rows (fast, RAM-trivial).
"""

from __future__ import annotations

import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from protea_contracts import ALL_FEATURES, CATEGORICAL_FEATURES

from protea_reranker_lab import native_boosters as nb
from protea_reranker_lab.native_boosters_mlflow import MlflowLogger, mlflow_enabled


def _numeric_feats(n: int = 3) -> list[str]:
    cat = set(CATEGORICAL_FEATURES)
    return [f for f in ALL_FEATURES if f not in cat][:n]


def _write_parquet(path, feats: list[str], n_per_cat: int = 40, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    cats, labels = [], []
    cols: dict[str, list] = {f: [] for f in feats}
    for c in ("nk", "lk", "pk"):
        for i in range(n_per_cat):
            cats.append(c)
            labels.append(int(i % 2))
            for f in feats:
                cols[f].append(float(rng.random()))
    data = {f: pa.array(cols[f], type=pa.float32()) for f in feats}
    data["label"] = pa.array(labels, type=pa.int8())
    data["category"] = pa.array(cats, type=pa.string())
    pq.write_table(pa.table(data), str(path))


# ---------------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------------

def test_infer_families_default_alignments_taxonomy() -> None:
    fams = nb.infer_active_feature_families()
    assert fams == ["alignment_nw", "annotation_meta", "knn", "length", "taxonomy_pair"]


def test_feature_schema_sha_deterministic() -> None:
    assert nb.feature_schema_sha() == nb.feature_schema_sha()
    assert len(nb.feature_schema_sha()) > 0


def test_resolve_numeric_features_drops_categoricals(tmp_path) -> None:
    feats = _numeric_feats()
    pqp = tmp_path / "train.parquet"
    _write_parquet(pqp, feats)
    resolved = nb.resolve_numeric_features(pqp, None)
    assert set(resolved) == set(feats)
    assert not (set(resolved) & set(CATEGORICAL_FEATURES))


def test_resolve_numeric_features_explicit_subset(tmp_path) -> None:
    feats = _numeric_feats()
    pqp = tmp_path / "train.parquet"
    _write_parquet(pqp, feats)
    # subset intersected with present columns; bogus names dropped.
    resolved = nb.resolve_numeric_features(pqp, [feats[0], "does_not_exist"])
    assert resolved == [feats[0]]


def test_accumulate_splits_per_category(tmp_path) -> None:
    feats = _numeric_feats()
    pqp = tmp_path / "train.parquet"
    _write_parquet(pqp, feats, n_per_cat=10)
    out = nb._accumulate(pqp, feats, batch_rows=7)  # batch < total to exercise streaming
    assert set(out) == {"nk", "lk", "pk"}
    for cat in ("nk", "lk", "pk"):
        x, y = out[cat]
        assert x.shape == (10, len(feats))
        assert x.dtype == np.float32
        assert len(y) == 10


# ---------------------------------------------------------------------------
# end-to-end (tiny) training
# ---------------------------------------------------------------------------

def test_train_native_boosters_writes_artifacts(tmp_path) -> None:
    feats = _numeric_feats()
    _write_parquet(tmp_path / "train.parquet", feats, n_per_cat=60, seed=1)
    _write_parquet(tmp_path / "eval.parquet", feats, n_per_cat=30, seed=2)
    cfg = nb.NativeBoosterConfig(
        train_parquet=tmp_path / "train.parquet",
        eval_parquet=tmp_path / "eval.parquet",
        out_dir=tmp_path / "out",
        dataset_name="unit-test",
        num_boost_round=10,
        early_stopping_rounds=5,
        batch_rows=25,
    )
    summary = nb.train_native_boosters(cfg, mlflow_logger=None)

    assert summary["dataset_name"] == "unit-test"
    assert summary["feature_schema_sha"] == nb.feature_schema_sha()
    assert set(summary["boosters"]) == {"nk", "lk", "pk"}
    for cat in ("nk", "lk", "pk"):
        assert (cfg.out_dir / f"ensemble_gbm_{cat.upper()}.txt").exists()
        assert summary["boosters"][cat]["train_rows"] == 60
    assert (cfg.out_dir / "summary.json").exists()


# ---------------------------------------------------------------------------
# MLflow gating
# ---------------------------------------------------------------------------

def test_mlflow_disabled_without_uri(monkeypatch) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    assert mlflow_enabled() is False
    assert MlflowLogger.maybe_create() is None


def test_mlflow_maybe_create_swallows_import_error(monkeypatch) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    # mlflow is not installed in the core lab env -> maybe_create returns None
    # instead of raising, so training stays unbreakable.
    if _mlflow_importable():
        pytest.skip("mlflow installed; import-error path not exercised")
    assert MlflowLogger.maybe_create() is None


def _mlflow_importable() -> bool:
    try:
        import mlflow  # noqa: F401
        return True
    except Exception:
        return False
